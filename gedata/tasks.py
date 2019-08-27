from __future__ import absolute_import, unicode_literals
from celery import shared_task
from celery_progress.backend import ProgressRecorder
import time
from datetime import datetime
from django.utils import timezone

import os
import re
import pickle
import pandas as pd

from statistics import mean
from scipy.stats import ttest_ind


from pygest import plot, algorithms
from pygest.rawdata import miscellaneous
from pygest.convenience import bids_val  # , create_symbol_to_id_map

from .models import PushResult, ResultSummary


@shared_task
def add(x, y):
    return x + y


@shared_task(bind=True)
def test_task(self, seconds):
    """ A test task to run for a given number of seconds

    :param self: available through "bind=True", allows a reference to the celery task this function becomes a part of.
    :param seconds: How many seconds to run
    """

    progress_recorder = ProgressRecorder(self)
    for i in range(seconds):
        time.sleep(1)
        progress_recorder.set_progress(i + 1, seconds)
    return 'done'


@shared_task
def clear_jobs():
    PushResult.objects.all().delete()


@shared_task(bind=True)
def collect_jobs(self, data_path="/data"):
    """ Traverse the output and populate the database with completed results.

    :param self: available through "bind=True", allows a reference to the celery task this function becomes a part of.
    :param data_path: default /data, base path to all of the results
    """

    progress_recorder = ProgressRecorder(self)

    def json_contents(json_file):
        """ Parse contents of json file into a dict

            I tried the standard json parser, but had repeated issues and failures.
            Regex works well and the code is still fairly clean.
        """
        items = {}
        with open(json_file, "r") as jf:
            for line in jf.readlines():
                clean_line = line.strip().rstrip(",").replace(": ", ":")
                m = re.match(".*\"(?P<k>.+)\":\"(?P<v>.+)\".*", clean_line)
                if m:
                    k = m.group('k')
                    v = m.group('v')
                    items[k] = v
        return items


    def seconds_elapsed(elapsed):
        parts = elapsed.split(":")
        if len(parts) != 3:
            return 0
        seconds = int(float(parts[2]))
        minutes = int(parts[1])
        if "days" in parts[0]:
            hours = int(parts[0].split(" days, ")[1])
            days = int(parts[0].split(" days, ")[0])
        elif "day" in parts[0]:
            hours = int(parts[0].split(" day, ")[1])
            days = int(parts[0].split(" day, ")[0])
        else:
            hours = int(parts[0])
            days = 0
        return seconds + (minutes * 60) + (hours * 3600) + (days * 3600 * 24)

    results = []
    i = 0
    n = 0

    # Get a list of files to parse.
    for shuffle in ['derivatives', 'shuffles', 'edgeshuffles', 'distshuffles', ]:
        d = os.path.join(data_path, shuffle)
        if os.path.isdir(d):
            for root, dirs, files in os.walk(d, topdown=True):
                for f in files:
                    if f[(-4):] == "json":
                        result = {
                            'data_dir': data_path,
                            'shuffle': shuffle,
                            'path': root,
                            'json_file': f,
                            'tsv_file': f.replace("json", "tsv"),
                            'log_file': f.replace("json", "log"),
                        }
                        results.append(result)
                        n += 1
                        progress_recorder.set_progress(i, n) #  Calculating...
                        # self.update_state(state="PROGRESS",
                        #                   meta={'current': i, 'total': i, 'message': "Discovering file {:,}".format(i)}
                        # )

    # Gather information about each result.
    for i, result in enumerate(results):
        # First, parse out the BIDS key-value pairs from the path.
        bids_pairs = []
        bids_dict = {'root': result['path'], 'name': result['json_file']}
        fs_parts = os.path.join(result['path'], result['json_file'])[: -5 :].split(os.sep)
        for fs_part in fs_parts:
            if '-' in fs_part:
                pairs = fs_part.split("_")
                for pair in pairs:
                    if '-' in pair:
                        p = pair.split("-")
                        bids_pairs.append((p[0], p[1]))
                    else:
                        # There should never be an 'extra' but we catch it to debug problems.
                        bids_pairs.append(('extra', pair))
        for bp in bids_pairs:
            bids_dict[bp[0]] = bp[1]
        result.update(bids_dict)

        # Second, parse the key-value pairs from the json file containing processing information.
        json_dict = json_contents(os.path.join(result['path'], result['json_file']))
        result.update(json_dict)

        # Finally, put it all into a model for storage in the database.
        r = PushResult(
            json_path=os.path.join(result['path'], result['json_file']),
            tsv_path=os.path.join(result['path'], result['json_file'].replace("json", "tsv")),
            log_path=os.path.join(result['path'], result['json_file'].replace("json", "log")),
            # For dates, we assume EDT (-0400) as most runs were in EDT. For those in EST, the hour makes no real difference.
            start_date=datetime.strptime(result.get("began", "1900-01-01 00:00:00") + "-0400", "%Y-%m-%d %H:%M:%S%z"),
            end_date=datetime.strptime(result.get("completed", "1900-01-01 00:00:00") + "-0400", "%Y-%m-%d %H:%M:%S%z"),
            host=result.get("host", ""),
            command=result.get("command", ""),
            version=result.get("pygest version", ""),
            duration=seconds_elapsed(result.get("elapsed", "")),
            shuffle=result.get("shuffle", ""),
            sub=result.get("sub", ""),
            hem=result.get("hem", " ").upper()[0],
            samp=result.get("samp", ""),
            prob=result.get("prob", ""),
            parby=result.get("parby", ""),
            splby=result.get("splby", ""),
            batch=result.get("batch", ""),
            tgt=result.get("tgt", ""),
            algo=result.get("algo", ""),
            norm=result.get("norm", ""),
            comp=result.get("comp", ""),
            mask=result.get("mask", ""),
            adj=result.get("adj", ""),
            seed=int(result.get("seed", "0")),
            columns=0,
            rows=0,
        )
        r.save()
        progress_recorder.set_progress(i, n)

    if PushResult.objects.count() > 0:
        s = ResultSummary(
            summary_date=timezone.now(),
            num_results=PushResult.objects.count(),
            num_actuals=PushResult.objects.filter(shuffle='derivatives').count(),
            num_shuffles=PushResult.objects.filter(shuffle='shuffles').count(),
            num_distshuffles=PushResult.objects.filter(shuffle='distshuffles').count(),
            num_edgeshuffles=PushResult.objects.filter(shuffle='edgeshuffles').count(),
            num_splits=0,
        )
        s.save()


shuffle_dir_map = {
    'none': 'derivatives',
    'edge': 'edgeshuffles',
    'dist': 'distshuffles',
    'agno': 'shuffles'
}


def comp_from_signature(signature, filename=False):
    """ Convert signature to comp string
    :param signature: The 7-character string indicating which group of results to use.
    :param filename: Set filename=True to get the comparator filename rather than its BIDS string representation.
    :return:
    """
    comp_map = {
        'HCPG': 'glasserconnectivitysim',
        'HCPW': 'hcpniftismoothgrandmeansim',
        'NKIG': 'indiglasserconnsim',
        'NKIW': 'indiconnsim',
        'glasserconnectivitysim': 'glasser-connectivity_sim.df',
        'hcpniftismoothgrandmeansim': 'hcp_niftismooth_grandmean_sim.df',
        'indiglasserconnsim': 'indi-glasser-conn_sim.df',
        'indiconnsim': 'indi-connectivity_sim.df',
    }
    if filename:
        return comp_map[comp_map[signature]]
    return comp_map[signature]


def test_score(tsv_file, base_path='/data', own_expr=False, probe_significance_threshold=0.01):
    """ Use tsv_file to figure out its complement test expression dataframe.
        Use the trained probe list to index the testing expression dataset.
        and return the new Mantel correlation """

    # Figure out where to get the test set's expression data
    batch = 'none'
    if (own_expr & ('batch-test' in tsv_file)) | ((not own_expr) & ('batch-train' in tsv_file)):
        batch = bids_val("batch", tsv_file).replace('train', 'test')
    elif (own_expr & ('batch-train' in tsv_file)) | ((not own_expr) & ('batch-test' in tsv_file)):
        batch = bids_val("batch", tsv_file).replace('test', 'train')

    # Figure out where to get data, based on which score we want.
    expr_file = os.path.join(base_path,
                             "splits", "sub-all_hem-A_samp-glasser_prob-fornito",
                             "batch-{}".format(batch),
                             "parcelby-{}_splitby-{}.df".format(bids_val('parby', tsv_file), bids_val('splby', tsv_file))
                             )
    # If comp_from_signature is given a comp BIDS string, it will return the filename of the comp file.
    comp_file = os.path.join(base_path, "conn", comp_from_signature(bids_val('comp', tsv_file)))

    # Get the actual data
    if os.path.exists(tsv_file) & os.path.exists(expr_file) & os.path.exists(comp_file):
        scoring_probes = algorithms.run_results(tsv_file, top=probe_significance_threshold)['top_probes']
        with open(expr_file, 'rb') as f:
            expr = pickle.load(f)
        with open(comp_file, 'rb') as f:
            comp = pickle.load(f)
    else:
        if not os.path.exists(tsv_file):
            print("ERR: NOT A TSV FILE: {}".format(tsv_file))
        if not os.path.exists(expr_file):
            print("ERR: NOT A EXP FILE: {}".format(expr_file))
        if not os.path.exists(comp_file):
            print("ERR: NOT A CMP FILE: {}".format(comp_file))
        return 0.0

    # And filter them both down to relevant data
    # This list MUST be in the same order as the comp columns due to 0's in upper triangle
    # print("{}, {},\n\t{} vs\n\t{} using\n\t{} probes.".format(
    #     "self" if own_expr else "othr", scoring_phase, expr_file, comp_file, tsv_file
    # ))
    overlapping_samples = [col for col in comp.columns if col in expr.columns]
    expr = expr.loc[scoring_probes, overlapping_samples]
    comp_mat = comp.loc[overlapping_samples, overlapping_samples]

    return algorithms.correlate(expr, comp_mat)


def test_overlap(tsv_file, probe_significance_threshold=0.01):
    """ Determine the percentage of overlap between the provided tsv_file and
        its complementary train/test tsv_file. """

    if 'test' in tsv_file:
        comp_file = tsv_file.replace('test', 'train')
    elif 'train' in tsv_file:
        comp_file = tsv_file.replace('train', 'test')
    else:
        print("Non train/test file: '{}'".format(tsv_file))
        return 0.0

    return algorithms.pct_similarity(
        [tsv_file, comp_file], map_probes_to_genes_first=False, top=probe_significance_threshold
    )


def dict_from_result(tsv_file, base_path, probe_significance_threshold=0.01):
    """ Return a key-value description of a single result. """

    splby = 'glasser' if 'splby-glasser' in tsv_file else 'wellid'
    parby = 'glasser' if 'parby-glasser' in tsv_file else 'wellid'

    if "batch-train" in tsv_file:
        phase = "train"
    else:
        phase = "test"

    if "/shuffles/" in tsv_file:
        shuffle = 'agno'
    elif "/distshuffles/" in tsv_file:
        shuffle = 'dist'
    elif "/edgeshuffles/" in tsv_file:
        shuffle = 'edge'
    else:
        shuffle = 'none'

    top_score = algorithms.best_score(tsv_file)

    return {
        'path': tsv_file,
        'phase': phase,
        'algo': bids_val('alg', tsv_file),
        'splby': splby,
        'parby': parby,
        'mask': bids_val('msk', tsv_file),
        'shuffle': shuffle,
        'top_score': top_score,
        'threshold': probe_significance_threshold,
        'train_score': test_score(tsv_file, base_path, own_expr=True, probe_significance_threshold=probe_significance_threshold),
        'test_score': test_score(tsv_file, base_path, own_expr=False, probe_significance_threshold=probe_significance_threshold),
        'test_overlap': test_overlap(tsv_file, probe_significance_threshold=probe_significance_threshold),
    }


def combine_gene_ranks(tsv_files):
    """ Go through all genes in the list of tsv_files and generate a single ranking. """

    gene_ranks = {}
    for tsv in tsv_files:
        df = pd.read_csv(tsv, sep='\t', index_col=0)
        for row in df.itertuples():
            if row.probe_id in gene_ranks.keys():
                gene_ranks[row.probe_id]['appearances'] += 1
                gene_ranks[row.probe_id]['ranks'].append(row.Index)
            else:
                gene_ranks[row.probe_id] = {
                    'appearances': 1,
                    'ranks': [row.Index, ],
                }
    rank_df = pd.DataFrame.from_dict(gene_ranks, orient='index')
    rank_df['rank'] = rank_df['ranks'].apply(mean)

    return rank_df


def pervasive_probes(tsvs, top):
    """ Go through the files provided, at the threshold specified, and report probes in all files. """

    print("    These results average {:0.2%} overlap.".format(
        algorithms.pct_similarity(tsvs, map_probes_to_genes_first=False, top=top)
    ))
    hitters = {}
    for i, tsv in enumerate(tsvs):
        if i == 0:
            hitters = set(algorithms.run_results(tsv, top=top)['top_probes'])
        else:
            hitters = hitters.intersection(set(algorithms.run_results(tsv, top=top)['top_probes']))
        if i == len(tsvs):
            print("    {} probes remain in all {} results.".format(len(hitters), i + 1))
    winners = pd.DataFrame({'probe_id': list(hitters)})
    winners['entrez_id'] = winners['probe_id'].map(miscellaneous.map_pid_to_eid_fornito)
    return winners


def ranked_probes(tsvs, top):
    """ Go through the files provided, at the threshold specified, and report probes in all files. """

    print("    These results average {:0.2%} overlap.".format(
        algorithms.pct_similarity(tsvs, map_probes_to_genes_first=False, top=top)
    ))
    all_rankings = pd.DataFrame()
    for i, tsv in enumerate(tsvs):
        df = pd.read_csv(tsv, sep='\t')
        rankings = pd.Series(data=df.index, index=df['probe_id'], name="rank{:03d}".format(i))
        if i == 0:
            all_rankings = pd.DataFrame(data=rankings)
        else:
            all_rankings[rankings.name] = rankings
        if i == len(tsvs):
            print("    ranked all probes in {} results.".format(i + 1))
    all_rankings['mean'] = all_rankings.mean(axis=1)
    all_rankings['entrez_id'] = all_rankings.index.map(miscellaneous.map_pid_to_eid_fornito)
    return all_rankings.sort_values('mean', ascending=True)


def describe_three_relevant_overlaps(relevant_results, phase, threshold):
    """ Filter results by the arguments, and report percent similarity and consistent top genes. """
    print("=== {} ===".format(phase))
    in_shuf = (relevant_results['shuffle'] == 'none')
    more_relevant_results = relevant_results[in_shuf]
    print("  {:,} out of {:,} results are relevant here.".format(len(more_relevant_results), len(relevant_results)))

    print("  Overlap between 16 random (train) halves, @{} = {:0.1%}".format(
        threshold, algorithms.pct_similarity(
            list(more_relevant_results[more_relevant_results['phase'] == 'train']['path']),
            map_probes_to_genes_first=False, top=threshold
        )
    ))
    print("  Overlap between 16 random (test) halves, @{} = {:0.1%}".format(
        threshold, algorithms.pct_similarity(
            list(more_relevant_results[more_relevant_results['phase'] == 'test']['path']),
            map_probes_to_genes_first=False, top=threshold
        )
    ))
    acrosses = []
    for t in more_relevant_results[more_relevant_results['phase'] == 'train']['path']:
        comps = [t, t.replace('train', 'test'), ]
        olap = algorithms.pct_similarity(comps, map_probes_to_genes_first=False, top=threshold)
        acrosses.append(olap)
        # print("    {:0.2%}% - {}".format(olap, t))
    print("  Overlap between each direct train-vs-test pair @{} = {:0.1%}".format(
        threshold, mean(acrosses)
    ))
    return more_relevant_results


@shared_task(bind=True)
def build_plot(self, plot_descriptor, data_path="/data", threshold=0.01):
    """ Traverse the output and populate the database with completed results.

    :param self: Allows interacting with celery
    :param plot_descriptor: Abbreviated string describing plot's underlying data
    :param data_path: default /data, base path to all of the results
    :param threshold: At which point in the ranking is a gene deemed relevant?
    """

    progress_recorder = ProgressRecorder(self)

    comp = comp_from_signature(plot_descriptor.upper()[:4])
    parby = "glasser" if plot_descriptor[3].upper() == "G" else "wellid"
    splby = "glasser" if plot_descriptor[4].upper() == "G" else "wellid"
    mask = 'none' if plot_descriptor[5:7] == "00" else plot_descriptor[5:7]
    algo = 'smrt'
    phase = 'train'

    print("Seeking comp of " + plot_descriptor.upper() + ", resolves to " + comp + ".")

    relevant_results_queryset = PushResult.objects.filter(
        samp="glasser", prob="fornito", algo="smrt", comp=comp, parby=parby, splby=splby, mask=mask,
    )
    n = len(relevant_results_queryset)
    progress_recorder.set_progress(0, n + 100, "Finding results")

    print("Found {:,} results ({} {} {} {} {} {} {})".format(
        n, "glasser", "fornito", algo, comp, parby, splby, mask,
    ))

    if len(relevant_results_queryset) > 0:
        # sym_id_map, hg = create_symbol_to_id_map()
        # id_sym_map = hg['Symbol'].to_dict()

        relevant_results = []
        for i, path in enumerate(relevant_results_queryset.values('tsv_path')):
            if os.path.isfile(path['tsv_path']):
                relevant_results.append(dict_from_result(path['tsv_path'], base_path=data_path))
                progress_recorder.set_progress(i + 1, n + 100, "Processing {:,}/{:,} results".format(i, n))
            else:
                print("ERR: DOES NOT EXIST: {}".format(path['tsv_path']))
        rdf = pd.DataFrame(relevant_results)

        progress_recorder.set_progress(n, n + 100, "Generating plot")
        f_train_test, (a1, a2, a3, a4) = plot.plot_train_vs_test(
            rdf,
            title="{}s, split by {}, {}-masked, {}-ranked, by {}".format(parby, splby, mask, algo, comp),
            fig_size=(14, 8), ymin=-0.10, ymax=0.75
        )

        # data_path should get into the PYGEST_DATA area, which is symbolically linked to /static, so just one write.
        f_train_test.savefig(os.path.join(data_path, "plots", "train_test_{}.png".format(plot_descriptor.lower())))

        """ Write out relevant gene lists as html. """
        progress_recorder.set_progress(n+20, n + 100, "Ranking genes")
        relevant_tsvs = list(describe_three_relevant_overlaps(rdf, phase, threshold)['path'])
        survivors = pervasive_probes(relevant_tsvs, threshold)
        all_ranked = ranked_probes(relevant_tsvs, threshold)
        all_ranked['rank'] = range(1, len(all_ranked.index) + 1, 1)
        with open(os.path.join(data_path, "plots", "train_test_{}.html".format(plot_descriptor.lower())), "wt") as f:
            f.write("<p>In test, real data Mantel correlations differ from shuffled data:\n  <ol>\n")
            for shf in ['edge', 'dist', 'agno']:
                t, p = ttest_ind(
                    # 'test_score' is in the test set, could do 'train_score', too
                    rdf[rdf['shuffle'] == 'none']['test_score'].values,
                    rdf[rdf['shuffle'] == shf]['test_score'].values,
                )
                f.write("    <li>{}: t = {}, p = {}</li>\n".format(shf, t, p))
            f.write("  </ol>\n</p>\n")

            """ Next, for the description file, report top genes. """
            progress_recorder.set_progress(n + 50, n + 100, "Summarizing genes")
            line = "Pure {} {}-ranked by-{}, mask={}".format(phase, algo, splby, mask)
            f.write("<p>" + line + "\n  <ol>\n")
            # print(line)
            for p in (list(all_ranked.index[0:20]) + [x for x in survivors['probe_id'] if x not in all_ranked.index[0:20]]):
                asterisk = " *" if p in list(survivors['probe_id']) else ""
                item_string = "probe {} -> gene {}, mean rank = {:0.1f}{}".format(
                    p, all_ranked.loc[p, 'entrez_id'], all_ranked.loc[p, 'mean'] + 1.0, asterisk
                )
                f.write("    <li value=\"{}\">{}</li>\n".format(all_ranked.loc[p, 'rank'], item_string))
                # print("{}. {}".format(all_ranked.loc[p, 'rank'], item_string))
            f.write("  </ol>\n</p>\n")
            f.write("<div id=\"notes_below\">")
            f.write("    <p>Asterisks indicate probes that survived to the top 1% in all 16 splits.</p>")
            f.write("</div>")

        """ Write out ranked gene list as text. 
        with open(os.path.join(data_path, "plots", "genes-from-{}_top1pct.txt".format(plot_descriptor.lower())), "wt") as f:
            for p in list(all_ranked.index):
                if p in id_sym_map.keys():
                    f.write(id_sym_map[p])
                else:
                    f.write("( " + str(p) + " )")
        """

        progress_recorder.set_progress(n + 100, n + 100, "Finished")
