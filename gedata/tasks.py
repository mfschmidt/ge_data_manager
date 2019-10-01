from __future__ import absolute_import, unicode_literals
from celery import shared_task
from celery_progress.backend import ProgressRecorder

import time
from datetime import datetime
import pytz

from django.utils import timezone as django_timezone

import os
import re
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import logging

from matplotlib import pyplot as plt
import matplotlib.patches as patches

from statistics import mean
from scipy.stats import ttest_ind

from pygest import plot, algorithms
from pygest.rawdata import miscellaneous
from pygest.convenience import bids_val, create_symbol_to_id_map, create_id_to_symbol_map, p_string
import pygest as ge

from .models import PushResult, ResultSummary


class NullHandler(logging.Handler):
    def emit(self, record):
        pass
null_handler = NullHandler()

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


def tz_aware_file_mtime(path):
    return pytz.timezone("America/New_York").localize(
        datetime.fromtimestamp(os.path.getmtime(path))
    )


@shared_task
def clear_jobs():
    PushResult.objects.all().delete()


@shared_task(bind=True)
def collect_jobs(self, data_path="/data", rebuild=False):
    """ Traverse the output and populate the database with completed results.

    :param self: available through "bind=True", allows a reference to the celery task this function becomes a part of.
    :param data_path: default /data, base path to all of the results
    :param rebuild: set to True to clear the entire database and build results from scratch, otherwise just updates
    """

    progress_recorder = ProgressRecorder(self)

    if ResultSummary.objects.count() > 0:
        last_result_datetime = ResultSummary.objects.latest('summary_date').summary_date
    else:
        last_result_datetime = ResultSummary.empty().summary_date

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
                        if rebuild or (tz_aware_file_mtime(os.path.join(root, f)) > last_result_datetime):
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
                            progress_recorder.set_progress(i, n)

    # Gather information about each result.
    for i, result in enumerate(results):
        # First, parse out the BIDS key-value pairs from the path.
        bids_pairs = []
        bids_dict = {'root': result['path'], 'name': result['json_file']}
        fs_parts = os.path.join(result['path'], result['json_file'])[: -5:].split(os.sep)
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
            summary_date=django_timezone.now(),
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
        'hcpg': 'glasserconnectivitysim',
        'hcpw': 'hcpniftismoothgrandmeansim',
        'nkig': 'indiglasserconnsim',
        'nkiw': 'indiconnsim',
        'glasserconnectivitysim': 'glasser-connectivity_sim.df',
        'hcpniftismoothgrandmeansim': 'hcp_niftismooth_grandmean_sim.df',
        'indiglasserconnsim': 'indi-glasser-conn_sim.df',
        'indiconnsim': 'indi-connectivity_sim.df',
    }
    if filename:
        return comp_map[comp_map[signature.lower()]]
    return comp_map[signature.lower()]


def one_mask(df, mask_type, sample_type, data_dir="/data"):
    """ return a vector of booleans from the lower triangle of a matching-matrix based on 'mask_type'

    :param df: pandas.DataFrame with samples as columns
    :param str mask_type: A list of strings to specify matching masks, or a minimum distance to mask out
    :param str sample_type: Samples can be 'wellid' or 'parcelid'
    :param data_dir: The root of the path to all pygest data
    :return: Boolean 1-D vector to remove items (False values in mask) from any sample x sample triangle vector
    """

    # If mask is a number, use it as a distance filter
    data = ge.Data(data_dir, null_handler)
    try:
        # Too-short values to mask out are False, keepers are True.
        min_dist = float(mask_type)
        mask_vector = np.array(
            data.distance_vector(df.columns, sample_type=sample_type) > min_dist,
            dtype=bool
        )
        print("        masking out {:,} of {:,} edges closer than {}mm apart.".format(
            np.count_nonzero(np.invert(mask_vector)), len(mask_vector), min_dist
        ))
        return mask_vector
    except TypeError:
        pass
    except ValueError:
        pass

    # Mask is not a number, see if it's a pickled dataframe
    if os.path.isfile(mask_type):
        with open(mask_type, 'rb') as f:
            mask_df = pickle.load(f)
        if isinstance(mask_df, pd.DataFrame):
            # Note what we started with so we can report after we tweak the dataframe.
            # Too-variant values to mask out are False, keepers are True.
            orig_vector = mask_df.values[np.tril_indices(n=mask_df.shape[0], k=-1)]
            orig_falses = np.count_nonzero(~orig_vector)
            orig_length = len(orig_vector)
            print("  M  found {} containing {:,} x {:,} mask".format(mask_type, mask_df.shape[0], mask_df.shape[1]))
            print("  M    generating {:,}-len vector with {:,} False values to mask.".format(orig_length, orig_falses))

            # We can only use well_ids found in BOTH df and our new mask, make shapes match.
            unmasked_ids = [well_id for well_id in df.columns if well_id not in mask_df.columns]
            usable_ids = [well_id for well_id in df.columns if well_id in mask_df.columns]
            usable_df = mask_df.reindex(index=usable_ids, columns=usable_ids)
            usable_vector = usable_df.values[np.tril_indices(n=len(usable_ids), k=-1)]
            usable_falses = np.count_nonzero(~usable_vector)
            usable_length = len(usable_vector)
            print("  M  {:,} well_ids not found in the mask; padding with Falses.".format(len(unmasked_ids)))

            pad_rows = pd.DataFrame(np.zeros((len(unmasked_ids), len(mask_df.columns)), dtype=bool),
                                    columns=mask_df.columns, index=unmasked_ids)
            mask_df = pd.concat([mask_df, pad_rows], axis=0)
            pad_cols = pd.DataFrame(np.zeros((len(mask_df.index), len(unmasked_ids)), dtype=bool),
                                    columns=unmasked_ids, index=mask_df.index)
            mask_df = pd.concat([mask_df, pad_cols], axis=1)
            mask_vector = mask_df.values[np.tril_indices(n=mask_df.shape[0], k=-1)]
            mask_falses = np.count_nonzero(~mask_vector)
            mask_trues = np.count_nonzero(mask_vector)
            print("  M  padded mask matrix out to {:,} x {:,}".format(
                mask_df.shape[0], mask_df.shape[1]
            ))
            print("  M    with {:,} True, {:,} False, {:,} NaNs in triangle.".format(
                mask_trues, mask_falses, np.count_nonzero(np.isnan(mask_vector))
            ))

            shaped_mask_df = mask_df.reindex(index=df.columns, columns=df.columns)
            shaped_vector = shaped_mask_df.values[np.tril_indices(n=len(df.columns), k=-1)]
            print("  M  masking out {:,} (orig {:,}, {:,} usable) hi-var".format(
                np.count_nonzero(~shaped_vector), orig_falses, usable_falses,
            ))
            print("  M    of {:,} (orig {:,}, {:,} usable) edges.".format(
                len(shaped_vector), orig_length, usable_length
            ))
            return shaped_vector
        else:
            print("  M  {} is a file, but not a pickled dataframe. Skipping this mask.".format(mask_type))
            do_nothing_mask = np.ones((len(df.columns), len(df.columns)), dtype=bool)
            return do_nothing_mask[np.tril_indices(n=len(df.columns), k=-1)]

    # Mask is not a number, so treat it as a matching filter
    if mask_type[:4] == 'none':
        items = list(df.columns)
    elif mask_type[:4] == 'fine':
        items = data.samples(samples=df.columns)['fine_name']
    elif mask_type[:6] == 'coarse':
        items = data.samples(samples=df.columns)['coarse_name']
    else:
        items = data.samples(samples=df.columns)['structure_name']
    mask_array = np.ndarray((len(items), len(items)), dtype=bool)

    # There is, potentially, a nice vectorized way to mark matching values as True, but I can't find it.
    # So, looping works and is easy to read, although it might cost us a few extra ms.
    for i, y in enumerate(items):
        for j, x in enumerate(items):
            # Generate one edge of the match matrix
            mask_array[i][j] = True if mask_type == 'none' else (x != y)
    mask_vector = mask_array[np.tril_indices(n=mask_array.shape[0], k=-1)]

    print("        masking out {:,} of {:,} '{}' edges.".format(
        sum(np.invert(mask_vector)), len(mask_vector), mask_type
    ))

    # if len(mask_vector) == 0:
    #     mask_vector = np.ones(int(len(df.columns) * (len(df.columns) - 1) / 2), dtype=bool)

    return mask_vector


def test_score(tsv_file, base_path='/data', own_expr=False, mask='none', probe_significance_threshold=0.01):
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
                             "parcelby-{}_splitby-{}.df".format(bids_val('parby', tsv_file),
                                                                bids_val('splby', tsv_file))
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

    """ Only correlate the top genes, as specified by probe_significance_threshold. """
    # And filter them both down to relevant data
    # This list MUST be in the same order as the comp columns due to 0's in upper triangle
    # print("{}, {},\n\t{} vs\n\t{} using\n\t{} probes.".format(
    #     "self" if own_expr else "othr", scoring_phase, expr_file, comp_file, tsv_file
    # ))
    overlapping_samples = [col for col in comp.columns if col in expr.columns]
    expr = expr.loc[scoring_probes, overlapping_samples]
    comp = comp.loc[overlapping_samples, overlapping_samples]
    expr_mat = np.corrcoef(expr, rowvar=False)
    comp_mat = comp.values
    expr_vec = expr_mat[np.tril_indices(n=expr_mat.shape[0], k=-1)]
    comp_vec = comp_mat[np.tril_indices(n=comp_mat.shape[0], k=-1)]

    if mask != "none":
        v_mask = one_mask(expr, mask, bids_val('parby', tsv_file), base_path)
        expr_vec = expr_vec[v_mask]
        comp_vec = comp_vec[v_mask]

    # return algorithms.correlate(expr, comp_mat)
    return np.corrcoef(expr_vec, comp_vec)[0, 1]


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


def dict_from_result(tsv_file, base_path, probe_sig_threshold=0.01, force_recalc=False):
    """ Return a key-value description of a single result. """

    # These calculations can be expensive when run a lot. Load a cached copy if possible.
    analyzed_path = tsv_file.replace(".tsv", ".top-{}.dict".format(probe_sig_threshold))
    if not force_recalc:
        if os.path.isfile(analyzed_path):
            with open(analyzed_path, 'rb') as f:
                return pickle.load(f)

    # There's no cached copy; go ahead and calculate everything.
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
    initial_score = algorithms.initial_score(tsv_file)
    mask = bids_val('mask', tsv_file)
    algo = bids_val('algo', tsv_file)

    result_dict = {
        # Characteristics available in the path, if necessary
        'path': tsv_file,
        'phase': phase,
        'algo': algo,
        'splby': splby,
        'parby': parby,
        'mask': mask,
        'shuffle': shuffle,
        # Neither path nor calculation, the threshold we use for calculations
        'threshold': probe_sig_threshold,
        # Calculations on the data within the tsv file
        'top_score': top_score,
        'initial_score': initial_score,
        'train_score': test_score(
            tsv_file, base_path, own_expr=True, mask='none', probe_significance_threshold=probe_sig_threshold),
        'test_score': test_score(
            tsv_file, base_path, own_expr=False, mask='none', probe_significance_threshold=probe_sig_threshold),
        'masked_train_score': test_score(
            tsv_file, base_path, own_expr=True, mask=mask, probe_significance_threshold=probe_sig_threshold),
        'masked_test_score': test_score(
            tsv_file, base_path, own_expr=False, mask=mask, probe_significance_threshold=probe_sig_threshold),
        'test_overlap': test_overlap(tsv_file, probe_significance_threshold=probe_sig_threshold),
    }

    # Cache results to prevent repeated recalculation of the same thing.
    with open(analyzed_path, 'wb') as f:
        pickle.dump(result_dict, f)

    return result_dict


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


def plot_all_train_vs_test(df, title="Title", fig_size=(8, 8), ymin=None, ymax=None):
    """ Plot everything from initial distributions, through training curves, to training outcomes.
        Then the results of using discovered genes in train and test sets both complete and masked.
        Then even report overlap internal to each cluster of differing gene lists.

        :param pandas.DataFrame df:
        :param str title: The title to put on the top of the whole plot
        :param fig_size: A tuple of inches across x inches high
        :param ymin: Hard code the bottom of the y-axis
        :param ymax: Hard code the top of the y-axis
    """

    # Calculate (or blindly accept) the range of the y-axis, which must be the same for all four axes.
    if (ymax is None) and (len(df.index) > 0):
        highest_possible_score = max(
            max(df['top_score']),
            max(df['train_score']), max(df['test_score']),
            max(df['masked_train_score']), max(df['masked_test_score']),
        )
    else:
        highest_possible_score = ymax
    if (ymin is None) and (len(df.index) > 0):
        lowest_possible_score = min(
            min(df['top_score']),
            min(df['train_score']), min(df['test_score']),
            min(df['masked_train_score']), min(df['masked_test_score']),
        )
    else:
        lowest_possible_score = ymin

    """ Plot the first pane, rising lines representing rising Mantel correlations as probes are dropped. """
    a = df.loc[df['shuffle'] == 'none', 'path']
    b = df.loc[df['shuffle'] == 'edge', 'path']
    c = df.loc[df['shuffle'] == 'dist', 'path']
    d = df.loc[df['shuffle'] == 'agno', 'path']
    fig, ax_curve = plot.push_plot([
        {'files': list(d), 'linestyle': ':', 'color': 'green'},
        {'files': list(c), 'linestyle': ':', 'color': 'red'},
        {'files': list(b), 'linestyle': ':', 'color': 'orchid'},
        {'files': list(a), 'linestyle': '-', 'color': 'black'}, ],
        # title="Split-half train vs test results",
        label_keys=['shuffle'],
        fig_size=fig_size,
        title="",
        plot_overlaps=False,
    )

    def box_and_swarm(figure, placement, label, y, data, ps=True):
        """ Create an axes object with a swarm plot draw over a box plot of the same data. """

        shuffle_order = ['none', 'edge', 'dist', 'agno']
        shuffle_color_boxes = sns.color_palette(['gray', 'orchid', 'red', 'green'])
        shuffle_color_points = sns.color_palette(['black', 'orchid', 'red', 'green'])
        annot_columns = [
            {'shuffle': 'none', 'xo': 0.0, 'xp': 0.0},
            {'shuffle': 'edge', 'xo': 1.0, 'xp': 0.5},
            {'shuffle': 'dist', 'xo': 2.0, 'xp': 1.0},
            {'shuffle': 'agno', 'xo': 3.0, 'xp': 1.5},
        ]

        ax = figure.add_axes(placement, label=label)
        sns.boxplot(data=data, x='shuffle', y=y, order=shuffle_order, palette=shuffle_color_boxes, ax=ax)
        sns.swarmplot(data=data, x='shuffle', y=y, order=shuffle_order, palette=shuffle_color_points, ax=ax)
        ax.set_ylabel(None)
        ax.set_xlabel(label)
        ax.set_ylim(ax_curve.get_ylim())

        """ Calculate overlaps and p-values for each column in the above plots, and annotate accordingly. """
        if ps:
            gap = 0.06
            actual_results = data[data['shuffle'] == 'none'][y].values
            try:
                global_max_y = max(data[y].values)
            except ValueError:
                global_max_y = highest_possible_score
            for i, col in enumerate(annot_columns):
                shuffle_results = data[data['shuffle'] == col['shuffle']]
                try:
                    # max_y = max(data[data['phase'] == 'train'][y].values)
                    local_max_y = max(shuffle_results[y].values)
                except ValueError:
                    local_max_y = highest_possible_score
                try:
                    y_pval = max(max(shuffle_results[y].values), max(actual_results)) + gap
                except ValueError:
                    y_pval = highest_possible_score + gap
                try:
                    t, p = ttest_ind(actual_results, shuffle_results[y].values)
                    # print("    plotting, full p = {}".format(p))
                    p_annotation = p_string(p, use_asterisks=False)
                except TypeError:
                    p_annotation = "p N/A"

                # y_pline = y_pval + 0.01 + (gap * i)
                y_pline = global_max_y + 0.01 + (i * gap)
                if i > 0:
                    ax.hlines(y_pline, 0.0, col['xo'], colors='gray', linewidth=1)
                    ax.vlines(0.0, y_pval, y_pline, colors='gray', linewidth=1)
                    ax.vlines(col['xo'], local_max_y + gap, y_pline, colors='gray', linewidth=1)
                    ax.text(gap + (i * 0.01), y_pline + 0.01, p_annotation, ha='left', va='bottom')

        return ax

    margin = 0.04
    row_height = 0.40

    """ Top Row """

    box_width = 0.14
    x_left = margin
    y_base = 1.0 - (2 * margin) - row_height
    fig.text(x_left, 1.0 - (2 * margin) + 0.01, "A) Training on altered training half", ha='left', va='bottom', fontsize=12)
    """ Rising training curve plot """
    ax_curve.set_position([x_left + box_width + margin, y_base, 1.0 - (5 * margin) - (3 * box_width), row_height])
    ax_curve.set_label('rise')
    ax_curve.set_xlabel('Training')
    # The top of the plot must be at least 0.25 higher than the highest value to make room for p-values.
    ax_curve.set_ylim(bottom=lowest_possible_score, top=highest_possible_score + 0.25)

    """ Initial box and swarm plots """
    ax_pre = box_and_swarm(fig, [x_left, y_base, box_width, row_height], 'Complete Mantel', 'initial_score', df)
    ax_pre.yaxis.tick_right()
    ax_pre.set_yticklabels([])
    ax_pre.set_ylabel('Mantel Correlation')
    ax_post = box_and_swarm(fig, [1.0 - (2 * (margin + box_width)), y_base, box_width, row_height], 'Peak Mantel', 'top_score', df)

    """ Train box and swarm overlap plots """
    x_left = 1.0 - margin - box_width
    fig.text(x_left, 1.0 - (2 * margin) + 0.01, "B) Overlapping genes", ha='left', va='bottom', fontsize=12)
    ax_train_overlap = box_and_swarm(fig, [x_left, y_base, box_width, row_height], 'Gene overlap', 'train_overlap', df)
    ax_train_overlap.set_ylabel('Gene agreement')
    # TODO: Set y-axis to percentages to distinguish that we've changed measures, here.

    """ Bottom Row """

    box_width = 0.20
    x_left = margin
    y_base = x_left  # + 0.40 height makes the top at 0.45
    fig.text(x_left, margin + row_height + 0.01, "C) Testing in unshuffled halves", ha='left', va='bottom', fontsize=12)
    """ Train box and swarm plots """
    ax_train_complete = box_and_swarm(
        fig, [x_left + (0 * (margin + box_width)), y_base, box_width, row_height], 'Train complete', 'train_score', df
    )
    ax_train_complete.yaxis.tick_right()
    ax_train_complete.set_yticklabels([])
    ax_train_complete.set_ylabel('Mantel Correlation')
    ax_train_masked = box_and_swarm(
        fig, [x_left + (1 * (margin + box_width)), y_base, box_width, row_height], 'Train masked', 'masked_train_score', df
    )

    """ Test box and swarm plots """
    ax_test_complete = box_and_swarm(
        fig, [x_left + (2 * (margin + box_width)), y_base, box_width, row_height], 'Test complete', 'test_score', df
    )
    ax_test_complete.yaxis.tick_right()
    ax_test_complete.set_yticklabels([])
    ax_test_complete.set_ylabel('Mantel Correlation')
    ax_test_masked = box_and_swarm(
        fig, [x_left + (3 * (margin + box_width)), y_base, box_width, row_height], 'Test masked', 'masked_test_score', df
    )


    fig.text(0.50, 0.99, title, ha='center', va='top', fontsize=14)

    return fig, (ax_pre, ax_curve, ax_post,
                 ax_train_complete, ax_train_masked, ax_test_complete, ax_test_masked,
                 ax_train_overlap,
                 )


def interpret_descriptor(plot_descriptor):
    """ Parse the plot descriptor into parts """
    comp = comp_from_signature(plot_descriptor[:4])
    parby = "glasser" if plot_descriptor[3].lower() == "g" else "wellid"
    splby = "glasser" if plot_descriptor[4].lower() == "g" else "wellid"
    mask = 'none' if plot_descriptor[5:7] == "00" else plot_descriptor[5:7]
    algo = 'once' if plot_descriptor[7] == "o" else "smrt"
    phase = 'train'
    opposite_phase = 'test'

    relevant_results_queryset = PushResult.objects.filter(
        samp="glasser", prob="fornito", algo=algo, comp=comp, parby=parby, splby=splby, mask=mask,
        batch__startswith=phase,
    )

    return comp, parby, splby, mask, algo, phase, opposite_phase, relevant_results_queryset


@shared_task(bind=True)
def build_plot(self, plot_descriptor, data_root="/data", threshold=0.01):
    """ Traverse the output and populate the database with completed results.

    :param self: Allows interacting with celery
    :param plot_descriptor: Abbreviated string describing plot's underlying data
    :param data_root: default /data, base path to all of the results
    :param threshold: At which point in the ranking is a gene deemed relevant?
    """

    progress_recorder = ProgressRecorder(self)

    comp, parby, splby, mask, algo, phase, opposite_phase, results = interpret_descriptor(plot_descriptor)

    n = len(results)
    progress_recorder.set_progress(0, n + 100, "Finding results")

    print("Found {:,} results ({} {} {} {} {} {} {})".format(
        n, "glasser", "fornito", algo, comp, parby, splby, mask,
    ))

    if len(results) > 0:
        relevant_results = []

        """ Calculate (or load) stats for individual tsv files. """
        for i, path in enumerate(results.values('tsv_path')):
            if os.path.isfile(path['tsv_path']):
                relevant_results.append(
                    dict_from_result(path['tsv_path'], base_path=data_root, probe_sig_threshold=threshold)
                )
                progress_recorder.set_progress(i + 1, n + 100, "Processing {:,}/{:,} results".format(i, n))
            else:
                print("ERR: DOES NOT EXIST: {}".format(path['tsv_path']))
        rdf = pd.DataFrame(relevant_results)

        """ Calculate grouped stats, overlap between each tsv file and its shuffle-based 'peers'. """
        for shuffle in list(set(rdf['shuffle'])):
            shuffle_mask = rdf['shuffle'] == shuffle
            # We can only do this in training data, unless we want to double the workload above for test, too.
            rdf.loc[shuffle_mask, 'train_overlap'] = algorithms.pct_similarity_list(list(rdf.loc[shuffle_mask, 'path']))

        rdf.to_pickle(os.path.join(data_root, "plots", "pickled_dataframe_for_{}.df".format(plot_descriptor.lower())))

        progress_recorder.set_progress(n, n + 100, "Generating plot")
        f_train_test, axes = plot_all_train_vs_test(
            rdf, title="{}s, split by {}, {}-masked, {}-ranked, by {}".format(parby, splby, mask, algo, comp),
            fig_size=(12, 12), ymin=-0.15, ymax=0.90
        )
        # data_path should get into the PYGEST_DATA area, which is symbolically linked to /static, so just one write.
        f_train_test.savefig(os.path.join(data_root, "plots", "train_test_{}.png".format(plot_descriptor.lower())))

        progress_recorder.set_progress(n + 20, n + 100, "Building probe to gene map")
        sym_id_map = create_symbol_to_id_map()
        id_sym_map = create_id_to_symbol_map()

        """ Write out relevant gene lists as html. """
        progress_recorder.set_progress(n + 50, n + 100, "Ranking genes")
        relevant_tsvs = list(describe_three_relevant_overlaps(rdf, phase, threshold)['path'])
        survivors = pervasive_probes(relevant_tsvs, threshold)
        all_ranked = ranked_probes(relevant_tsvs, threshold)
        all_ranked['rank'] = range(1, len(all_ranked.index) + 1, 1)
        with open(os.path.join(data_root, "plots", "train_test_{}.html".format(plot_descriptor.lower())), "wt") as f:
            f.write("<p><span class=\"heavy\">Mantel correlations in independent test data.</span> ")
            f.write("Gene lists from training on real data result in higher correlations in real, independent \n")
            f.write("test data than gene lists from shuffled data.\n</p>")
            f.write("  <ol>\n")
            for shf in ['edge', 'dist', 'agno']:
                t, p = ttest_ind(
                    # 'test_score' is in the test set, could do 'train_score', too
                    rdf[rdf['shuffle'] == 'none']['test_score'].values,
                    rdf[rdf['shuffle'] == shf]['test_score'].values,
                )
                f.write("    <li>{}: t = {:0.2f}, p = {:0.10f}</li>\n".format(shf, t, p))
            f.write("  </ol>\n")

            """ Next, for the description file, report top genes. """
            progress_recorder.set_progress(n + 70, n + 100, "Summarizing genes")
            line = "Top genes from {}, {}-ranked by-{}, mask={}".format(phase, algo, splby, mask)
            f.write("<p>" + line + "\n  <ol>\n")
            # print(line)
            for p in (list(all_ranked.index[0:20]) + [x for x in survivors['probe_id'] if
                                                      x not in all_ranked.index[0:20]]):
                asterisk = " *" if p in list(survivors['probe_id']) else ""
                gene_id = all_ranked.loc[p, 'entrez_id']
                gene_id_string = "<a href=\"https://www.ncbi.nlm.nih.gov/gene/{g}\" target=\"_blank\">{g}</a>".format(
                    g=gene_id
                )
                if gene_id in id_sym_map:
                    gene_symbol = id_sym_map[gene_id]
                elif gene_id in sym_id_map:
                    gene_symbol = sym_id_map[gene_id]
                else:
                    gene_symbol = "not_found"
                gene_symbol_string = "<a href=\"https://www.ncbi.nlm.nih.gov/gene/?term={g}\" target=\"_blank\">{g}</a>".format(
                    g=gene_symbol
                )
                item_string = "probe {} -> entrez {} -> gene {}, mean rank {:0.1f}{}".format(
                    p, gene_id_string, gene_symbol_string, all_ranked.loc[p, 'mean'] + 1.0, asterisk
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

    else:
        with open(os.path.join(data_root, "plots", "train_test_{}.html".format(plot_descriptor.lower())), "wt") as f:
            f.write("<p>No results for {}</p>\n".format(plot_descriptor.lower()))
        progress_recorder.set_progress(n + 100, n + 100, "No results")


def plot_scores_over_thresholds(relevant_results, phase, shuffle):
    """ Generate a plot for viewing different metrics dependent on the threshold applied to the top probes. """

    plot_data = relevant_results[(relevant_results['phase'] == phase) & (relevant_results['shuffle'] == shuffle)]

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.lineplot(x="threshold", y="top_score", data=plot_data, color="gray", ax=ax, label="peak")
    sns.lineplot(x="threshold", y="train_score", data=plot_data, color="green", ax=ax, label="train")
    sns.lineplot(x="threshold", y="test_score", data=plot_data, color="red", ax=ax, label="test")
    sns.lineplot(x="threshold", y="test_overlap", data=plot_data, color="orchid", ax=ax, label="overlap")

    rect = patches.Rectangle((158, -0.3), 5.0, 1.0, facecolor='gray', fill=True, alpha=0.25)
    ax.add_patch(rect)

    ax.legend(labels=['peak', 'train', 'test', 'overlap'])
    plt.suptitle("Scores by top probe threshold in '{}, {}-shuffled' data".format(phase, shuffle))
    ax.set_ylabel('Mantel correlation')
    return fig, ax


@shared_task(bind=True)
def assess_performance(self, plot_descriptor, data_root="/data"):
    """ Calculate metrics at different thresholds for relevant genes and plot them.

    :param self: Allows interacting with celery
    :param plot_descriptor: Abbreviated string describing plot's underlying data
    :param data_root: default /data, base path to all of the results
    """

    # It's tough to pass a list of thresholds via url, so it's hard-coded here.
    thresholds = [4, 8, 12, 16, 20, 24, 32, 64, 128, 157, 160, 256, 320, 512, 1024]
    progress_recorder = ProgressRecorder(self)

    comp, parby, splby, mask, algo, phase, opposite_phase, results = interpret_descriptor(plot_descriptor)

    n = len(results)
    total = n * len(thresholds) + 10
    progress_recorder.set_progress(0, total, "Finding results")
    print("Found {:,} results ({} {} {} {} {} {} {})".format(n, "glasser", "fornito", algo, comp, parby, splby, mask))

    if len(results) > 0:
        relevant_results = []

        """ Calculate (or load) stats for individual tsv files. """
        for i, path in enumerate(results.values('tsv_path')):
            if os.path.isfile(path['tsv_path']):
                for j, threshold in enumerate(thresholds):
                    relevant_results.append(
                        dict_from_result(path['tsv_path'], base_path=data_root, probe_sig_threshold=threshold)
                    )
                    progress_recorder.set_progress(
                        i * len(thresholds) + j, total, "Processing {:,}/{:,} results".format(i, n)
                    )
            else:
                print("ERR: DOES NOT EXIST: {}".format(path['tsv_path']))
        rdf = pd.DataFrame(relevant_results)

        # Just temporarily, in case debugging offline is necessary
        rdf.to_pickle(os.path.join(data_root, "plots", "rdf.df"))

        """ Calculate grouped stats, overlap between each tsv file and its shuffle-based 'peers'. """
        progress_recorder.set_progress(n * len(thresholds), total, "Generating overlap lists")
        shuffles = list(set(rdf['shuffle']))
        # for k, shuffle in enumerate(shuffles):
        #     shuffle_mask = rdf['shuffle'] == shuffle
        #     # We can only do this in training data, unless we want to double the workload above for test, too.
        #     # Generate a list of lists for each file's single 'train_overlap' cell in our dataframe.
        #     rdf.loc[shuffle_mask, 'train_overlap'] = [
        #         algorithms.pct_similarity_list(list(rdf.loc[shuffle_mask, 'path']), top=t) for t in thresholds
        #     ]
        #     progress_recorder.set_progress(n * len(thresholds) + (k * 2), total, "overlap list {}/{}".format(
        #         k, len(shuffles)
        #     ))

        progress_recorder.set_progress(n * len(thresholds) + (len(shuffles) * 2), total, "Generating plot")
        # Plot performance of thresholds on correlations.
        phase_mask = rdf['phase'] == phase
        f_perf, a_perf = plot_scores_over_thresholds(
            rdf[phase_mask & (rdf['shuffle'] == 'none')], phase, 'none'
        )
        f_perf.savefig(os.path.join(data_root, "plots", "performance_{}.png".format(plot_descriptor)))

        progress_recorder.set_progress(total, total, "Finished")

    return None
