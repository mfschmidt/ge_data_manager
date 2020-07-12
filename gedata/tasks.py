from __future__ import absolute_import, unicode_literals
from celery import shared_task
from celery_progress.backend import ProgressRecorder

from datetime import datetime
import pytz

from django.utils import timezone as django_timezone

import os
import glob
import re
import pickle
import numpy as np
import pandas as pd
import logging
import json
from scipy import stats

from pygest import algorithms
from pygest.convenience import bids_val
import pygest as ge

from .models import PushResult, ResultSummary
from .plots import plot_all_train_vs_test, plot_performance_over_thresholds, plot_overlap
from .plots import plot_fig_2, plot_fig_3, plot_fig_4
from .plots import describe_overlap, describe_mantel
from .genes import describe_genes, write_result_as_entrezid_ranking

from .decorators import print_duration


class NullHandler(logging.Handler):
    def emit(self, record):
        pass


null_handler = NullHandler()


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
    """ Convert a string from the json file, like "5 days, 2:45:32.987", into integer seconds.

    :param elapsed: string scraped from json file
    :return: seconds represented by the elapsed string, as an integer
    """

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
    """ Return New_York time, with timezone, from file's last modified timestamp. """
    return pytz.timezone("America/New_York").localize(
        datetime.fromtimestamp(os.path.getmtime(path))
    )


def gather_results(pattern=None, data_root="/data", pr=None):
    """ Quickly find available files, and queue any heavy processing elsewhere.

    :param pattern: glob pattern to restrict files searched for
    :param data_root: default /data, base path to all of the results
    :param pr: progress_recorder to report intermediate status.
    """

    if pr:
        pr.set_progress(0, 100, "Looking for files")

    # Get a list of files to parse.
    if pattern is None:
        glob_pattern = os.path.join(data_root, "derivatives", "*", "*", "*", "*.tsv")
    else:
        glob_pattern = os.path.join(data_root, pattern)
    print("Starting to glob files ('{}')".format(glob_pattern))
    sda = os.listdir(os.path.join(data_root, "derivatives"))[0]
    sdb = os.listdir(os.path.join(data_root, "derivatives", sda))[0]
    print(os.path.join(data_root, "derivatives", sda, sdb))
    print(os.listdir(os.path.join(data_root, "derivatives", sda, sdb)))
    files = glob.glob(glob_pattern)
    print("File glob complete; discovered {:,} results".format(len(files)))
    if pr:
        pr.set_progress(0, len(files), "Checking filesystem against database")

    dupe_count = 0
    new_count = 0
    # import random
    # for i, path in enumerate(random.sample(files, 500)):
    for i, path in enumerate(files):
        if PushResult.objects.filter(tsv_path=path).exists():
            dupe_count += 1
        else:
            new_count += 1
            f = path.split(os.sep)[-1]
            rel_path = path[(len(data_root) + 1):(len(path) - len(f) - 1)]
            result = {
                'data_dir': data_root,
                'rel_path': rel_path,
                'path': os.path.join(data_root, rel_path),
                'tsv_file': f,
                'json_file': f.replace(".tsv", ".json"),
                'log_file': f.replace(".tsv", ".log"),
            }

            # Gather information about each result.
            for bids_pair in re.findall(
                r"[^\W_]+-[^\W_]*", os.path.join(result.get('path', ''), result.get('tsv_file', ''))
            ):
                bids_key, bids_value = bids_pair.split("-")
                result[bids_key] = bids_value
            if "mask" in result and result.get("mask", "") == "none":
                result["mask"] = "00"

            # Second, parse the key-value pairs from the json file containing processing information.
            result.update(json_contents(os.path.join(result['path'], result['json_file'])))

            split_key = "batch-test" if "batch-test" in result['path'] else "batch-train"
            split = extract_seed(os.path.join(result['path'], result['tsv_file']), split_key)
            resample = "whole"
            if 200 <= split <= 299:
                resample = "split-half"
            elif 400 <= split <= 499:
                resample = "split-quarter"

            # Finally, put it all into a model for storage in the database.
            r = PushResult(
                sourcedata=build_descriptor(
                    result.get("comp", ""), result.get("splby", ""), result.get("mask", ""),
                    result.get("norm", ""), split, level="long",
                ),
                descriptor=build_descriptor(
                    result.get("comp", ""), result.get("splby", ""), result.get("mask", ""),
                    result.get("norm", ""), split, level="short",
                ),
                resample=resample,
                json_path=os.path.join(result['path'], result['json_file']),
                tsv_path=os.path.join(result['path'], result['tsv_file']),
                log_path=os.path.join(result['path'], result['log_file']),
                # For dates, we assume EDT (-0400) as most runs were in EDT. If EST, the hour makes no real difference.
                start_date=datetime.strptime(
                    result.get("began", "1900-01-01 00:00:00") + "-0400", "%Y-%m-%d %H:%M:%S%z"
                ),
                end_date=datetime.strptime(
                    result.get("completed", "1900-01-01 00:00:00") + "-0400", "%Y-%m-%d %H:%M:%S%z"
                ),
                host=result.get("host", ""),
                command=result.get("command", ""),
                version=result.get("pygest version", ""),
                duration=seconds_elapsed(result.get("elapsed", "")),
                sub=result.get("sub", ""),
                hem=result.get("hem", " ").upper()[0],
                samp=result.get("samp", ""),
                prob=result.get("prob", ""),
                parby=result.get("parby", ""),
                splby=result.get("splby", ""),
                batch=result.get("batch", ""),
                tgt=result.get("tgt", ""),
                algo=result.get("algo", ""),
                shuf=result.get("shuf", ""),
                norm=result.get("norm", ""),
                comp=result.get("comp", ""),
                mask=result.get("mask", ""),
                adj=result.get("adj", ""),
                seed=int(result.get("seed", 0)),
                split=split,
                columns=0,
                rows=0,
            )
            r.save()
        if pr:
            pr.set_progress(i, len(files), "Summarizing new files")

    print("Processed {:,} files; found {:,} new results and {:,} were already in the database (now {:,}).".format(
        len(files), new_count, dupe_count, PushResult.objects.count(),
    ))
    # print(PushResult.objects.values('descriptor').annotate(count=Count('descriptor')))

    if PushResult.objects.count() > 0:
        print("Saving a summary, dated {}.".format(str(django_timezone.now())))
        print("There are {} PushResults in the database.".format(PushResult.objects.count()))
        s = ResultSummary.current()
        print("Summary: ", s.to_json())
        s.save()
    else:
        print("No PushResults, not saving a summary.")


@shared_task(bind=True)
def gather_results_as_task(self, pattern=None, data_root="/data"):
    """ Celery wrapper for gather_results() """
    progress_recorder = ProgressRecorder(self)
    gather_results(pattern=pattern, data_root=data_root, pr=progress_recorder)


@shared_task(bind=True)
def clear_all_jobs(self):
    """ Delete all results from the database, not the filesystem, and create a new summary indicating no records.

    :param self: available through "bind=True", allows a reference to the celery task this function becomes a part of.
    """
    progress_recorder = ProgressRecorder(self)
    progress_recorder.set_progress(0, 100, "Deleting file records (not files)")
    PushResult.objects.all().delete()
    progress_recorder.set_progress(50, 100, "Updating summaries")
    s = ResultSummary.empty(timestamp=True)
    print("Summary: ", s.to_json())
    s.save()
    progress_recorder.set_progress(100, 100, "Cleared")


def clear_some_jobs(descriptor=None):
    """ Delete all results from the database, not the filesystem, and create a new summary indicating no records.

    :param descriptor: id string to specify which group's cache data will be cleared.
    """
    rdict = interpret_descriptor(descriptor)
    rdict['query_set'].delete()
    s = ResultSummary.current()
    print("Summary: ", s.to_json())
    s.save()


def comp_from_signature(signature, filename=False):
    """ Convert signature to comp string
    :param signature: The 7-character string indicating which group of results to use.
    :param filename: Set filename=True to get the comparator filename rather than its BIDS string representation.
    :return:
    """
    comp_map = {
        'hcpg': 'hcpniftismoothconnparbyglassersim',
        'hcpw': 'hcpniftismoothconnsim',
        'nkig': 'indiglasserconnsim',
        'nkiw': 'indiconnsim',
        'f__g': 'fearglassersim',
        'f__w': 'fearconnsim',
        'n__g': 'neutralglassersim',
        'n__w': 'neutralconnsim',
        'fn_g': 'fearneutralglassersim',
        'fn_w': 'fearneutralconnsim',
        'px_w': 'glasserwellidsproximity',
        'px_g': 'glasserparcelsproximity',
        'pxlw': 'glasserwellidslogproximity',
        'pxlg': 'glasserparcelslogproximity',
        'hcpniftismoothconnparbyglassersim': 'hcp_niftismooth_conn_parby-glasser_sim.df',
        'hcpniftismoothconnsim': 'hcp_niftismooth_conn_sim.df',
        'indiglasserconnsim': 'indi-glasser-conn_sim.df',
        'indiconnsim': 'indi-connectivity_sim.df',
        'fearglasserconnsim': 'fear_glasser_sim.df',
        'fearconnsim': 'fear_conn_sim.df',
        'neutralglassersim': 'neutral_glasser_sim.df',
        'neutralconnsim': 'neutral_conn_sim.df',
        'fearneutralglassersim': 'fear-neutral_glasser_sim.df',
        'fearneutralconnsim': 'fear-neutral_conn_sim.df',
        'glasserwellidsproximity': 'glasser-wellids-proximity',
        'glasserparcelsproximity': 'glasser-parcels-proximity',
        'glasserwellidslogproximity': 'glasser-wellids-log-proximity',
        'glasserparcelslogproximity': 'glasser-parcels-log-proximity',
    }

    if filename:
        return comp_map[comp_map[signature.lower()]]
    return comp_map[signature.lower()]


def test_score(tsv_file, base_path='/data', own_expr=False, mask='none', probe_significance_threshold=None):
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
    expr_file = os.path.join(
        base_path, "splits", "sub-all_hem-A_samp-glasser_prob-fornito", "batch-{}".format(batch),
        "parcelby-{}_splitby-{}.{}.df".format(
            bids_val('parby', tsv_file), bids_val('splby', tsv_file),
            "raw" if bids_val('norm', tsv_file) == "none" else bids_val('norm', tsv_file)
        )
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
    print("   tsv: {}".format(tsv_file))
    print("  expr: {}".format(expr_file))
    print("  comp: {}".format(comp_file))
    print("  test score from top {:,} probes, and {:,} samples".format(len(scoring_probes), len(overlapping_samples)))
    expr = expr.loc[scoring_probes, overlapping_samples]
    comp = comp.loc[overlapping_samples, overlapping_samples]
    expr_mat = np.corrcoef(expr, rowvar=False)
    comp_mat = comp.values
    expr_vec = expr_mat[np.tril_indices(n=expr_mat.shape[0], k=-1)]
    comp_vec = comp_mat[np.tril_indices(n=comp_mat.shape[0], k=-1)]

    if mask != "none":
        data = ge.Data(base_path, external_logger=null_handler)
        v_mask = algorithms.one_mask(expr, mask, bids_val('parby', tsv_file), data, null_handler)
        expr_vec = expr_vec[v_mask]
        comp_vec = comp_vec[v_mask]

    # return algorithms.correlate(expr, comp_mat)
    return np.corrcoef(expr_vec, comp_vec)[0, 1]


def train_vs_test_overlap(tsv_file, probe_significance_threshold=None):
    """ Determine the percentage of overlap between the provided tsv_file and
        its complementary train/test tsv_file. """

    if 'test' in tsv_file:
        comp_file = tsv_file.replace('test', 'train')
    elif 'train' in tsv_file:
        comp_file = tsv_file.replace('train', 'test')
    else:
        print("Non train/test file: '{}'".format(tsv_file))
        return 0.0

    # Sometimes, a comparision will be requested between a train file and its nonexistent test counterpart.
    # In that situation, it's best to just return 0.0 rather than add all the logic to determine test completion.
    if os.path.isfile(tsv_file) and os.path.isfile(comp_file):
        return algorithms.pct_similarity(
            [tsv_file, comp_file], map_probes_to_genes_first=False, top=probe_significance_threshold
        )
        # raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), comp_file)
    else:
        return 0.00


def extract_seed(path, key):
    """ Scrape the 5-character seed from the path and return it as an integer.

    :param path: path to the tsv file containing results
    :param key: substring preceding the seed, "batch-train" for splits, seed-" for shuffles
    """
    try:
        i = path.find(key) + len(key)
        return int(path[i:i + 5])
    except ValueError:
        return 0


def results_as_dict(tsv_file, base_path, probe_sig_threshold=None, use_cache=True):
    """ Return a key-value description of a single result.
        This is the workhorse of all of these plots. All calculations from these runs come from this function.
    """

    # These calculations can be expensive when run a lot. Load a cached copy if possible.
    analyzed_path = tsv_file.replace(".tsv", ".top-{}.v4.json".format(
        "peak" if probe_sig_threshold is None else "{:04}".format(probe_sig_threshold)
    ))
    if use_cache and os.path.isfile(analyzed_path):
        saved_dict = json.load(open(analyzed_path, 'r'))
        return saved_dict

    mask = bids_val('mask', tsv_file)
    norm = bids_val('norm', tsv_file)
    shuf = bids_val('shuf', tsv_file)
    seed = extract_seed(tsv_file, "_seed-")

    result_dict = algorithms.run_results(tsv_file, probe_sig_threshold)
    result_dict.update({
        # Characteristics available in the path, if necessary
        'path': tsv_file,
        'phase': 'train' if 'batch-train' in tsv_file else 'test',
        'algo': bids_val('algo', tsv_file),
        'splby': 'glasser' if 'splby-glasser' in tsv_file else 'wellid',
        'parby': 'glasser' if 'parby-glasser' in tsv_file else 'wellid',
        'mask': mask,
        'norm': norm,
        'shuf': shuf,
        # Neither path nor calculation, the threshold we use for calculations
        'threshold': "peak" if probe_sig_threshold is None else "{:04}".format(probe_sig_threshold),
        # Calculations on the data within the tsv file - think about what we want to maximize, beyond Mantel.
        # These use data from this file, its original train and test expression data, and conn-sim data to calculate
        # the following.
        'train_score': test_score(
            tsv_file, base_path, own_expr=True, mask='none', probe_significance_threshold=probe_sig_threshold),
        'test_score': test_score(
            tsv_file, base_path, own_expr=False, mask='none', probe_significance_threshold=probe_sig_threshold),
        'masked_train_score': test_score(
            tsv_file, base_path, own_expr=True, mask=mask, probe_significance_threshold=probe_sig_threshold),
        'masked_test_score': test_score(
            tsv_file, base_path, own_expr=False, mask=mask, probe_significance_threshold=probe_sig_threshold),
        'train_vs_test_overlap': train_vs_test_overlap(tsv_file, probe_significance_threshold=probe_sig_threshold),
        'split': extract_seed(tsv_file, "batch-train"),
        'seed': seed,
        'real_tsv_from_shuffle': tsv_file.replace("shuf-" + shuf, "shuf-none").replace("_seed-{:05d}".format(seed), ""),
    })

    # Cache results to prevent repeated recalculation of the same thing.
    json.dump(result_dict, open(analyzed_path, 'w'))

    return result_dict


def ready_erminej(base_path):
    """ If ermineJ is not available, download and prepare it. """

    import urllib.request as request
    import gzip
    import subprocess

    # Ensure pre-requisites exist and are accessible
    ej_path = os.path.join(base_path, "genome", "erminej")
    os.makedirs(ej_path, exist_ok=True)
    os.makedirs(os.path.join(ej_path, "data"), exist_ok=True)
    ontology = {
        'url': 'http://archive.geneontology.org/latest-termdb/go_daily-termdb.rdf-xml.gz',
        'file': '2019-07-09-erminej_go.rdf-xml',
    }
    annotation = {
        'url': 'https://gemma.msl.ubc.ca/annots/Generic_human_ncbiIds_noParents.an.txt.gz',
        'file': '2020-04-22-erminej_human_annotation_entrezid.txt',
    }
    software = {
        'url': 'http://home.pavlab.msl.ubc.ca/ermineJ/distributions/ermineJ-3.1.2-generic-bundle.zip',
        'file': 'ermineJ-3.1.2-generic-bundle.zip',
    }
    for prereq in [ontology, annotation, ]:
        if not os.path.exists(os.path.join(ej_path, prereq['file'])):
            print("Downloading fresh {}, it didn't exist.".format(prereq['file']))
            response = request.urlopen(prereq['url'])
            with open(os.path.join(ej_path, prereq['file']), "wb") as f:
                f.write(gzip.decompress(response.read()))
    if not os.path.exists(os.path.join(ej_path, software['file'])):
        response = request.urlopen(software['url'])
        with open(os.path.join(ej_path, software['file']), "wb") as f:
            data = response.read()
            f.write(data)

    # Unzip ermineJ software, if necessary
    if not os.path.exists(os.path.join(ej_path, "ermineJ-3.1.2", "bin", "ermineJ.sh")):
        print("Unzipping ermineJ software.")
        p = subprocess.run(
            ['unzip', '-u', os.path.join(ej_path, software['file']), ],
            cwd=ej_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if p.returncode != 0:
            print(p.stdout.decode())
            print(p.stderr.decode())

        p = subprocess.run(
            ['chmod', "a+x", os.path.join(ej_path, "ermineJ-3.1.2", "bin", "ermineJ.sh"), ],
            cwd=ej_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if p.returncode != 0:
            print(p.stdout.decode())
            print(p.stderr.decode())

    return_dict = {
        "executable": os.path.join(ej_path, "ermineJ-3.1.2", "bin", "ermineJ.sh"),
        "ontology": os.path.join(ej_path, ontology['file']),
        "annotation": os.path.join(ej_path, annotation['file']),
        "data": os.path.join(ej_path, "data"),
        "ready": True,
    }
    for req_path in ["executable", "ontology", "annotation", "data", ]:
        if not os.path.exists(return_dict[req_path]):
            return_dict["ready"] = False
    return return_dict


def run_gene_ontology(tsv_file, base_path="/data"):
    """ Run ermineJ gene ontology on one gene ordering.

    :param tsv_file: full path to a PyGEST result
    :param base_path: PyGEST data root
    """

    import subprocess

    ej = ready_erminej(base_path)

    go_path = tsv_file.replace(".tsv", ".ejgo_roc")

    rank_file = write_result_as_entrezid_ranking(tsv_file)

    # Only run gene ontology if it does not yet exist.
    print("initiating ermineJ run on '{}...{}'".format(tsv_file[:30], tsv_file[-30:]))
    print("  'ready': {}, 'go_path': ...{}".format(ej["ready"], go_path[-36:]))
    if ej["ready"] and not os.path.isfile(go_path):
        p = subprocess.run(
            [
                ej['executable'],
                '-d', ej["data"],
                '--annots', ej['annotation'],
                '--classFile', ej['ontology'],
                '--scoreFile', rank_file,
                '--test', 'ROC',  # Method for computing significance. ROC best for ordinal rankings
                '--mtc', 'FDR',  # FDR indicates Benjamini-Hochberg corrections for false discovery rate
                '--reps', 'BEST',  # If a gene has multiple probes/ids in input, use BEST
                '--genesOut',  # Include gene symbols in output
                '--minClassSize', '5',  # smallest gene set size to be considered
                '--maxClassSize', '128',  # largest gene set size to be considered
                '-aspects', 'BCM',  # Test against all three GO components
                '-b', 'false',  # Big is not better, rankings are low==good
                '--logTrans', 'false',  # If we fed p-values, we would set this to true
                '--output', go_path,
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Write the log file
        with open(tsv_file.replace(".tsv", ".ejlog"), "w") as f:
            f.write("STDOUT:\n")
            f.write(p.stdout.decode())
            f.write("STDERR:\n")
            f.write(p.stderr.decode())


def build_descriptor(comp, splitby, mask, normalization, split, level="short"):
    """ Generate a shorthand descriptor for the group a result belongs to. """

    # From actual file, or from path to result, boil down comparator to its abbreviation
    comp_map = {
        'hcp_niftismooth_conn_parby-glasser_sim.df': ('hcpg', "HCP [rest] (glasser parcels)"),
        'hcpniftismoothconnparbyglassersim': ('hcpg', "HCP [rest] (glasser parcels)"),
        'hcp_niftismooth_conn_sim.df': ('hcpw', "HCP [rest] (wellids)"),
        'hcpniftismoothconnsim': ('hcpw', "HCP [rest] (wellids)"),
        'indi-glasser-conn_sim.df': ('nkig', "NKI [rest] (glasser parcels)"),
        'indiglasserconnsim': ('nkig', "NKI [rest] (glasser parcels)"),
        'indi-connectivity_sim.df': ('nkiw', "NKI [rest] (wellids)"),
        'indiconnsim': ('nkiw', "NKI [rest] (wellids)"),
        'fear_glasser_sim.df': ('f__g', "HCP [task: fear] (glasser parcels)"),
        'fearglassersim': ('f__g', "HCP [task: fear] (glasser parcels)"),
        'fear_conn_sim.df': ('f__w', "HCP [task: fear] (wellids)"),
        'fearconnsim': ('f__w', "HCP [task: fear] (wellids)"),
        'neutral_glasser_sim.df': ('n__g', "HCP [task: neutral] (glasser parcels)"),
        'neutralglassersim': ('n__g', "HCP [task: neutral] (glasser parcels)"),
        'neutral_conn_sim.df': ('n__w', "HCP [task: neutral] (wellids)"),
        'neutralconnsim': ('n__w', "HCP [task: neutral] (wellids)"),
        'fear-neutral_glasser_sim.df': ('fn_g', "HCP [task: fear-neutral] (glasser parcels)"),
        'fearneutralglassersim': ('fn_g', "HCP [task: fear-neutral] (glasser parcels)"),
        'fear-neutral_conn_sim.df': ('fn_w', "HCP [task: fear-neutral] (wellids)"),
        'fearneutralconnsim': ('fn_w', "HCP [task: fear-neutral] (wellids)"),
        'glasserwellidsproximity': ('px_w', "Proximity (wellids)"),
        'glasserparcelsproximity': ('px_g', "Proximity (glasser parcels)"),
        'glasserwellidslogproximity': ('pxlw', "log Proximity (wellids)"),
        'glasserparcelslogproximity': ('pxlg', "log Proximity (glasser parcels)"),
    }

    # Make short string for split seed and normalization
    split = int(split)
    if 200 <= split < 300:
        xv = "2"
        xvlong = "halves"
    elif 400 <= split < 500:
        xv = "4"
        xvlong = "quarters"
    else:
        xv = "_"
        xvlong = "undefined"
    norm = "s" if normalization == "srs" else "_"

    # Build and return the descriptor
    if level == "short":
        return "{}{}{:0>2}{}{}{}".format(
            comp_map[comp][0],
            splitby[0],
            0 if mask == "none" else int(mask),
            's',
            norm,
            xv,
        )
    elif level == "long":
        return "{}, {}-normalized, {}".format(
            comp_map[comp][1], normalization, xvlong,
        )
    else:
        return "Undefined"


def interpret_descriptor(descriptor):
    """ Parse the plot descriptor into parts """
    rdict = {
        'descriptor': descriptor,
        'comp': comp_from_signature(descriptor[:4]),
        'parby': "glasser" if descriptor[3].lower() == "g" else "wellid",
        'splby': "glasser" if descriptor[4].lower() == "g" else "wellid",
        'mask': descriptor[5:7],
        'algo': 'once' if descriptor[7] == "o" else "smrt",
        'norm': 'srs' if ((len(descriptor) > 8) and (descriptor[8] == "s")) else "none",
        'xval': descriptor[9] if len(descriptor) > 9 else '0',
        'phase': 'train',
        'opposite_phase': 'test',
    }

    try:
        rdict['threshold'] = int(descriptor[8:])
    except ValueError:
        # This catches "peak" or "", both should be fine as None
        rdict['threshold'] = None

    # The xval, or cross-validation sample split range indicator,
    # is '2' for splits in the 200's and '4' for splits in the 400s.
    # Splits in the 200's are split-halves; 400's are split-quarters.
    split_min = 0
    split_max = 999
    if rdict['xval'] == '2':
        split_min = 200
        split_max = 299
    if rdict['xval'] == '4':
        split_min = 400
        split_max = 499
    rdict['query_set'] = PushResult.objects.filter(
        samp="glasser", prob="fornito", split__gte=split_min, split__lte=split_max,
        algo=rdict['algo'], comp=rdict['comp'], parby=rdict['parby'], splby=rdict['splby'],
        mask=rdict['mask'], norm=rdict['norm'],
        batch__startswith=rdict['phase']
    )

    rdict['n'] = rdict['query_set'].count()
    # Files are named with mask-none rather than mask-00, so tweak that key
    rdict['glob_mask'] = "none" if rdict['mask'] == "00" else rdict['mask']
    rdict['glob_pattern'] = os.path.join(
        "derivatives", "sub-all_hem-A_samp-glasser_prob-fornito",
        "parby-{parby}_splby-{splby}_batch-{phase}00{xval}*",
        "tgt-max_algo-{algo}_shuf-*",
        "sub-all_comp-{comp}_mask-{glob_mask}_norm-{norm}_adj-none*.tsv"
    ).format(**rdict)

    return rdict


def calc_ttests(row, df):
    """ for a given dataframe row, if it's a real result, return its t-test t-value vs all shuffles in df. """
    if row.shuf == 'none':
        return stats.ttest_1samp(
            df[(df['threshold'] == row.threshold)]['train_score'],
            row.train_score,
        )[0]
    else:
        return 0.0


def calc_real_v_shuffle_overlaps(row, df):
    """ for a given dataframe row, if it's a real result, return its overlap pctage vs all shuffles in df. """
    if row.shuf == 'none':
        overlaps = []
        for shuffled_tsv in df[(df['threshold'] == row.threshold)]['path']:
            top_threshold = row.threshold
            if top_threshold == "peak" or top_threshold == 0:
                top_threshold = None
            overlaps.append(algorithms.pct_similarity([row.path, shuffled_tsv], top=top_threshold))
        return np.mean(overlaps)
    else:
        return 0.0


def calc_total_overlap(row, df):
    """ This doesn't really work, yet, as experimentation is being done for what it should mean. """
    if row.shu == 'none':
        overlaps = []
        for shuffled_tsv in df[(df['threshold'] == row.threshold)]['path']:
            overlaps.append(algorithms.pct_similarity([row.path, shuffled_tsv], top=row.threshold))
        return np.mean(overlaps)
    else:
        return 0.0


@print_duration
def update_is_necessary(descriptor, data_root="/data"):
    """ Figure out if there are new files to include, necessitating recalculation and cache updates, or not.

    :param descriptor: An abbreviation indicating which items we are calculating
    :param str data_root: The PYGEST_DATA base path where all PyGEST data files can be found
    """

    # Find all results in our database that match our descriptor
    rdict = interpret_descriptor(descriptor)

    # Find all files in the filesystem that match our descriptor
    glob_pattern = os.path.join(data_root, rdict['glob_pattern'])
    print("Starting to glob files ('{}')".format(glob_pattern))
    files = glob.glob(glob_pattern)

    print("For {}, found {} database entries and {} result files downstream of {}.".format(
        descriptor, len(rdict['query_set']), len(files), data_root,
    ))

    print("For descriptor {}, {:,} records in database -vs- {:,} files".format(
        descriptor, len(rdict['query_set']), len(files)
    ))
    return len(rdict['query_set']) < len(files), rdict


@print_duration
def calculate_individual_stats(
        descriptor, progress_recorder, progress_from=0, progress_to=99, data_root="/data", use_cache=True
):
    """ Determine the list of results necessary, and build or load a dataframe around them. """

    do_update, rdict = update_is_necessary(descriptor)
    print("Update necessary? - {}".format(do_update))
    if do_update:
        # Delete relevant database records, then rebuild them with newly discovered files.
        print("   clearing jobs (in calculate_individual_stats, because db out of date)")
        clear_some_jobs(descriptor)
        print("   clearing macros (in calculate_individual_stats, because db out of date)")
        clear_macro_cache(descriptor, data_root=data_root)
        print("   gathering results (in calculate_individual_stats, because db out of date)")
        gather_results(pattern=rdict['glob_pattern'], data_root=data_root)

    rdict = interpret_descriptor(descriptor)
    progress_recorder.set_progress(progress_from, progress_to, "Step 1/3<br />Finding results")
    print("Found {:,} results in database ({} {} {} {} {} {} {} {}) @{}".format(
        rdict['n'], "glasser", "fornito",
        rdict['algo'], rdict['comp'], rdict['parby'], rdict['splby'], rdict['mask'], rdict['norm'],
        'peak' if rdict['threshold'] is None else rdict['threshold'],
    ))

    cache_file = os.path.join(data_root, "plots", "cache", "{}_summary_individual.df".format(rdict['descriptor']))
    if use_cache and os.path.isfile(cache_file):
        """ Load results from a cached file, if possible"""
        print("Loading individual data from cache, {}".format(cache_file))
        df = pickle.load(open(cache_file, "rb"))
    else:
        """ Calculate results for individual tsv files. """
        relevant_results = []
        for i, path in enumerate(rdict['query_set'].values('tsv_path')):
            print("Calculating stats for #{}. {}".format(i, path['tsv_path']))
            if os.path.isfile(path['tsv_path']):
                relevant_results.append(
                    results_as_dict(path['tsv_path'], base_path=data_root, probe_sig_threshold=rdict['threshold'])
                )
                run_gene_ontology(path['tsv_path'], base_path=data_root)
            else:
                print("ERR: DOES NOT EXIST: {}".format(path['tsv_path']))

            complete_portion = ((i + 1) / rdict['n']) * (progress_to - progress_from)
            progress_recorder.set_progress(
                progress_from + complete_portion, 100,
                "Step 1/3<br />Processing {:,}/{:,} results".format(i, rdict['n'])
            )

        df = pd.DataFrame(relevant_results)

        os.makedirs(os.path.join(data_root, "plots", "cache"), exist_ok=True)
        df.to_pickle(cache_file, protocol=4)

    progress_recorder.set_progress(
        progress_to - 1, 100,
        "Step 1/3<br />Processed {:,} results".format(rdict['n'])
    )

    return rdict, df


@print_duration
def calculate_group_stats(
        rdict, rdf, progress_recorder, progress_from=0, progress_to=99, data_root="/data", use_cache=True
):
    """ Using meta-data from each result, calculate statistics between results and about the entire group. """

    t_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    rdf.to_pickle("/data/plots/cache/group_{}_in_rdf.df".format(t_stamp), protocol=4)

    progress_recorder.set_progress(progress_from, 100, "Step 2/3<br />1. Within-shuffle overlap")
    """ Calculate similarity within split-halves and within shuffle-seeds. """
    n = len(set(rdf['shuf']))
    local_dfs = []
    for i, shuffle in enumerate(list(set(rdf['shuf']))):
        """ This explores the idea of viewing similarity between same-split-seed runs and same-shuffle-seed runs
            vs all shuffled runs of the type. All three of these are "internal" or "intra-list" overlap.
        """

        # If these data were cached, load them rather than re-calculate.
        post_file = os.path.join(data_root, "plots", "cache", "{}_summary_group_{}.df".format(
            rdict['descriptor'], shuffle
        ))
        if use_cache and os.path.isfile(post_file):
            """ Load results from a cached file, if possible"""
            print("Loading individual data from cache, {}".format(post_file))
            local_df = pickle.load(open(post_file, 'rb'))
        else:
            shuffle_mask = rdf['shuf'] == shuffle
            local_df = rdf.loc[shuffle_mask, :]
            local_n = len(local_df)
            print("Calculating percent overlaps and Kendall taus for {} {}-shuffles".format(local_n, shuffle))
            progress_recorder.set_progress(
                (100.0 * i / n), 100, "Step 2/3<br />2. {}-shuffle similarity".format(shuffle)
            )
            progress_delta = float(progress_to - progress_from) * 0.25 / n

            """ Calculate overlaps and such, only if there are results available to calculate. """
            if local_n > 0:
                # Overlap and Kendall tau intra- all members of this shuffle_mask (all 32 'none' shuffles perhaps)
                before_overlap = datetime.now()
                local_df['train_overlap'] = algorithms.pct_similarity_list(
                    list(local_df['path']), top=rdict['threshold']
                )
                before_ktau = datetime.now()
                local_df['train_ktau'] = algorithms.kendall_tau_list(
                    list(local_df['path']),
                )
                after_ktau = datetime.now()
                print("Overlaps for {} took {}".format(shuffle, str(before_ktau - before_overlap)))
                print("Kendall taus for {} took {}".format(shuffle, str(after_ktau - before_ktau)))

                progress_recorder.set_progress(
                    (100.0 * i / n) + (progress_delta * 1), 100, "Step 2/3<br />3. {}-shuffle seeds".format(shuffle)
                )
                # For each shuffled result, compare it against same-shuffled results from the same split
                for split in list(set(local_df['split'])):
                    # These values will all be 'nan' for unshuffled results. There's only one per split.
                    split_mask = local_df['split'] == split
                    if np.sum(split_mask) > 0:
                        # For a given split (within-split, across-seeds):
                        print("    overlap and ktau of {} {}-split {}-shuffles".format(
                            np.sum(shuffle_mask & split_mask), split, shuffle
                        ))
                        # The following similarities are masked to include only one shuffle type, and one split-half
                        local_df.loc[split_mask, 'overlap_by_split'] = algorithms.pct_similarity_list(
                            list(local_df.loc[split_mask, 'path']), top=rdict['threshold']
                        )
                        local_df.loc[split_mask, 'ktau_by_split'] = algorithms.kendall_tau_list(
                            list(local_df.loc[split_mask, 'path']),
                        )

                progress_recorder.set_progress(
                    (100.0 * i / n) + (progress_delta * 2), 100, "Step 2/3<br />4. {}-shuffle seeds".format(shuffle)
                )
                # For each shuffled result, compare it against same-shuffled results from the same shuffle seed
                for seed in list(set(local_df['seed'])):
                    seed_mask = local_df['seed'] == seed
                    if np.sum(seed_mask) > 0:
                        # For a given seed (within seed, across splits):
                        print("    overlap and ktau of {} {}-seed {}-shuffles".format(
                            np.sum(shuffle_mask & seed_mask), seed, shuffle
                        ))
                        local_df.loc[seed_mask, 'overlap_by_seed'] = algorithms.pct_similarity_list(
                            list(local_df.loc[seed_mask, 'path']), top=rdict['threshold']
                        )
                        local_df.loc[seed_mask, 'ktau_by_seed'] = algorithms.kendall_tau_list(
                            list(local_df.loc[seed_mask, 'path']),
                        )

                progress_recorder.set_progress(
                    (100.0 * i / n) + (progress_delta * 3), 100, "Step 2/3<br />5. {}-shuffle seeds".format(shuffle)
                )
                """ For each result in actual split-half train data, compare it to its shuffles. """
                # Each unshuffled run will match itself only for these two, resulting in a 1.0 perfect comparison.
                # So we overwrite them with a better comparison
                local_df['real_v_shuffle_overlap'] = local_df.apply(
                    lambda x: algorithms.pct_similarity(
                        [x.path, x.real_tsv_from_shuffle], top=rdict['threshold']
                    ), axis=1
                )
                local_df['real_v_shuffle_ktau'] = local_df.apply(
                    lambda x: algorithms.kendall_tau(
                        [x.path, x.real_tsv_from_shuffle]
                    ), axis=1
                )

            print("Pickling {}".format(post_file))
            local_df.to_pickle(post_file, protocol=4)

        # Whether from cache or calculation, keep a list of each separate shuffle dataframe for later concatenation.
        local_dfs.append(local_df)

    new_group_rdf = pd.concat(local_dfs, axis='index')
    new_group_rdf.to_pickle("/data/plots/cache/group_{}_out_rdf.df".format(t_stamp), protocol=4)

    progress_recorder.set_progress(progress_to, 100, "Step 2/3<br />Similarity calculated")

    return pd.concat(local_dfs, axis='index')


@print_duration
def write_gene_lists(rdict, rdf, progress_recorder, data_root="/data"):
    """ Rank genes and write them out as two csvs. """
    df_ranked_full, gene_description = describe_genes(rdf, rdict, progress_recorder)
    df_ranked_full.to_csv(os.path.join(data_root, "plots", "{}_ranked_full.csv".format(rdict['descriptor'])))
    df_ranked_full[['entrez_id', ]].to_csv(
        os.path.join(data_root, "plots", "{}_raw_ranked.csv".format(rdict['descriptor']))
    )

    return gene_description


@shared_task(bind=True)
def clear_macro_caches(self, descriptor, progress_from=0, progress_to=100, data_root="/data"):
    """ Remove the summary caches in /data/plots/cache/*.df """

    p_delta = progress_to - progress_from
    progress_recorder = ProgressRecorder(self)
    progress_recorder.set_progress(progress_from, progress_to, "Clearing existing plots")
    files_removed = 0

    print("Removing plots from {} and cache files from {}.".format(
        os.path.join(data_root, "plots"), os.path.join(data_root, "plots", "cache"),
    ))

    for f in os.listdir(os.path.join(data_root, "plots")):
        path = os.path.join(data_root, "plots", f)
        if os.path.isfile(path) and descriptor in f:
            print("deleting {}".format(f))
            os.remove(path)
            files_removed += 1
        # else:
        #     print("skipping {} ({})".format(f, descriptor))

    progress_recorder.set_progress(progress_from + (p_delta / 2), progress_to, "Clearing summary caches")

    for f in os.listdir(os.path.join(data_root, "plots", "cache")):
        path = os.path.join(data_root, "plots", "cache", f)
        if os.path.isfile(path) and descriptor in f:
            print("deleting {}".format(f))
            os.remove(path)
            files_removed += 1
        # else:
        #     print("skipping {} ({})".format(f, descriptor))

    progress_recorder.set_progress(progress_to, progress_to, "Cleared {} files".format(files_removed))


def clear_macro_cache(descriptor, data_root="/data"):
    """ Remove the summary caches in /data/plots/cache/*.df """

    files_removed = 0
    print("Removing plots from {} and cache files from {}.".format(
        os.path.join(data_root, "plots"), os.path.join(data_root, "plots", "cache"),
    ))

    def print_and_remove(path, file):
        if os.path.isfile(path) and descriptor in file:
            print("deleting {}".format(file))
            os.remove(path)
            return 1
        else:
            return 0

    for f in os.listdir(os.path.join(data_root, "plots")):
        files_removed += print_and_remove(os.path.join(data_root, "plots", f), f)

    for f in os.listdir(os.path.join(data_root, "plots", "cache")):
        files_removed += print_and_remove(os.path.join(data_root, "plots", "cache", f), f)


@shared_task(bind=True)
def clear_micro_caches(self, descriptor, progress_from=0, progress_to=100):
    """ Determine the list of results necessary, and build or load a dataframe around them. """

    progress_recorder = ProgressRecorder(self)
    progress_recorder.set_progress(progress_from, progress_to, "Clearing individual result caches")
    rdict = interpret_descriptor(descriptor)
    print("Found {:,} results ({} {} {} {} {} {} {} {}) @{}".format(
        rdict['n'], "glasser", "fornito",
        rdict['algo'], rdict['comp'], rdict['parby'], rdict['splby'], rdict['mask'], rdict['norm'],
        'peak' if rdict['threshold'] is None else rdict['threshold'],
    ))

    """ Delete calculated results for individual tsv files. """
    n_removed = 0
    for i, path in enumerate(rdict['query_set'].values('tsv_path')):
        analyzed_path = path['tsv_path'].replace(".tsv", ".top-{}.v3.json".format(
            "peak" if rdict['threshold'] is None else "{:04}".format(rdict['threshold'])
        ))
        if os.path.isfile(analyzed_path):
            os.remove(analyzed_path)
            n_removed += 1
        complete_portion = ((i + 1) / rdict['n']) * (progress_to - progress_from)
        progress_recorder.set_progress(
            progress_from + complete_portion, 100, "Deleting {:,}/{:,} results, removed {:,} caches.".format(
                i, rdict['n'], n_removed
            )
        )

    progress_recorder.set_progress(progress_to, 100, "Removed {:,} cached meta-results".format(n_removed))
    return rdict


@shared_task(bind=True)
def assess_everything(self, plot_descriptor, data_root="/data"):
    """ 1. Collect all results available (from database, not filesystem), and their individual stats.
        2. Read or calculate group statistics on the set of results.
        3. Build figures.
        4. Generate text reports, including figures.

    :param self: interact with celery via decorator
    :param plot_descriptor: Abbreviated string, like 'hcpww16ss' describing underlying data
    :param data_root: default root path to find all results
    """

    progress_recorder = ProgressRecorder(self)

    """ 1. Find all result files.
           Caches a json file for each result, and a summary dataframe, greatly speeding the process on subsequent runs.
    """
    dt_begin = datetime.now()
    rdict, rdf = calculate_individual_stats(
        plot_descriptor, progress_recorder, progress_from=0, progress_to=99, data_root=data_root
    )
    dt_down01 = datetime.now()
    print("{:,} records for full analysis. [{}]".format(len(rdf), str(dt_down01 - dt_begin)))

    """ 2. Calculate grouped stats, overlap within each group of tsv files.
           This, too, caches the resultant calculations.
    """
    rdf = calculate_group_stats(
        rdict, rdf, progress_recorder, progress_from=0, progress_to=99, data_root=data_root
    )
    dt_down02 = datetime.now()
    print("{:,} group stats. [{}]".format(len(rdf), str(dt_down02 - dt_down01)))

    i = 0  # i is the index into how many plots and reports to build, for reporting progress
    n = 6
    plotted_shuffles = ["none", "be04", "agno", ]

    """ 3. Plot aggregated data. """
    plot_title = "Mantels: {}s, split by {}, {}-masked, {}-ranked, {}-normed, by {}, top-{}".format(
        rdict['parby'], rdict['splby'], rdict['mask'], rdict['algo'], rdict['norm'], plot_descriptor[:3].upper(),
        'peak' if rdict['threshold'] is None else rdict['threshold']
    )
    progress_recorder.set_progress(100.0 * i / n, 100, "Step 3/3<br />Over-Plotting fig 2")
    f_over, axes = plot_all_train_vs_test(rdf, title=plot_title, fig_size=(14, 14), y_min=-0.15, y_max=0.70)
    f_over.savefig(os.path.join(data_root, "plots", "{}_f_over-plot-fig-2.png".format(plot_descriptor.lower())))
    i += 1
    dt_down03 = datetime.now()
    print("Figure 2 (overplot) generated in [{}]".format(len(rdf), str(dt_down03 - dt_down02)))

    plot_title = "Mantels: {}s, split by {}, {}-masked, {}-ranked, {}-normed, by {}, top-{}".format(
        rdict['parby'], rdict['splby'], rdict['mask'], rdict['algo'], rdict['norm'], plot_descriptor[:3].upper(),
        'peak' if rdict['threshold'] is None else rdict['threshold']
    )
    progress_recorder.set_progress(100.0 * i / n, 100, "Step 3/3<br />Plotting figure 2")
    f_2, axes = plot_fig_2(rdf, shuffles=plotted_shuffles, title=plot_title, fig_size=(12, 6), y_min=-0.15, y_max=0.50)
    f_2.savefig(os.path.join(data_root, "plots", "{}_fig_2.png".format(plot_descriptor.lower())))
    i += 1
    dt_down03 = datetime.now()
    print("Figure 2 generated in [{}]".format(len(rdf), str(dt_down03 - dt_down02)))

    progress_recorder.set_progress(100.0 * i / n, 100, "Step 3/3<br />Plotting figure 3")
    f_3, axes = plot_fig_3(rdf, shuffles=plotted_shuffles, title=plot_title, fig_size=(6, 5), y_min=-0.1, y_max=0.50)
    f_3.savefig(os.path.join(data_root, "plots", "{}_fig_3.png".format(plot_descriptor.lower())))
    i += 1
    dt_down04 = datetime.now()
    print("Figure 3 generated in [{}]".format(len(rdf), str(dt_down04 - dt_down03)))

    progress_recorder.set_progress(100.0 * i / n, 100, "Step 3/3<br />Plotting figure 4")
    f_4, axes = plot_fig_4(rdf, shuffles=plotted_shuffles, title=plot_title, y_min=0.0, y_max=0.80, fig_size=(13, 5),)
    f_4.savefig(os.path.join(data_root, "plots", "{}_fig_4.png".format(plot_descriptor.lower())))
    i += 1
    dt_down05 = datetime.now()
    print("Figure 4 generated in [{}]".format(len(rdf), str(dt_down05 - dt_down04)))

    # TODO: Add plot for some result over all four distance masks.
    dt_down06 = datetime.now()
    print("Nothing generated in [{}]".format(len(rdf), str(dt_down06 - dt_down05)))

    """ 4. Describe results in text form. """
    progress_recorder.set_progress(100.0 * i / n, 100, "Step 3/3<br />Writing result text")
    mantel_description = describe_mantel(rdf, descriptor=plot_descriptor.lower(), title=plot_title, )
    i += 1
    dt_down07 = datetime.now()
    print("Mantel result text generated in [{}]".format(len(rdf), str(dt_down07 - dt_down06)))

    progress_recorder.set_progress(100.0 * i / n, 100, "Step 3/3<br />Writing gene lists")
    gene_description = write_gene_lists(rdict, rdf, progress_recorder, data_root="/data")
    i += 1
    dt_down08 = datetime.now()
    print("Gene lists generated in [{}]".format(len(rdf), str(dt_down08 - dt_down07)))

    progress_recorder.set_progress(100.0 * i / n, 100, "Step 3/3<br />Writing overlap description")
    overlap_description = describe_overlap(rdf, descriptor=plot_descriptor.lower(), title=plot_title, )
    dt_down09 = datetime.now()
    print("Overlap descriptions generated in [{}]".format(len(rdf), str(dt_down09 - dt_down08)))

    with open(os.path.join(data_root, "plots", "{}_report.html".format(plot_descriptor.lower())), 'w') as f:
        f.write("<h1>Mantel Correlations</h1>\n")
        f.write(mantel_description)
        f.write("<h1>Overlap Descriptions</h1>\n")
        f.write(overlap_description)
        f.write("<h1>Probe/Gene Descriptions</h1>\n")
        f.write(gene_description)
    i += 1

    progress_recorder.set_progress(99, 100, "Step 3/3. Plotting finished")
    print("Completely finished in [{}] ({} to {})".format(
        str(datetime.now() - dt_begin), str(dt_begin), str(datetime.now())
    ))


@shared_task(bind=True)
def just_genes(self, plot_descriptor, data_root="/data"):
    """ 1. Collect all results available (from database, not filesystem), and their individual stats.
        2. Read or calculate group statistics on the set of results.
        3. Build figures.
        4. Generate text reports, including figures.

    :param self: interact with celery via decorator
    :param plot_descriptor: Abbreviated string, like 'hcpww16ss' describing underlying data
    :param data_root: default root path to find all results
    """

    progress_recorder = ProgressRecorder(self)
    print("In just_genes, about to start.")
    print("In just_genes, calculating individual stats.")
    rdict, rdf = calculate_individual_stats(
        plot_descriptor, progress_recorder, progress_from=0, progress_to=5, data_root=data_root
    )
    print("In just_genes, calculating group stats.")
    rdf = calculate_group_stats(
        rdict, rdf, progress_recorder, progress_from=5, progress_to=10, data_root=data_root
    )
    progress_recorder.set_progress(10, 100, "Just writing gene lists")
    print("In just_genes, working on gene descriptions.")
    gene_description = write_gene_lists(rdict, rdf, progress_recorder, data_root="/data")
    with open(os.path.join(data_root, "plots", "{}_genes.html".format(plot_descriptor.lower())), 'w') as f:
        f.write("<h1>Probe/Gene Descriptions</h1>\n")
        f.write(gene_description)


@shared_task(bind=True)
def assess_mantel(self, plot_descriptor, data_root="/data"):
    """ Traverse the output and populate the database with completed results.

    :param self: Allows interacting with celery
    :param plot_descriptor: Abbreviated string describing plot's underlying data
    :param data_root: default /data, base path to all of the results
    """

    progress_recorder = ProgressRecorder(self)

    rdict, rdf = calculate_individual_stats(
        plot_descriptor, progress_recorder, progress_from=0, progress_to=74, data_root=data_root
    )
    print("{} records for Mantel assessment.".format(len(rdf)))

    progress_recorder.set_progress(74, 100, "Generating plot")
    f_train_test, axes = plot_all_train_vs_test(
        rdf, title="Mantels: {}s, split by {}, {}-masked, {}-ranked, {}-normed, by {}, top-{}".format(
            rdict['parby'], rdict['splby'], rdict['mask'], rdict['algo'], rdict['norm'], plot_descriptor[:3].upper(),
            'peak' if rdict['threshold'] is None else rdict['threshold']
        ),
        fig_size=(12, 12), y_min=-0.15, y_max=0.90
    )
    f_train_test.savefig(os.path.join(data_root, "plots", "{}_mantel.png".format(plot_descriptor.lower())))

    progress_recorder.set_progress(78, 100, "Generating figure 2")
    f_2, axes = plot_fig_2(
        rdf, shuffles=['none', 'dist', 'be04', 'agno', ],
        title="Mantels: {}s, split by {}, {}-masked, {}-ranked, {}-normed, by {}, top-{}".format(
            rdict['parby'], rdict['splby'], rdict['mask'], rdict['algo'], rdict['norm'], plot_descriptor[:3].upper(),
            'peak' if rdict['threshold'] is None else rdict['threshold']
        ),
        fig_size=(10, 6), y_min=-0.15, y_max=0.90
    )
    f_2.savefig(os.path.join(data_root, "plots", "{}_fig_2.png".format(plot_descriptor.lower())))

    progress_recorder.set_progress(82, 100, "Generating figure 3")
    f_3, axes = plot_fig_3(
        rdf, shuffles=['none', 'dist', 'be04', 'agno', ],
        title="Mantels: {}s, split by {}, {}-masked, {}-ranked, {}-normed, by {}, top-{}".format(
            rdict['parby'], rdict['splby'], rdict['mask'], rdict['algo'], rdict['norm'], plot_descriptor[:3].upper(),
            'peak' if rdict['threshold'] is None else rdict['threshold']
        ),
        fig_size=(8, 6), y_min=-0.1, y_max=0.8
    )
    f_3.savefig(os.path.join(data_root, "plots", "{}_fig_3.png".format(plot_descriptor.lower())))

    progress_recorder.set_progress(86, 100, "Generating text")
    description = describe_mantel(
        rdf, descriptor=plot_descriptor.lower(),
        title="Mantels: {}s, split by {}, {}-masked, {}-ranked, {}-normed, by {}, top-{}".format(
            rdict['parby'], rdict['splby'], rdict['mask'], rdict['algo'], rdict['norm'], plot_descriptor[:3].upper(),
            'peak' if rdict['threshold'] is None else rdict['threshold']
        ),
    )
    with open(os.path.join(data_root, "plots", "{}_mantel.html".format(rdict['descriptor'])), 'w') as f:
        f.write(description)

    progress_recorder.set_progress(90, 100, "Building probe to gene map")

    """ Write out relevant gene lists as html. """
    gene_description = write_gene_lists(rdict, rdf, progress_recorder, data_root)
    with open(os.path.join(data_root, "plots", "{}_genes.html".format(rdict['descriptor'])), 'w') as f:
        f.write(gene_description)

    progress_recorder.set_progress(100, 100, "Finished")


@shared_task(bind=True)
def assess_overlap(self, plot_descriptor, data_root="/data"):
    """ Traverse the output and populate the database with completed results.

    :param self: Allows interacting with celery
    :param plot_descriptor: Abbreviated string describing plot's underlying data
    :param data_root: default /data, base path to all of the results
    """

    progress_recorder = ProgressRecorder(self)

    rdict, rdf = calculate_individual_stats(
        plot_descriptor, progress_recorder, progress_from=0, progress_to=80, data_root=data_root
    )

    """ Calculate grouped stats, overlap within each group of tsv files. """
    post_file = os.path.join(data_root, "plots", "cache", "{}_ol_post.df".format(plot_descriptor.lower()))
    if os.path.isfile(post_file):
        with open(post_file, 'rb') as f:
            rdf = pickle.load(f)
    else:
        progress_recorder.set_progress(80, 100, "Within-shuffle overlap")
        rdf['split'] = rdf['path'].apply(extract_seed, args=("batch-train",))
        rdf['seed'] = rdf['path'].apply(extract_seed, args=("_seed-",))

        """ Calculate similarity within split-halves and within shuffle-seeds. """
        for shuffle in list(set(rdf['shuf'])):
            shuffle_mask = rdf['shuf'] == shuffle
            # We can only do this in training data, unless we want to double the workload above for test, too.
            rdf.loc[shuffle_mask, 'train_overlap'] = algorithms.pct_similarity_list(
                list(rdf.loc[shuffle_mask, 'path']), top=rdict['threshold']
            )

            # For each shuffled result, compare it against same-shuffled results from the same split
            for split in list(set(rdf['split'])):
                split_mask = rdf['split'] == split
                rdf.loc[shuffle_mask & split_mask, 'overlap_by_split'] = algorithms.pct_similarity_list(
                    list(rdf.loc[shuffle_mask & split_mask, 'path']), top=rdict['threshold']
                )

            # For each shuffled result, compare it against same-shuffled results from the same shuffle seed
            for seed in list(set(rdf['seed'])):
                seed_mask = rdf['seed'] == seed
                rdf.loc[shuffle_mask & seed_mask, 'overlap_by_seed'] = algorithms.pct_similarity_list(
                    list(rdf.loc[shuffle_mask & seed_mask, 'path']), top=rdict['threshold']
                )

            """ For each result in actual split-half train data, compare it to its shuffles. """
            rdf['real_tsv_from_shuffle'] = rdf['path'].replace("shuf-" + shuffle, "shuf-none")
            rdf.loc[shuffle_mask, 'real_v_shuffle_overlap'] = rdf.loc[shuffle_mask, :].apply(
                lambda x: algorithms.pct_similarity([x.path, x.real_tsv_from_shuffle], top=rdict['threshold']), axis=1)

    rdf.to_pickle(post_file, protocol=4)

    progress_recorder.set_progress(90, 100, "Generating plot")
    print("Plotting overlaps with {} threshold(s).".format(len(set(rdf['threshold']))))
    f_overlap, axes = plot_overlap(
        rdf, shuffles=['none', 'dist', 'be04', 'agno', ],
        title="Overlaps: {}s, split by {}, {}-masked, {}-ranked, by {}, top-{}".format(
            rdict['parby'], rdict['splby'], rdict['mask'], rdict['algo'], plot_descriptor[:3].upper(),
            'peak' if rdict['threshold'] is None else rdict['threshold']
        ),
        fig_size=(12, 7), y_min=0.0, y_max=1.0,
    )
    f_overlap.savefig(os.path.join(data_root, "plots", "{}_overlap.png".format(plot_descriptor.lower())))

    progress_recorder.set_progress(92, 100, "Generating figure 4")
    f_4, axes = plot_fig_4(
        rdf, shuffles=['none', 'dist', 'be04', 'agno', ],
        title="Overlaps: {}s, split by {}, {}-masked, {}-ranked, {}-normed, by {}, top-{}".format(
            rdict['parby'], rdict['splby'], rdict['mask'], rdict['algo'], rdict['norm'], plot_descriptor[:3].upper(),
            'peak' if rdict['threshold'] is None else rdict['threshold']
        ), y_min=0.0, y_max=1.0,
        fig_size=(8, 6),
    )
    f_4.savefig(os.path.join(data_root, "plots", "{}_fig_4.png".format(plot_descriptor.lower())))

    progress_recorder.set_progress(95, 100, "Generating description")
    description = describe_overlap(
        rdf, descriptor=plot_descriptor.lower(),
        title="Overlaps: {}s, split by {}, {}-masked, {}-ranked, by {}, top-{}".format(
            rdict['parby'], rdict['splby'], rdict['mask'], rdict['algo'], plot_descriptor[:3].upper(),
            'peak' if rdict['threshold'] is None else rdict['threshold']
        ),
    )
    with open(os.path.join(data_root, "plots", "{}_overlap.html".format(plot_descriptor.lower())), "w") as f:
        f.write(description)

    progress_recorder.set_progress(100, 100, "Finished")


@shared_task(bind=True)
def assess_performance(self, plot_descriptor, data_root="/data"):
    """ Calculate metrics at different thresholds for relevant genes and plot them.

    :param self: Allows interacting with celery
    :param plot_descriptor: Abbreviated string describing plot's underlying data
    :param data_root: default /data, base path to all of the results
    """

    progress_recorder = ProgressRecorder(self)
    rdict, rdf = calculate_individual_stats(
        plot_descriptor, progress_recorder, progress_from=0, progress_to=2, data_root=data_root
    )

    # It's tough to pass a list of thresholds via url, so it's hard-coded here.
    thresholds = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 48, 64, 80,
                  96, 128, 157, 160, 192, 224, 256, 298, 320, 352, 384, 416, 448, 480, 512, ]
    total_pieces = rdict['n'] * len(thresholds)

    # Determine an end-point, correlating to 100%
    # There are two sections: first, results * thresholds; second, overlaps, which will just have to normalize to 50/50
    progress_recorder.set_progress(2, 100, "Finding results")
    print("Found {:,} results ({} {} {} {} {} {} {})".format(
        rdict['n'], "glasser", "fornito", rdict['algo'], rdict['comp'], rdict['parby'], rdict['splby'], rdict['mask']
    ))

    if rdict['n'] > 0:
        post_file = os.path.join(data_root, "plots", "cache", "{}_ap_post.df".format(plot_descriptor.lower()))
        if os.path.isfile(post_file):
            with open(post_file, "rb") as f:
                rdf = pickle.load(f)
        else:

            relevant_results = []

            """ Calculate (or load) stats for individual tsv files. """
            for path_idx, path in enumerate(rdict['query_set'].values('tsv_path')):
                if os.path.isfile(path['tsv_path']):
                    for threshold_idx, threshold in enumerate(thresholds):
                        relevant_results.append(
                            results_as_dict(
                                path['tsv_path'], base_path=data_root,
                                probe_sig_threshold=None if threshold == 0 else threshold
                            )
                        )
                        progress_recorder.set_progress(
                            2 + ((path_idx * len(thresholds) + threshold_idx) / (total_pieces * 2)) * 88, 100,
                            "Processing {:,}/{:,} results".format(path_idx, rdict['n'])
                        )
                else:
                    print("ERR: DOES NOT EXIST: {}".format(path['tsv_path']))
            rdf = pd.DataFrame(relevant_results)

            # Just temporarily, in case debugging offline is necessary
            os.makedirs(os.path.join(data_root, "plots", "cache"), exist_ok=True)
            rdf.to_pickle(
                os.path.join(data_root, "plots", "cache", "{}_ap_pre.df".format(plot_descriptor.lower())), protocol=4
            )

            """ Calculate grouped stats, overlap between each tsv file and its shuffle-based 'peers'. """
            progress_recorder.set_progress(90, 100, "Generating overlap lists")
            rdf['split'] = rdf['path'].apply(extract_seed, args=("batch-train", ))
            splits = list(set(rdf['split']))
            shuffles = list(set(rdf['shuf']))
            calcs = len(splits) * len(shuffles)
            for shuffle_idx, shuffle in enumerate(shuffles):
                shuffle_mask = rdf['shuf'] == shuffle
                for split_idx, split in enumerate(splits):
                    progress_recorder.set_progress(
                        total_pieces + int(((shuffle_idx * len(splits) + split_idx) / calcs) * (total_pieces / 2)),
                        total_pieces * 2 + calcs,
                        "overlap list {}:{}/{}:{}".format(shuffle_idx, len(shuffles), split_idx, len(splits))
                    )
                    split_mask = rdf['split'] == split
                    # We can only do this in training data, unless we want to double the prior workload for test, too.
                    # Generate a list of lists for each file's single 'train_overlap' cell in our dataframe.
                    # rdf here is typically a (num thresholds *) 784-row df of each result.tsv path and its scores.
                    # We apply functions to the 'none'-shuffled rows, but pass them the shuffled rows for comparison
                    rdf["t_mantel_" + shuffle] = rdf[rdf['shuf'] == 'none'].apply(
                        calc_ttests, axis=1, df=rdf[shuffle_mask & split_mask]
                    )
                    rdf["overlap_real_vs_" + shuffle] = rdf[rdf['shuf'] == 'none'].apply(
                        calc_real_v_shuffle_overlaps, axis=1, df=rdf[shuffle_mask & split_mask]
                    )
                    # rdf["tau_real_vs_" + shuffle] = rdf[rdf['shuffle'] == 'none'].apply(
                    #     calc_total_overlap, axis=1, df=rdf[shuffle_mask & split_mask]
                    # )
                # rdf["complete_overlap_vs_" + shuffle] = rdf.apply(calc_total_overlap, axis=1, df=rdf[shuffle_mask])

            # Just temporarily, in case debugging offline is necessary
            rdf.to_pickle(post_file, protocol=4)

        progress_recorder.set_progress(95, 100, "Generating plot")
        # Plot performance of thresholds on correlations.

        f_full_perf, a_full_perf = plot_performance_over_thresholds(
            rdf[(rdf['phase'] == rdict['phase']) & (rdf['shuf'] == 'none')],
        )
        # f_full_perf.savefig(os.path.join(data_root, "plots", "{}_performance.png".format(plot_descriptor[: -4])))
        f_full_perf.savefig(os.path.join(data_root, "plots", "{}_performance.png".format(plot_descriptor)))

        progress_recorder.set_progress(100, 100, "Finished")

    return None
