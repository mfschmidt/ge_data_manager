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
from .genes import describe_genes


class NullHandler(logging.Handler):
    def emit(self, record):
        pass
null_handler = NullHandler()

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
    """ Convert a string from the json file, like "5 days, 2:45:32.987", into integer seconds. """
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


def derivative_path(shuffle_path):
    """ from a given path to a shuffled result, return the path to the real result it corresponds with. """
    if "derivatives" in shuffle_path:
        return shuffle_path

    new_path = shuffle_path
    shuffles = [
        "edgeshuffles", "distshuffles", "edge04shuffles", "edge08shuffles", "edge16shuffles", "shuffles",
    ]
    # "shuffles" MUST be last in this list, or it will match prior shuffle types and partially replace them.
    for s_to_replace in shuffles:
        new_path = new_path.replace(s_to_replace, "derivatives")
    return new_path[: -15] + ".tsv"


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
    shuffles = [
        'derivatives', 'shuffles', 'distshuffles', 'edgeshuffles',
        'edge04shuffles', 'edge08shuffles', 'edge16shuffles',
    ]
    for shuffle in shuffles:
        d = os.path.join(data_path, shuffle)
        if os.path.isdir(d):
            for root, dirs, files in os.walk(d, topdown=True):
                for f in files:
                    if (f[(-4):] == "json") and (not "." in f[:-5]):
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

    print("{:,} results from {}...".format(len(results), data_path))

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

        split_key = "batch-test" if "batch-test" in result['path'] else "batch-train"

        # Finally, put it all into a model for storage in the database.
        r = PushResult(
            descriptor=build_descriptor(
                result.get("comp", ""), result.get("splby", ""), result.get("mask", ""),
                result.get("norm", ""), result.get("batch", ""),
            ),
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
            split=extract_seed(os.path.join(result['path'], result['json_file'].replace("json", "tsv")), split_key),
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
            num_edge04shuffles=PushResult.objects.filter(shuffle='edge04shuffles').count(),
            num_edge08shuffles=PushResult.objects.filter(shuffle='edge08shuffles').count(),
            num_edge16shuffles=PushResult.objects.filter(shuffle='edge16shuffles').count(),
            num_splits=0,
        )
        s.save()


shuffle_dir_map = {
    'none': 'derivatives',
    'edge': 'edgeshuffles',
    'be04': 'edge04shuffles',
    'be08': 'edge08shuffles',
    'be16': 'edge16shuffles',
    'dist': 'distshuffles',
    'agno': 'shuffles',
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
        'f__g': 'fearglassersim',
        'f__w': 'fearsim',
        'n__g': 'neutralglassersim',
        'n__w': 'neutralsim',
        'fn_g': 'fearneutralglassersim',
        'fn_w': 'fearneutralsim',
        'glasserconnectivitysim': 'glasser-connectivity_sim.df',
        'hcpniftismoothgrandmeansim': 'hcp_niftismooth_grandmean_sim.df',
        'indiglasserconnsim': 'indi-glasser-conn_sim.df',
        'indiconnsim': 'indi-connectivity_sim.df',
        'fearglassersim': 'fear_glasser_sim.df',
        'fearsim': 'fear_sim.df',
        'neutralglassersim': 'neutral_glasser_sim.df',
        'neutralsim': 'neutral_sim.df',
        'fearneutralglassersim': 'fear-neutral_glasser_sim.df',
        'fearneutralsim': 'fear-neutral_sim.df',
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
                mask_trues, mask_falses, np.count_nonzero(mask_vector.isnan())
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
        v_mask = one_mask(expr, mask, bids_val('parby', tsv_file), base_path)
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
    analyzed_path = tsv_file.replace(".tsv", ".top-{}.v3.json".format(
        "peak" if probe_sig_threshold is None else "{:04}".format(probe_sig_threshold)
    ))
    if use_cache and os.path.isfile(analyzed_path):
        saved_dict = json.load(open(analyzed_path, 'r'))
        return saved_dict

    # There's no cached copy; go ahead and calculate everything.
    if "/shuffles/" in tsv_file:
        shuffle = 'agno'
    elif "/distshuffles/" in tsv_file:
        shuffle = 'dist'
    elif "/edgeshuffles/" in tsv_file:
        shuffle = 'edge'
    elif "/edge04shuffles/" in tsv_file:
        shuffle = 'be04'
    elif "/edge08shuffles/" in tsv_file:
        shuffle = 'be08'
    elif "/edge16shuffles/" in tsv_file:
        shuffle = 'be16'
    else:
        shuffle = 'none'

    mask = bids_val('mask', tsv_file)
    norm = bids_val('norm', tsv_file)

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
        'shuffle': shuffle,
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
        'seed': extract_seed(tsv_file, "_seed-"),
        'real_tsv_from_shuffle': derivative_path(tsv_file),
    })

    # Cache results to prevent repeated recalculation of the same thing.
    json.dump(result_dict, open(analyzed_path, 'w'))

    return result_dict


def build_descriptor(comp, splitby, mask, normalization, batch):
    """ Generate a shorthand descriptor for the group a result belongs to. """

    # From actual file, or from path to result, boil down comparator to its abbreviation
    comp_map = {
        'glasser-connectivity_sim.df': 'hcpg',
        'glasserconnectivitysim': 'hcpg',
        'hcp_niftismooth_grandmean_sim.df': 'hcpw',
        'hcpniftismoothgrandmeansim': 'hcpw',
        'indi-glasser-conn_sim.df': 'nkig',
        'indiglasserconnsim': 'nkig',
        'indi-connectivity_sim.df': 'nkiw',
        'indiconnsim': 'nkiw',
        'fear_glasser_sim.df': 'f__g',
        'fearglassersim': 'f__g',
        'fear_sim.df': 'f__w',
        'fearsim': 'f__w',
        'neutral_glasser_sim.df': 'n__g',
        'neutralglassersim': 'n__g',
        'neutral_sim.df': 'n__w',
        'neutralsim': 'n__w',
        'fear-neutral_glasser_sim.df': 'fn_g',
        'fearneutralglassersim': 'fn_g',
        'fear-neutral_sim.df': 'fn_w',
        'fearneutralsim': 'fn_w',
    }

    # Make short string for split seed and normalization
    split = int(batch[-5:])
    if 200 <= split < 300:
        xv = "2"
    elif 400 <= split < 500:
        xv = "4"
    else:
        xv = "_"
    norm = "s" if normalization == "srs" else "_"

    # Build and return the descriptor
    return "{}{}{:0>2}{}{}{}".format(
        comp_map[comp],
        splitby[0],
        0 if mask == "none" else int(mask),
        's',
        norm,
        xv,
    )


def interpret_descriptor(descriptor):
    """ Parse the plot descriptor into parts """
    comp = comp_from_signature(descriptor[:4])
    parby = "glasser" if descriptor[3].lower() == "g" else "wellid"
    splby = "glasser" if descriptor[4].lower() == "g" else "wellid"
    mask = 'none' if descriptor[5:7] == "00" else descriptor[5:7]
    algo = 'once' if descriptor[7] == "o" else "smrt"
    norm = 'srs' if ((len(descriptor) > 8) and (descriptor[8] == "s")) else "none"
    xval = descriptor[9] if len(descriptor) > 9 else '0'
    phase = 'train'
    opposite_phase = 'test'

    try:
        threshold = int(descriptor[8:])
    except ValueError:
        # This catches "peak" or "", both should be fine as None
        threshold = None

    # The xval, or cross-validation sample split range indicator,
    # is '2' for splits in the 200's and '4' for splits in the 400s.
    # Splits in the 200's are split-halves; 400's are split-quarters.
    split_min = 0
    split_max = 999
    if xval == '2':
        split_min = 200
        split_max = 299
    if xval == '4':
        split_min = 400
        split_max = 499
    relevant_results_queryset = PushResult.objects.filter(
        samp="glasser", prob="fornito", algo=algo, comp=comp, parby=parby, splby=splby, mask=mask, norm=norm,
        batch__startswith=phase, split__gte=split_min, split__lte=split_max
    )

    return {
        'comp': comp,
        'parby': parby,
        'splby': splby,
        'mask': mask,
        'algo': algo,
        'norm': norm,
        'xval': xval,
        'phase': phase,
        'opposite_phase': opposite_phase,
        'threshold': threshold,
        'query_set': relevant_results_queryset,
        'n': relevant_results_queryset.count(),
        'descriptor': descriptor.lower(),
    }


def calc_ttests(row, df):
    """ for a given dataframe row, if it's a real result, return its t-test t-value vs all shuffles in df. """
    if row.shuffle == 'none':
        return stats.ttest_1samp(
            df[(df['threshold'] == row.threshold)]['train_score'],
            row.train_score,
        )[0]
    else:
        return 0.0


def calc_real_v_shuffle_overlaps(row, df):
    """ for a given dataframe row, if it's a real result, return its overlap pctage vs all shuffles in df. """
    if row.shuffle == 'none':
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
    if row.shuffle == 'none':
        overlaps = []
        for shuffled_tsv in df[(df['threshold'] == row.threshold)]['path']:
            overlaps.append(algorithms.pct_similarity([row.path, shuffled_tsv], top=row.threshold))
        return np.mean(overlaps)
    else:
        return 0.0


def calculate_individual_stats(
        descriptor, progress_recorder, progress_from=0, progress_to=99, data_root="/data", use_cache=True
):
    """ Determine the list of results necessary, and build or load a dataframe around them. """

    rdict = interpret_descriptor(descriptor)
    progress_recorder.set_progress(progress_from, progress_to, "Step 1/3<br />Finding results")
    print("Found {:,} results ({} {} {} {} {} {} {} {}) @{}".format(
        rdict['n'], "glasser", "fornito",
        rdict['algo'], rdict['comp'], rdict['parby'], rdict['splby'], rdict['mask'], rdict['norm'],
        'peak' if rdict['threshold'] is None else rdict['threshold'],
    ))

    cache_file = os.path.join(data_root, "plots", "cache", "{}_summary_individual.df".format(rdict['descriptor']))
    if use_cache and os.path.isfile(cache_file):
        """ Load results from a cached file, if possible"""
        with open(cache_file, "rb") as f:
            df = pickle.load(f)
    else:
        """ Calculate results for individual tsv files. """
        relevant_results = []
        for i, path in enumerate(rdict['query_set'].values('tsv_path')):
            if os.path.isfile(path['tsv_path']):
                relevant_results.append(
                    results_as_dict(path['tsv_path'], base_path=data_root, probe_sig_threshold=rdict['threshold'])
                )
            else:
                print("ERR: DOES NOT EXIST: {}".format(path['tsv_path']))
            complete_portion = ((i + 1) / rdict['n']) * (progress_to - progress_from)
            progress_recorder.set_progress(
                progress_from + complete_portion, 100,
                "Step 1/3<br />Processing {:,}/{:,} results".format(i, rdict['n'])
            )

        df = pd.DataFrame(relevant_results)

        os.makedirs(os.path.join(data_root, "plots", "cache"), exist_ok=True)
        df.to_pickle(cache_file)

    progress_recorder.set_progress(
        progress_to - 1, 100,
        "Step 1/3<br />Processed {:,} results".format(rdict['n'])
    )

    return rdict, df


def calculate_group_stats(
        rdict, rdf, progress_recorder, progress_from=0, progress_to=99, data_root="/data", use_cache=True
):
    """ Using meta-data from each result, calculate statistics between results and about the entire group. """

    progress_recorder.set_progress(progress_from, 100, "Step 2/3<br />1. Within-shuffle overlap")
    post_file = os.path.join(data_root, "plots", "cache", "{}_summary_group.df".format(rdict['descriptor']))
    if use_cache and os.path.isfile(post_file):
        """ Load results from a cached file, if possible"""
        with open(post_file, 'rb') as f:
            rdf = pickle.load(f)
    else:
        """ Calculate similarity within split-halves and within shuffle-seeds. """
        n = len(set(rdf['shuffle']))
        for i, shuffle in enumerate(list(set(rdf['shuffle']))):
            """ This explores the idea of viewing similarity between same-split-seed runs and same-shuffle-seed runs
                vs all shuffled runs of the type. All three of these are "internal" or "intra-list" overlap.
            """
            shuffle_mask = rdf['shuffle'] == shuffle
            local_df = rdf.loc[shuffle_mask, :]
            local_n = len(local_df)
            print("Calculating percent overlaps and Kendall taus for {} {}-shuffles".format(local_n, shuffle))
            progress_recorder.set_progress((100.0 * i / n), 100, "Step 2/3<br />2. {}-shuffle similarity".format(shuffle))
            progress_delta = float(progress_to - progress_from) * 0.25 / n

            """ Calculate overlaps and such, only if there are results available to calculate. """
            if local_n > 0:
                # We can only do this in training data, unless we want to double the workload for test, too.
                rdf.loc[shuffle_mask, 'train_overlap'] = algorithms.pct_similarity_list(
                    list(local_df['path']), top=rdict['threshold']
                )
                rdf.loc[shuffle_mask, 'train_ktau'] = algorithms.kendall_tau_list(
                    list(local_df['path']),
                )

                progress_recorder.set_progress(
                    (100.0 * i / n) + (progress_delta * 1), 100, "Step 2/3<br />3. {}-shuffle seeds".format(shuffle)
                )
                # For each shuffled result, compare it against same-shuffled results from the same split
                for split in list(set(local_df['split'])):
                    # These values will all be 'nan' for unshuffled results. There's only one per split.
                    split_mask = rdf['split'] == split
                    if sum(split_mask) > 0:
                        print("    overlap and ktau of {} {}-split {}-shuffles".format(sum(split_mask), split, shuffle))
                        # The following similarities are masked to include only one shuffle type, and one split-half
                        rdf.loc[shuffle_mask & split_mask, 'overlap_by_split'] = algorithms.pct_similarity_list(
                            list(local_df.loc[split_mask, 'path']), top=rdict['threshold']
                        )
                        rdf.loc[shuffle_mask & split_mask, 'ktau_by_split'] = algorithms.kendall_tau_list(
                            list(local_df.loc[split_mask, 'path']),
                        )

                progress_recorder.set_progress(
                    (100.0 * i / n) + (progress_delta * 2), 100, "Step 2/3<br />4. {}-shuffle seeds".format(shuffle)
                )
                # For each shuffled result, compare it against same-shuffled results from the same shuffle seed
                for seed in list(set(local_df['seed'])):
                    seed_mask = rdf['seed'] == seed
                    if sum(seed_mask) > 0:
                        print("    overlap and ktau of {} {}-seed {}-shuffles".format(sum(seed_mask), seed, shuffle))
                        rdf.loc[shuffle_mask & seed_mask, 'overlap_by_seed'] = algorithms.pct_similarity_list(
                            list(local_df.loc[seed_mask, 'path']), top=rdict['threshold']
                        )
                        rdf.loc[shuffle_mask & seed_mask, 'ktau_by_seed'] = algorithms.kendall_tau_list(
                            list(local_df.loc[seed_mask, 'path']),
                        )

                progress_recorder.set_progress(
                    (100.0 * i / n) + (progress_delta * 3), 100, "Step 2/3<br />5. {}-shuffle seeds".format(shuffle)
                )
                """ For each result in actual split-half train data, compare it to its shuffles. """
                # Each unshuffled run will match itself only for these two, resulting in a 1.0 perfect comparison.
                # So we overwrite them with a better comparison
                rdf.loc[shuffle_mask, 'real_v_shuffle_overlap'] = local_df.apply(lambda x:
                    algorithms.pct_similarity([x.path, x.real_tsv_from_shuffle], top=rdict['threshold']), axis=1
                )
                rdf.loc[shuffle_mask, 'real_v_shuffle_ktau'] = local_df.apply(lambda x:
                    algorithms.kendall_tau([x.path, x.real_tsv_from_shuffle]), axis=1
                )

    progress_recorder.set_progress(progress_to, 100, "Step 2/3<br />Similarity calculated")
    print("Pickling {}".format(post_file))
    rdf.to_pickle(post_file)

    return rdf


def write_gene_lists(rdict, rdf, progress_recorder, data_root="/data"):
    """ Rank genes and write them out as two csvs. """
    df_ranked_full, gene_description = describe_genes(rdf, rdict, progress_recorder)
    df_ranked_full.to_csv(os.path.join(data_root, "plots", "{}_ranked_full.csv".format(rdict['descriptor'])))
    df_ranked_full[['entrez_id', ]].to_csv(os.path.join(data_root, "plots", "{}_ranked.csv".format(rdict['descriptor'])))

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
    rdict, rdf = calculate_individual_stats(
        plot_descriptor, progress_recorder, progress_from=0, progress_to=99, data_root=data_root
    )
    print("{} records for full analysis.".format(len(rdf)))

    """ 2. Calculate grouped stats, overlap within each group of tsv files.
           This, too, caches the resultant calculations.
    """
    rdf = calculate_group_stats(
        rdict, rdf, progress_recorder, progress_from=0, progress_to=99, data_root=data_root
    )

    i = 0  # i is the index into how many plots and reports to build, for reporting progress
    n = 6

    """ 3. Plot aggregated data. """
    plot_title = "Mantels: {}s, split by {}, {}-masked, {}-ranked, {}-normed, by {}, top-{}".format(
        rdict['parby'], rdict['splby'], rdict['mask'], rdict['algo'], rdict['norm'], plot_descriptor[:3].upper(),
        'peak' if rdict['threshold'] is None else rdict['threshold']
    )
    progress_recorder.set_progress(100.0 * i / n, 100, "Step 3/3<br />Plotting figure 2")
    f_2, axes = plot_fig_2(rdf, title=plot_title, fig_size=(12, 6), y_min=-0.15, y_max=0.50)
    f_2.savefig(os.path.join(data_root, "plots", "{}_fig_2.png".format(plot_descriptor.lower())))
    i += 1

    progress_recorder.set_progress(100.0 * i / n, 100, "Step 3/3<br />Plotting figure 3")
    f_3, axes = plot_fig_3(rdf, title=plot_title, fig_size=(6, 5), y_min=-0.1, y_max=0.50)
    f_3.savefig(os.path.join(data_root, "plots", "{}_fig_3.png".format(plot_descriptor.lower())))
    i += 1

    progress_recorder.set_progress(100.0 * i / n, 100, "Step 3/3<br />Plotting figure 4")
    f_4, axes = plot_fig_4(rdf, title=plot_title, y_min=0.0, y_max=0.80, fig_size=(13, 5),)
    f_4.savefig(os.path.join(data_root, "plots", "{}_fig_4.png".format(plot_descriptor.lower())))
    i += 1

    """ 4. Describe results in text form. """
    progress_recorder.set_progress(100.0 * i / n, 100, "Step 3/3<br />Writing result text")
    mantel_description = describe_mantel(rdf, descriptor=plot_descriptor.lower(), title=plot_title, )
    i += 1

    progress_recorder.set_progress(100.0 * i / n, 100, "Step 3/3<br />Writing gene lists")
    gene_description = write_gene_lists(rdict, rdf, progress_recorder, data_root="/data")
    i += 1

    progress_recorder.set_progress(100.0 * i / n, 100, "Step 3/3<br />Writing overlap description")
    overlap_description = describe_overlap(rdf, descriptor=plot_descriptor.lower(), title=plot_title, )

    with open(os.path.join(data_root, "plots", "{}_report.html".format(plot_descriptor.lower())), 'w') as f:
        f.write("<h1>Mantel Correlations</h1>\n")
        f.write(mantel_description)
        f.write("<h1>Overlap Descriptions</h1>\n")
        f.write(overlap_description)
        f.write("<h1>Probe/Gene Descriptions</h1>\n")
        f.write(gene_description)
    i += 1

    progress_recorder.set_progress(99, 100, "Step 3/3. Plotting finished")


@shared_task(bind=True)
def assess_mantel(self, plot_descriptor, data_root="/data"):
    """ Traverse the output and populate the database with completed results.

    :param self: Allows interacting with celery
    :param plot_descriptor: Abbreviated string describing plot's underlying data
    :param data_root: default /data, base path to all of the results
    """

    progress_recorder = ProgressRecorder(self)

    rdict, rdf = calculate_individual_stats(plot_descriptor, progress_recorder, progress_from=0, progress_to=74, data_root=data_root)
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
        rdf, title="Mantels: {}s, split by {}, {}-masked, {}-ranked, {}-normed, by {}, top-{}".format(
            rdict['parby'], rdict['splby'], rdict['mask'], rdict['algo'], rdict['norm'], plot_descriptor[:3].upper(),
            'peak' if rdict['threshold'] is None else rdict['threshold']
        ),
        fig_size=(10, 6), y_min=-0.15, y_max=0.90
    )
    f_2.savefig(os.path.join(data_root, "plots", "{}_fig_2.png".format(plot_descriptor.lower())))

    progress_recorder.set_progress(82, 100, "Generating figure 3")
    f_3, axes = plot_fig_3(
        rdf, title="Mantels: {}s, split by {}, {}-masked, {}-ranked, {}-normed, by {}, top-{}".format(
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
        for shuffle in list(set(rdf['shuffle'])):
            shuffle_mask = rdf['shuffle'] == shuffle
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
            rdf['real_tsv_from_shuffle'] = rdf['path'].apply(derivative_path)
            rdf.loc[shuffle_mask, 'real_v_shuffle_overlap'] = rdf.loc[shuffle_mask, :].apply(
                lambda x: algorithms.pct_similarity([x.path, x.real_tsv_from_shuffle], top=rdict['threshold']), axis=1)

    rdf.to_pickle(post_file)

    progress_recorder.set_progress(90, 100, "Generating plot")
    print("Plotting overlaps with {} threshold(s).".format(len(set(rdf['threshold']))))
    f_overlap, axes = plot_overlap(
        rdf,
        title="Overlaps: {}s, split by {}, {}-masked, {}-ranked, by {}, top-{}".format(
            rdict['parby'], rdict['splby'], rdict['mask'], rdict['algo'], plot_descriptor[:3].upper(),
            'peak' if rdict['threshold'] is None else rdict['threshold']
        ),
        fig_size=(12, 7), y_min=0.0, y_max=1.0,
    )
    f_overlap.savefig(os.path.join(data_root, "plots", "{}_overlap.png".format(plot_descriptor.lower())))

    progress_recorder.set_progress(92, 100, "Generating figure 4")
    f_4, axes = plot_fig_4(
        rdf, title="Overlaps: {}s, split by {}, {}-masked, {}-ranked, {}-normed, by {}, top-{}".format(
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

    # It's tough to pass a list of thresholds via url, so it's hard-coded here.
    thresholds = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 48, 64, 80,
                  96, 128, 157, 160, 192, 224, 256, 298, 320, 352, 384, 416, 448, 480, 512, ]
    progress_recorder = ProgressRecorder(self)

    rdict, rdf = calculate_individual_stats(
        plot_descriptor, progress_recorder, progress_from=0, progress_to=2, data_root=data_root
    )

    # Determine an end-point, correlating to 100%
    # There are two sections: first, results * thresholds; second, overlaps, which will just have to normalize to 50/50
    total = rdict['n'] * len(thresholds) * 2  # The *2 allows for later overlaps to constitute 50% of the reported percentage
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
            for i, path in enumerate(rdict['query_set'].values('tsv_path')):
                if os.path.isfile(path['tsv_path']):
                    for j, threshold in enumerate(thresholds):
                        relevant_results.append(
                            results_as_dict(
                                path['tsv_path'], base_path=data_root,
                                probe_sig_threshold=None if threshold == 0 else threshold
                            )
                        )
                        progress_recorder.set_progress(
                            2 + ((i * len(thresholds) + j) / total) * 88, 100,
                            "Processing {:,}/{:,} results".format(i, rdict['n'])
                        )
                else:
                    print("ERR: DOES NOT EXIST: {}".format(path['tsv_path']))
            rdf = pd.DataFrame(relevant_results)

            # Just temporarily, in case debugging offline is necessary
            os.makedirs(os.path.join(data_root, "plots", "cache"), exist_ok=True)
            rdf.to_pickle(os.path.join(data_root, "plots", "cache", "{}_ap_pre.df".format(plot_descriptor.lower())))

            """ Calculate grouped stats, overlap between each tsv file and its shuffle-based 'peers'. """
            progress_recorder.set_progress(90, 100, "Generating overlap lists")
            rdf['split'] = rdf['path'].apply(extract_seed, args=("batch-train", ))
            splits = list(set(rdf['split']))
            shuffles = list(set(rdf['shuffle']))
            calcs = len(splits) * len(shuffles)
            for k, shuffle in enumerate(shuffles):
                shuffle_mask = rdf['shuffle'] == shuffle
                for l, split in enumerate(splits):
                    progress_recorder.set_progress(
                        (rdict['n'] * len(thresholds)) + int(((k * len(splits) + l) / calcs) * (rdict['n'] * len(thresholds) / 2)),
                        total + calcs,
                        "overlap list {}:{}/{}:{}".format(k, len(shuffles), l, len(splits))
                    )
                    split_mask = rdf['split'] == split
                    # We can only do this in training data, unless we want to double the prior workload for test, too.
                    # Generate a list of lists for each file's single 'train_overlap' cell in our dataframe.
                    # At this point, rdf is typically a (num thresholds *) 784-row dataframe of each result.tsv path and its scores.
                    # We apply these functions to only the 'none'-shuffled rows, but pass them the shuffled rows for comparison
                    rdf["t_mantel_" + shuffle] = rdf[rdf['shuffle'] == 'none'].apply(
                        calc_ttests, axis=1, df=rdf[shuffle_mask & split_mask]
                    )
                    rdf["overlap_real_vs_" + shuffle] = rdf[rdf['shuffle'] == 'none'].apply(
                        calc_real_v_shuffle_overlaps, axis=1, df=rdf[shuffle_mask & split_mask]
                    )
                    # rdf["tau_real_vs_" + shuffle] = rdf[rdf['shuffle'] == 'none'].apply(
                    #     calc_total_overlap, axis=1, df=rdf[shuffle_mask & split_mask]
                    # )
                # rdf["complete_overlap_vs_" + shuffle] = rdf.apply(calc_total_overlap, axis=1, df=rdf[shuffle_mask])

            # Just temporarily, in case debugging offline is necessary
            rdf.to_pickle(post_file)

        progress_recorder.set_progress(95, 100, "Generating plot")
        # Plot performance of thresholds on correlations.

        f_full_perf, a_full_perf = plot_performance_over_thresholds(
            rdf[(rdf['phase'] == rdict['phase']) & (rdf['shuffle'] == 'none')],
        )
        # f_full_perf.savefig(os.path.join(data_root, "plots", "{}_performance.png".format(plot_descriptor[: -4])))
        f_full_perf.savefig(os.path.join(data_root, "plots", "{}_performance.png".format(plot_descriptor)))

        progress_recorder.set_progress(100, 100, "Finished")

    return None

