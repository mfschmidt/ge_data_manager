import os
from statistics import mean
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ranksums
import re

from pygest import algorithms
from pygest.rawdata import miscellaneous
from pygest.convenience import create_symbol_to_id_map, create_id_to_symbol_map, get_ranks_from_file
from pygest.erminej import get_ranks_from_ejgo_file

from .decorators import print_duration


@print_duration
def pervasive_results(files, threshold):
    """ Go through the files provided, at the threshold specified, and report ids in all files. """

    print("    These results average {:0.2%} overlap.".format(
        algorithms.pct_similarity(files, map_probes_to_genes_first=False, top=threshold)
    ))
    hitters = {}
    for i, f in enumerate(files):
        if ".ejgo" in f:
            fn = algorithms.run_ontology
            fld = 'top_gos'
        else:
            fn = algorithms.run_results
            fld = 'top_probes'
        if i == 0:
            hitters = set(fn(f, top=threshold)[fld])
        else:
            hitters = hitters.intersection(set(fn(f, top=threshold)[fld]))
        if i == len(files):
            print("    {} ids remain in all {} results.".format(len(hitters), i + 1))

    return pd.DataFrame({'id': list(hitters)})


def column_name_from_file(path, prefix, seq):
    """ From a path to a file and a preset prefix, generate a unique column name for this file's results.

        :param str path: The path to a file
        :param str prefix: A prefix to start the column name
        :param int seq: A number to sequentially identify the source of the column's data
        :returns str: A string to use as a column name
    """

    # Decide on a unique column name for this particular gene ranking
    # Looking for path like ...batch-train00400...
    m_split = re.search(r"(?P<sub>{})-train(?P<val>[a-zA-Z0-9+]+)".format("batch"), path)
    split = int(m_split.group("val")) if m_split is not None else None
    m_seed = re.search(r"(?P<sub>{})-(?P<val>[0-9+]+)".format("seed"), path)
    seed = int(m_seed.group("val")) if m_seed is not None else None
    column_name = "{}_rank_{:04d}".format(
        prefix, split if split is not None else 32 + seq,
    )
    if seed is not None:
        column_name = "{}-{:04d}".format(column_name, seed)

    return column_name


@print_duration
def ranked_probes(tsvs, rank_type, top):
    """ Go through the files provided in tsvs, at the threshold specified in top, and report probes in all files.

        :param list tsvs: A list of result files
        :param rank_type: a string to label/discriminate different groups of results
        :param top: Threshold for how many of the top genes to consider relevant
        :returns: dataframe with probes as rows, and a column of probe ranks for each tsv, plus a 'mean' column.
    """

    if len(tsvs) < 1:
        return None

    print("    Ranking genes in {:,} results, which average {:0.2%} overlap.".format(
        len(tsvs), algorithms.pct_similarity(tsvs, map_probes_to_genes_first=False, top=top)
    ))

    # Collect raw rankings of all probes in all tsvs
    rs = []
    for i, tsv in enumerate(tsvs):
        # Use the calculated column name for this ranking
        column_name = column_name_from_file(tsv, rank_type, i)
        df = get_ranks_from_file(tsv)
        rs.append(df.rename(columns={'rank': column_name})[column_name])
    df_rankings = pd.concat(rs, axis=1)

    # Replace NaNs (genes in at least one list, but not all) with the worst rankings in each column
    # A NaN represents a score too low to make the given list and should penalize the mean ranking
    # This should no longer ever happen, as we no longer use truncated gene lists.
    if df_rankings.isnull().values.any():
        print("ACK! truncating probes in genes.py:ranked_probes(); Why?!?!")
        for col in df_rankings.columns:
            # Replace this column's nans with this column's highest (worst) rankings
            print("column has {} values ranging from {} to {}; {} are NaN. Replace them with {}".format(
                len(df_rankings),
                df_rankings[col].min(), df_rankings[col].max(),
                len(df_rankings.loc[np.isnan(df_rankings.loc[:, col]), col]),
                len(list(range(int(max(df_rankings[col]) + 1), len(df_rankings) + 1)))
            ))
            df_rankings.loc[np.isnan(df_rankings.loc[:, col]), col] = range(
                int(max(df_rankings[col]) + 1), len(df_rankings) + 1
            )

    df_rankings[rank_type + '_mean'] = df_rankings.mean(axis=1)
    if 'entrez_id' not in df_rankings.columns:
        df_rankings['entrez_id'] = df_rankings.index.map(miscellaneous.map_pid_to_eid_fornito)
    return df_rankings.sort_values(rank_type + '_mean', ascending=True)


@print_duration
def ranked_ontologies(ejgos, rank_type):
    """ Go through the files provided in ejgos, and return a dataframe of terms with ranks in all files.

        :param list ejgos: A list of result files
        :param rank_type: a string to label/discriminate different groups of results
        :returns: dataframe with GO terms as rows, and a column of ranks for each ejgo, plus a 'mean' column.
    """

    if len(ejgos) < 1:
        return None

    print("    Ranking ontologies in {:,} results.".format(len(ejgos)))

    # Collect raw rankings of all probes in all ejgos
    rankings = []
    for i, ejgo in enumerate(ejgos):
        column_name = column_name_from_file(ejgo, rank_type, i)
        # Use the calculated column name for this ranking
        df = get_ranks_from_ejgo_file(ejgo)
        rankings.append(df.rename(columns={'rank': column_name})[column_name])
    df_rankings = pd.concat(rankings, axis=1)

    df_rankings[rank_type + '_mean'] = df_rankings.mean(axis=1)

    return df_rankings.sort_values(rank_type + '_mean', ascending=True)


@print_duration
def describe_three_relevant_overlaps(relevant_results, phase, threshold):
    """ Filter results by the arguments, and report percent similarity, kendall tau and consistent top genes
        for 'train', 'test', 'train vs test'.

    :param relevant_results:
    :param phase:
    :param threshold:
    :return pd.DataFrame: dataframe containing un-shuffled results
    """

    print("=== {} ===".format(phase))
    unshuffled_results = relevant_results[relevant_results['shuf'] == 'none']
    print("  {:,} out of {:,} results are unshuffled.".format(len(unshuffled_results), len(relevant_results)))

    for which_phase in ["train", "test", ]:
        phased_results = unshuffled_results[unshuffled_results['phase'] == which_phase]
        if len(phased_results) > 0:
            print("  Overlap between {} random ({}) halves, @{} = {:0.1%}; kendall tau is {:0.03}".format(
                len(phased_results), phase, threshold,
                algorithms.pct_similarity(list(phased_results['path']), map_probes_to_genes_first=False, top=threshold),
                algorithms.kendall_tau(list(phased_results['path']))
            ))
        else:
            print("  No {} results for overlap or kendall tau calculations @{}.".format(phase, threshold))

    acrosses = []
    for t in unshuffled_results[unshuffled_results['phase'] == 'train']['path']:
        comps = [t, t.replace('train', 'test'), ]
        comps = [f for f in comps if os.path.isfile(f)]
        if len(comps) > 0:
            olap = algorithms.pct_similarity(comps, map_probes_to_genes_first=False, top=threshold)
            acrosses.append(olap)
            # print("    {:0.2%}% - {}".format(olap, t))
    print("  Overlap between each direct train-vs-test pair @{} = {:0.1%}".format(
        threshold, mean(acrosses)
    ))
    return unshuffled_results


@print_duration
def rank_respecting_shuffles(real_files, shuffle_files, shuffle_name):
    """ Given gene rankings from actual data and gene rankings from shuffled data,
        calculate how likely each gene is to have outperformed the shuffle in actual data.
    """

    # Determine the files' extensions to determine whether to parse tsv results or ejgo results
    ext = real_files[0][real_files[0].rfind("."):][1:]
    # Rearrange results to use ranking, ordered by probe_id index
    # Get a separate dataframe of probe rankings for each result for reals and shuffles
    if ext == "tsv":
        reals = ranked_probes(real_files, "reals", None).sort_index()
        shuffles = ranked_probes(shuffle_files, "shufs", None).sort_index()
    elif ext.startswith("ejgo"):
        reals = ranked_ontologies(real_files, "reals").sort_index()
        shuffles = ranked_ontologies(shuffle_files, "shufs").sort_index()
    else:
        print("The {} extension is not understood, boycotting!! <genes.py:rank_respecting_shuffles()>".format(ext))
        return None

    print("Writing {} real and {} shuffled {}, named {} to csv. <genes.py:rank_respecting_shuffles()>".format(
        reals.shape, shuffles.shape, ext, shuffle_name
    ))
    reals.to_csv("/data/plots/cache/_ranked_real_{}_{}_.csv".format(ext[:3], shuffle_name))
    shuffles.to_csv("/data/plots/cache/_ranked_shuf_{}_{}_.csv".format(ext[:3], shuffle_name))

    new_df = pd.DataFrame(data=None, index=reals.index)

    # For speed, create a separate curated dataframe for each split, then apply a function rather than for-looping
    # dfs_by_split = {}
    for split in sorted([int(col[11:]) for col in reals.columns if "reals_rank_" in col]):  # should be 1 to 32
        # dfs_by_split[split] = {}
        # dfs_by_split[split]['reals'] = reals["reals_rank_{:04}".format(split)]
        shuffle_columns = [col for col in shuffles.columns if "shufs_rank_{:04}".format(split) in col]
        print("  found {} ranks to compare for split {}".format(len(shuffle_columns), split))
        # dfs_by_split[split]['shufs'] = shuffles[shuffle_columns]

        # Values compared here are ranks, 1 being best. So counts are how many times genes scored better
        # in shuffled data than in each split's real data.
        new_df['{}-over-real_{:03}_count'.format(shuffle_name, split)] = shuffles[shuffle_columns].lt(
            reals["reals_rank_{:04}".format(split)], axis='index'
        ).sum(axis='columns')

    # And, without regard to split, compare real to shuffles.
    reals_ranks = reals[[col for col in reals.columns if "reals_rank_" in col]]

    def ttests_vs_real(row):
        t_ranksums, p_ranksums = ranksums(row.values, reals_ranks.loc[row.name, :].values)
        t_ttestind, p_ttestind = ttest_ind(row.values, reals_ranks.loc[row.name, :].values)
        return t_ranksums, p_ranksums, t_ttestind, p_ttestind

    ttest_df = shuffles[[col for col in shuffles.columns if "shufs_rank_" in col]].apply(
        lambda row: ttests_vs_real(row), axis=1
    ).apply(pd.Series)
    ttest_df.columns = ["{}_for-{}".format(test, shuffle_name) for test in [
        't_by-ranksums', 'p_by-ranksums', 't_by-ttestind', 'p_by-ttestind'
    ]]
    ttest_df.index = shuffles.index
    new_df = pd.concat([new_df, ttest_df, ], axis='columns')

    # Count how many shuffled rankings, for each gene, are better than the real data.
    hits = shuffles[[col for col in shuffles.columns if "shufs_rank" in col]].lt(reals['reals_mean'], axis=0)
    new_df['n_' + shuffle_name + '_gt-realmean'] = hits.sum(axis=1)
    new_df['p_by-ave-count_for-' + shuffle_name] = new_df['n_' + shuffle_name + '_gt-realmean'] / len(hits.columns)
    new_df['delta_for-' + shuffle_name] = shuffles['shufs_mean'] - reals['reals_mean']

    # Count within-split outperformance
    cols_to_sum = [col for col in new_df.columns if col.startswith("{}-over-real_".format(shuffle_name))]
    new_df['p_by-ind-count_for-' + shuffle_name] = new_df[cols_to_sum].sum(axis='columns') / len(shuffle_files)
    print("  counted {} real and {} shuffle files".format(len(real_files), len(shuffle_files)))

    # new_df.to_csv("/data/plots/cache/_hits_" + shuffle_name + "_.csv")

    return new_df


def extract_description_for(values, ranks, thresholds, comparison_string):
    """ Report values from df_with_ranks[col] by thresholds.

    :param values: Series containing values, indexed by probe_id
    :param ranks: Series containing ranks, indexed by probe_id
    :param thresholds: Thresholds for reporting out those values
    :param comparison_string: A string used to describe the comparison underlying the p-value
    :return: descriptive text
    """

    if len(thresholds) < 1:
        return "No thresholds requested"

        # Calculate a quantity of values under each threshold,
        # and pair the quantity with the threshold.
    quants = list(zip([(values < t).sum() for t in thresholds], thresholds))
    print("{} values less than {} in {}".format(*quants[0], values.name, ))

    return "{}: {:,} {} {}-{}, p<{} ({}). {} of these {:,} genes is {:,}".format(
        values.name[-4:], quants[0][0], "genes perform better in",
        values.name[-4:], comparison_string, thresholds[0],
        ", ".join(["{:,} <{:0.2f}".format(*q) for q in quants]),
        comparison_string, quants[0][0],
        0 if quants[0][0] < 1 else int(ranks[values[values < 0.05].index].mean()),
    )


def p_real_shuffle_ttest(rdf, comparator):
    """ Calculate p values between real and each shuffle via t-test.

    :param rdf: dataframe with 'shuf' field for comparing 'none' to all other groups and comparator value field
    :param comparator: field containing numeric data for comparisions between groups
    :return: list of html <li> strings describing the comparisons
    """

    output = []
    actuals = rdf[rdf['shuf'] == 'none']
    for shf in list(set(rdf[rdf['shuf'] != 'none']['shuf'])):
        shuffles = rdf[rdf['shuf'] == shf]
        if len(shuffles) > 0 and len(actuals) > 0:
            tt, pt = ttest_ind(actuals[comparator].values, shuffles[comparator].values)
            tr, pr = ranksums(actuals[comparator].values, shuffles[comparator].values)
            output.append("  <li>discovery in real vs discovery for {} in {} ({}) ({})</li>".format(
                comparator, shf,
                "t-test: t = {:0.2f}, p = {:0.5f}".format(tt, pt),
                "ranksum-test: t = {:0.2f}, p = {:0.5f}".format(tr, pr),
            ))
    return output


def p_real_shuffle_count(rdf, all_ranked, path_field='path'):
    """ Calculate p values between real and shuffle via counting outperformance.
    """

    actuals = rdf[rdf['shuf'] == 'none']
    ave_p_lines = []
    ind_p_lines = []
    d_lines = []
    for shf in list(set(rdf[rdf['shuf'] != 'none']['shuf'])):
        shuffles = rdf[rdf['shuf'] == shf]
        if len(shuffles) > 0 and len(actuals) > 0:
            all_ranked = pd.concat([
                all_ranked,
                rank_respecting_shuffles(list(actuals[path_field]), list(shuffles[path_field]), shf).sort_index()
            ], axis=1)

            ave_p_lines.append(extract_description_for(
                all_ranked['p_by-ave-count_for-' + shf], all_ranked['raw_rank'], (0.05, 0.01),
                "the average real ranking",
            ))
            ind_p_lines.append(extract_description_for(
                all_ranked['p_by-ind-count_for-' + shf], all_ranked['raw_rank'], (0.05, 0.01),
                "each shuffle's re-sampling source"
            ))
            d_lines.append(extract_description_for(
                all_ranked['delta_for-' + shf], all_ranked['raw_rank'], (1000, 2000, 5000),
                "the average ranking"
            ))

        else:
            crap_out_line = "{}: {} actual and {} {}-shuffled results, no comparisons available.".format(
                shf, len(actuals), len(shuffles), shf
            )
            ave_p_lines.append(crap_out_line)
            ind_p_lines.append(crap_out_line)
            d_lines.append(crap_out_line)

    return ave_p_lines, ind_p_lines, d_lines


@print_duration
def describe_genes(rdf, rdict, progress_recorder):
    """ Create a description of the top genes. """

    progress_recorder.set_progress(86, 100, "Building gene maps")
    sym_id_map = create_symbol_to_id_map()
    id_sym_map = create_id_to_symbol_map()

    progress_recorder.set_progress(90, 100, "Ranking genes")
    relevant_tsvs = list(describe_three_relevant_overlaps(rdf, rdict['phase'], rdict['threshold'])['path'])
    survivors = pervasive_results(relevant_tsvs, rdict['threshold'])
    survivors['entrez_id'] = survivors['id'].map(miscellaneous.map_pid_to_eid_fornito)

    # Calculate the raw ranking of all genes over all splits, and create a new 'raw_rank' to summarize them.
    all_ranked = ranked_probes(relevant_tsvs, "raw", rdict['threshold'])

    all_ranked.to_csv("/data/plots/cache/intermediate_a.csv")

    all_ranked['raw_rank'] = range(1, len(all_ranked.index) + 1, 1)
    # all_ranked['probe_id'] = all_ranked.index

    all_ranked.to_csv("/data/plots/cache/intermediate_b.csv")

    # Calculate p with a t-test between real and shuffled.
    # 'test_score' is in the test set, and is most appropriate and conservative; could do 'train_score', too
    output = [
        "<p>{} {} {}</p>".format(
            "Comparison of Mantel correlations between the test half",
            "(using probes discovered in the train half)",
            "vs connectivity similarity."
        ),
        "<ol>",
    ] + p_real_shuffle_ttest(rdf, 'test_score') + [
        "</ol>",
    ]

    progress_recorder.set_progress(92, 100, "Calculating p-values per gene")
    # Calculate delta and p for each gene by counting how many times its shuffled rank outperforms its real rank.
    ave_p_lines, ind_p_lines, d_lines = p_real_shuffle_count(rdf, all_ranked, path_field='path')

    all_ranked.to_csv("/data/plots/cache/intermediate_c.csv")

    output.append("<p>{}</p>".format(" ".join([
        "Relevant genes should perform better in real data than shuffled, but some don't.",
        "<ul>p-values by average real ranking<li>", "</li><li>".join(ave_p_lines), "</li></ul>",
        "<ul>p-values by individual real rankings<li>", "</li><li>".join(ind_p_lines), "</li></ul>",
        "<ul>delta values<li>", "</li><li>".join(d_lines), "</li></ul>",
        "That doesn't mean they performed well, necessarily, just better in shuffled than real.",
        "See {}_ranked_full.csv for quantitative details.".format(rdict['descriptor']),
    ])))

    """ Next, for the description file, report top genes. """
    progress_recorder.set_progress(95, 100, "Summarizing genes")
    line1 = "Top {} genes from {} {}, {}-ranked by-{}, mask={}".format(
        '"peak"' if rdict['threshold'] is None else rdict['threshold'],
        len(relevant_tsvs), rdict['phase'], rdict['algo'], rdict['splby'], rdict['mask']
    )
    line2 = "This is a ranking of ALL probes, so the selected threshold does not change it."
    output.append("<p>" + line1 + " " + line2 + "</p>\n  <ol>")
    # print(line)
    all_ranked = all_ranked.sort_values("raw_rank")
    for p in (list(all_ranked.index[0:20]) + [x for x in survivors['id'] if x not in all_ranked.index[0:20]]):
        asterisk = " *" if p in list(survivors['id']) else ""
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
            p, gene_id_string, gene_symbol_string, all_ranked.loc[p, 'raw_mean'] + 1.0, asterisk
        )
        output.append("    <li value=\"{}\">{}</li>".format(all_ranked.loc[p, 'raw_rank'], item_string))
        # print("{}. {}".format(all_ranked.loc[p, 'rank'], item_string))
    output.append("  </ol>\n</p>")
    output.append("<div id=\"notes_below\">")
    output.append("    <p>Asterisks indicate probes making the top {} in all splits.</p>".format(rdict['threshold']))
    output.append("</div>")

    # Return the all_ranked dataframe, with two key-like Series as the first columns (this just orders the columns)
    ordered_columns = [
        "entrez_id",
        *sorted([item for item in all_ranked.columns if "id" not in item and "rank" not in item]),
        *sorted([item for item in all_ranked.columns if "rank" in item]),
    ]
    remaining_columns = [col for col in all_ranked.columns if col not in ordered_columns]

    all_ranked.to_csv("/data/plots/cache/intermediate_d.csv")
    all_ranked.to_pickle("/data/plots/cache/intermediate_d.df")

    return all_ranked[ordered_columns + remaining_columns], "\n".join(output)


@print_duration
def describe_ontologies(rdf, rdict, progress_recorder):
    """

    :param rdf:
    :param rdict:
    :param progress_recorder:
    :return:
    """

    go_threshold = 0.01

    progress_recorder.set_progress(90, 100, "Ranking ontologies")

    # Find ontologies and their rankings
    relevant_gos = list(describe_three_relevant_overlaps(rdf, rdict['phase'], rdict['threshold'])['path'])
    relevant_gos = [f.replace(".tsv", ".ejgo_roc_0002-2048") for f in relevant_gos]
    survivors = pervasive_results(relevant_gos, go_threshold)
    all_ranked = ranked_ontologies(relevant_gos, "raw")
    all_ranked.to_csv("/data/plots/cache/intermediate_ontologyranks_a.csv")

    all_ranked['raw_rank'] = range(1, len(all_ranked.index) + 1, 1)
    all_ranked['go_id'] = all_ranked.index
    all_ranked.to_csv("/data/plots/cache/intermediate_ontologyranks_b.csv")

    progress_recorder.set_progress(92, 100, "Calculating p-values per GO term")
    # Calculate delta and p for each gene by counting how many times its shuffled rank outperforms its real rank.
    ave_p_lines, ind_p_lines, d_lines = p_real_shuffle_count(rdf, all_ranked, path_field='ejgo_path')

    all_ranked.to_csv("/data/plots/cache/intermediate_ontologyranks_c.csv")

    output = ["<p>{}</p>".format(" ".join([
        "Relevant ontologies should perform better in real data than shuffled, but some don't.",
        "<ul>p-values by average real ranking<li>", "</li><li>".join(ave_p_lines), "</li></ul>",
        "<ul>p-values by individual real rankings<li>", "</li><li>".join(ind_p_lines), "</li></ul>",
        "<ul>delta values<li>", "</li><li>".join(d_lines), "</li></ul>",
        "That doesn't mean they performed well, necessarily, just better in shuffled than real.",
        "See {}_ranked_full.csv for quantitative details.".format(rdict['descriptor']),
    ])), ]

    """ Next, for the description file, report top ontologies. """
    progress_recorder.set_progress(95, 100, "Summarizing ontologies")
    line1 = "Top ontologies from {} {}, {}-ranked by-{}, mask={}".format(
        len(relevant_gos), rdict['phase'], rdict['algo'], rdict['splby'], rdict['mask']
    )
    output.append("<p>" + line1 + "</p>\n  <ol>")

    all_ranked = all_ranked.sort_values("raw_rank")
    for p in (list(all_ranked.index[0:20]) + [x for x in all_ranked.index[20:] if x in list(survivors['id'])]):
        asterisk = " *" if p in list(survivors['id']) else ""
        go_id_string = "<a href=\"http://amigo.geneontology.org/amigo/term/{p}\" target=\"_blank\">{p}</a>".format(p=p)
        item_string = "GO term {}, mean rank {:0.1f}{}".format(
            go_id_string, all_ranked.loc[p, 'raw_mean'] + 1.0, asterisk
        )
        output.append("    <li value=\"{}\">{}</li>".format(all_ranked.loc[p, 'raw_rank'], item_string))
        # print("{}. {}".format(all_ranked.loc[p, 'rank'], item_string))
    output.append("  </ol>\n</p>")
    output.append("<div id=\"notes_below\">")
    output.append("    <p>Asterisks indicate ontologies with p<{} in all splits.</p>".format(go_threshold))
    output.append("</div>")

    all_ranked.to_csv("/data/plots/cache/intermediate_ontologyranks_d.csv")

    # Return the all_ranked dataframe, with two key-like Series as the first columns
    ordered_columns = [
        *sorted([item for item in all_ranked.columns if "id" not in item and "rank" not in item]),
        *sorted([item for item in all_ranked.columns if "rank" in item]),
    ]
    remaining_columns = [col for col in all_ranked.columns if col not in ordered_columns]

    all_ranked.to_csv("/data/plots/cache/intermediate_ontologyranks_e.csv")

    return all_ranked[ordered_columns + remaining_columns], "\n".join(output)
