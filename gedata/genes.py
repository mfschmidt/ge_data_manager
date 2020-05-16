import os
from statistics import mean
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ranksums
import re

from pygest import algorithms
from pygest.rawdata import miscellaneous
from pygest.convenience import create_symbol_to_id_map, create_id_to_symbol_map, get_ranks_from_file


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


def ranked_probes(tsvs, rank_type, top):
    """ Go through the files provided in tsvs, at the threshold specified in top, and report probes in all files.

        :param list tsvs: A list of result files
        :param rank_type: a string to label/discriminate different groups of results
        :param top: Threshold for how many of the top genes to consider relevant
        :returns: dataframe with probes as rows, and a column of probe ranks for each result, plus a 'mean' column.
    """

    if len(tsvs) < 1:
        return None

    print("    Ranking genes in {:,} results, which average {:0.2%} overlap.".format(
        len(tsvs), algorithms.pct_similarity(tsvs, map_probes_to_genes_first=False, top=top)
    ))

    # Collect raw rankings of all probes in all tsvs
    rs = []
    for i, tsv in enumerate(tsvs):
        # Decide on a unique column name for this particular gene ranking
        # Looking for path like ...batch-train00400...
        m_split = re.search(r"(?P<sub>{})-train(?P<val>[a-zA-Z0-9+]+)".format("batch"), tsv)
        split = int(m_split.group("val")) if m_split is not None else None
        m_seed = re.search(r"(?P<sub>{})-(?P<val>[0-9+]+)".format("seed"), tsv)
        seed = int(m_seed.group("val")) if m_seed is not None else None
        column_name = "{}_rank_{:04d}".format(
            rank_type, split if split is not None else 32 + i,
        )
        if seed is not None:
            column_name = "{}-{:04d}".format(column_name, seed)

        # Use the calculated column name for this ranking
        rs.append(get_ranks_from_file(tsv, column_name=column_name)[column_name])
    dfr = pd.concat(rs, axis=1)

    # Replace NaNs (genes in at least one list, but not all) with the worst rankings in each column
    # A NaN represents a score too low to make the given list and should penalize the mean ranking
    # This should no longer ever happen, as we no longer use truncated gene lists.
    if dfr.isnull().values.any():
        print("ACK! truncating probes in genes.py:ranked_probes(); Why?!?!")
        for col in dfr.columns:
            # Replace this column's nans with this column's highest (worst) rankings
            print("column has {} values ranging from {} to {}; {} are NaN. Replace them with {}".format(
                len(dfr),
                dfr[col].min(), dfr[col].max(),
                len(dfr.loc[np.isnan(dfr.loc[:, col]), col]),
                len(list(range(int(max(dfr[col]) + 1), len(dfr) + 1)))
            ))
            dfr.loc[np.isnan(dfr.loc[:, col]), col] = range(int(max(dfr[col]) + 1), len(dfr) + 1)

    dfr[rank_type + '_mean'] = dfr.mean(axis=1)
    if 'entrez_id' not in dfr.columns:
        dfr['entrez_id'] = dfr.index.map(miscellaneous.map_pid_to_eid_fornito)
    return dfr.sort_values(rank_type + '_mean', ascending=True)


def describe_three_relevant_overlaps(relevant_results, phase, threshold):
    """ Filter results by the arguments, and report percent similarity and consistent top genes.
    """

    print("=== {} ===".format(phase))
    unshuffled_results = relevant_results[relevant_results['shuffle'] == 'none']
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


def rank_genes_respecting_shuffles(real_files, shuffle_files, shuffle_name):
    """ Given gene rankings from actual data and gene rankings from shuffled data,
        calculate how likely each gene is to have outperformed the shuffle in actual data.
    """

    # Rearrange results to use ranking, ordered by probe_id index
    # Get a separate dataframe of probe rankings for each result for reals and shuffles
    reals = ranked_probes(real_files, "reals", None).sort_index()
    shuffles = ranked_probes(shuffle_files, "shufs", None).sort_index()

    print("Writing {} reals and {} shuffles, named {} to csv. <genes.py:rank_genes_respecting_shuffles()>".format(
        reals.shape, shuffles.shape, shuffle_name
    ))
    reals.to_csv("/data/plots/cache/_reals_{}_.csv".format(shuffle_name))
    shuffles.to_csv("/data/plots/cache/_shufs_{}_.csv".format(shuffle_name))

    new_df = pd.DataFrame(data=None, index=reals.index)

    # For speed, create a separate curated dataframe for each split, then apply a function rather than for-looping
    # dfs_by_split = {}
    for split in sorted([int(col[11:]) for col in reals.columns if "reals_rank_" in col]):  # should be 1 to 32
        # dfs_by_split[split] = {}
        # dfs_by_split[split]['reals'] = reals["reals_rank_{:04}".format(split)]
        shuffle_columns = [col for col in shuffles.columns if "shufs_rank_{:04}".format(split) in col]
        print("  found {} ranks to compare for split {}".format(len(shuffle_columns), split))
        # dfs_by_split[split]['shuffles'] = shuffles[shuffle_columns]

        # Values compared here are ranks, 1 being best. So counts are how many times genes scored better
        # in shuffled data than in real data.
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
    new_df['p_by-count_for-' + shuffle_name] = new_df['n_' + shuffle_name + '_gt-realmean'] / len(hits.columns)
    new_df['delta_for-' + shuffle_name] = shuffles['shufs_mean'] - reals['reals_mean']

    # new_df.to_csv("/data/plots/cache/_hits_" + shuffle_name + "_.csv")

    return new_df


def describe_genes(rdf, rdict, progress_recorder):
    """ Create a description of the top genes. """

    progress_recorder.set_progress(86, 100, "Building gene maps")
    sym_id_map = create_symbol_to_id_map()
    id_sym_map = create_id_to_symbol_map()

    progress_recorder.set_progress(90, 100, "Ranking genes")
    relevant_tsvs = list(describe_three_relevant_overlaps(rdf, rdict['phase'], rdict['threshold'])['path'])
    survivors = pervasive_probes(relevant_tsvs, rdict['threshold'])

    # Calculate the raw ranking of all genes over all splits, and create a new 'raw_rank' to summarize them.
    all_ranked = ranked_probes(relevant_tsvs, "raw", rdict['threshold'])
    all_ranked['raw_rank'] = range(1, len(all_ranked.index) + 1, 1)
    all_ranked['probe_id'] = all_ranked.index

    # all_ranked.to_csv("/data/plots/cache/intermediate_a.csv")

    output = ["<p>Comparison of Mantel correlations between the test half (using probes discovered in the train half) vs connectivity similarity.</p>",
              "<ol>"]
    # Calculate p with a t-test between real and shuffled.
    actuals = rdf[rdf['shuffle'] == 'none']
    for shf in list(set(rdf[rdf['shuffle'] != 'none']['shuffle'])):
        # 'test_score' is in the test set, and is most appropriate and conservative; could do 'train_score', too
        shuffles = rdf[rdf['shuffle'] == shf]
        if len(shuffles) > 0 and len(actuals) > 0:
            tt, pt = ttest_ind(actuals['test_score'].values, shuffles['test_score'].values)
            tr, pr = ranksums(actuals['test_score'].values, shuffles['test_score'].values)
            output.append("  <li>discovery in real vs discovery in {} ({}) ({})</li>".format(shf,
                "t-test: t = {:0.2f}, p = {:0.5f}".format(tt, pt),
                "ranksum-test: t = {:0.2f}, p = {:0.5f}".format(tr, pr),
            ))
    output.append("</ol>")

    progress_recorder.set_progress(92, 100, "Calculating p-values per gene")
    # Calculate delta and p for each gene by counting how many times its shuffled rank outperforms its real rank.
    p_lines = []
    d_lines = []
    for shf in list(set(rdf[rdf['shuffle'] != 'none']['shuffle'])):
        shuffles = rdf[rdf['shuffle'] == shf]
        if len(shuffles) > 0 and len(actuals) > 0:
            tmpdf = rank_genes_respecting_shuffles(list(actuals['path']), list(shuffles['path']), shf).sort_index()
            all_ranked = pd.concat([all_ranked, tmpdf], axis=1)

            # Extract p-values.
            low_p_indices = all_ranked['probe_id'].isin(tmpdf[tmpdf['p_by-count_for-' + shf] < 0.05].index)
            top_by_p = all_ranked[low_p_indices]
            print("{} items in top_by_p (real outperforms shuffled @ p<0.05)".format(len(top_by_p)))

            # Extract deltas.
            high_delta_indices = all_ranked['probe_id'].isin(tmpdf[tmpdf['delta_for-' + shf] > 1000].index)
            top_by_delta = all_ranked[high_delta_indices]
            print("{} items in top_by_delta (real ranks >1000 places better than shuffled)".format(len(top_by_delta)))

            p_lines.append("{}: {:,} {} {}-shuffled data (p<0.05) ({:,} if p<0.01). {} {:,} genes is {:,}".format(
                shf,
                len(tmpdf[tmpdf['p_by-count_for-' + shf] < 0.05].index),
                "genes perform better in actual than",
                shf,
                len(tmpdf[tmpdf['p_by-count_for-' + shf] < 0.01].index),
                "The average ranking of these",
                len(tmpdf[tmpdf['p_by-count_for-' + shf] < 0.05].index),
                0 if len(top_by_p) < 1 else int(top_by_p['raw_rank'].mean()),
            ))
            d_lines.append("{}: {:,} {} {}-shuffled data. ({:,} > 2000). {} {:,} genes is {:,}".format(
                shf,
                len(tmpdf[tmpdf['delta_for-' + shf] > 1000].index),
                "genes perform 1000 spots better in actual than",
                shf,
                len(tmpdf[tmpdf['delta_for-' + shf] > 2000].index),
                "The average ranking of these",
                len(tmpdf[tmpdf['delta_for-' + shf] > 1000].index),
                0 if len(top_by_delta) < 1 else int(top_by_delta['raw_rank'].mean()),
            ))
        else:
            crap_out_line = "{}: {} actual and {} {}-shuffled results, no comparisons available.".format(
                shf, len(actuals), len(shuffles), shf
            )
            p_lines.append(crap_out_line)
            d_lines.append(crap_out_line)

    output.append("<p>{}</p>".format(" ".join([
        "Some genes survive optimization longer in real data than shuffled.",
        "<ul><li>", "</li><li>".join(p_lines), "</li></ul>",
        "<ul><li>", "</li><li>".join(d_lines), "</li></ul>",
        "That doesn't mean they performed well, just better in actual than shuffled.",
        "See {}_ranked_full.csv for quantitative details.".format(rdict['descriptor']),
    ])))
    # This is unnecessary as df_gene_ps is no longer created.
    # all_ranked = pd.concat([all_ranked, df_gene_ps], axis=1)

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
            p, gene_id_string, gene_symbol_string, all_ranked.loc[p, 'raw_mean'] + 1.0, asterisk
        )
        output.append("    <li value=\"{}\">{}</li>".format(all_ranked.loc[p, 'raw_rank'], item_string))
        # print("{}. {}".format(all_ranked.loc[p, 'rank'], item_string))
    output.append("  </ol>\n</p>")
    output.append("<div id=\"notes_below\">")
    output.append("    <p>Asterisks indicate probes making the top {} in all 16 splits.</p>".format(rdict['threshold']))
    output.append("</div>")

    # Return the all_ranked dataframe, with two key-like Series as the first columns
    ordered_columns = ["entrez_id", "probe_id", ] + \
                      sorted([item for item in all_ranked.columns if "_id" not in item and "rank" not in item]) + \
                      sorted([item for item in all_ranked.columns if "rank" in item]) + \
                      sorted([item for item in all_ranked.columns if "_id" in item])
    return all_ranked[ordered_columns + [col for col in all_ranked.columns if col not in ordered_columns]], "\n".join(output)
