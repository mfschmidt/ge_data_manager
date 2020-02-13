import os
from statistics import mean
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

from pygest import algorithms
from pygest.rawdata import miscellaneous
from pygest.convenience import create_symbol_to_id_map, create_id_to_symbol_map, get_ranks_from_file


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

    if len(tsvs) < 1:
        return None

    print("    These {:,} results average {:0.2%} overlap.".format(
        len(tsvs), algorithms.pct_similarity(tsvs, map_probes_to_genes_first=False, top=top)
    ))

    # Collect raw rankings of all probes in all tsvs
    rs = []
    for i, tsv in enumerate(tsvs):
        rs.append(get_ranks_from_file(tsv, column_name="rank{:04d}".format(i))["rank{:04d}".format(i)])
    dfr = pd.concat(rs, axis=1)

    # Replace NaNs (genes in at least one list, but not all) with the worst rankings in each column
    # A NaN represents a score too low to make the given list and should penalize the mean ranking
    # This should no longer ever happen, as we no longer use truncated gene lists.
    if dfr.isnull().values.any():
        "ACK! truncating probes in genes.py:ranked_probes(); Why?!?!"
        for col in dfr.columns:
            # Replace this column's nans with this column's highest (worst) rankings
            print("column has {} values ranging from {} to {}; {} are NaN. Replace them with {}".format(
                len(dfr),
                dfr[col].min(), dfr[col].max(),
                len(dfr.loc[np.isnan(dfr.loc[:, col]), col]),
                len(list(range(int(max(dfr[col]) + 1), len(dfr) + 1)))
            ))
            dfr.loc[np.isnan(dfr.loc[:, col]), col] = range(int(max(dfr[col]) + 1), len(dfr) + 1)

    dfr['mean'] = dfr.mean(axis=1)
    dfr['entrez_id'] = dfr.index.map(miscellaneous.map_pid_to_eid_fornito)
    return dfr.sort_values('mean', ascending=True)


def describe_three_relevant_overlaps(relevant_results, phase, threshold):
    """ Filter results by the arguments, and report percent similarity and consistent top genes. """
    print("=== {} ===".format(phase))
    unshuffled_results = relevant_results[relevant_results['shuffle'] == 'none']
    print("  {:,} out of {:,} results are unshuffled.".format(len(unshuffled_results), len(relevant_results)))

    for which_phase in ["train", "test", ]:
        phased_results = unshuffled_results[unshuffled_results['phase'] == which_phase]
        if len(phased_results) > 0:
            print("  Overlap between 16 random ({}) halves, @{} = {:0.1%}; kendall tau is {:0.03}".format(
                phase, threshold,
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


def rank_genes_respecting_shuffles(real_files, shuffle_files):
    """ Given gene rankings from actual data and gene rankings from shuffled data,
        calculate how likely each gene is to have outperformed the shuffle in actual data.
    """

    # Rearrange results to use ranking, ordered by probe_id index
    reals = ranked_probes(real_files, None).sort_index()
    shuffles = ranked_probes(shuffle_files, None).sort_index()

    # Count how many shuffled rankings, for each gene, are better than the real data.
    hits = shuffles[[col for col in shuffles.columns if "rank" in col]].lt(reals['mean'], axis=0)
    n = len(hits.columns)
    hits['hits'] = hits.sum(axis=1)
    hits['p'] = hits['hits'] / n

    # The delta is how much better the gene performs in real data than shuffled, higher positive is better
    hits['delta'] = shuffles['mean'] - reals['mean']

    return hits.sort_values(by='p')


def describe_genes(rdf, rdict, progress_recorder):
    """ Create a description of the top genes. """

    progress_recorder.set_progress(86, 100, "Building gene maps")
    sym_id_map = create_symbol_to_id_map()
    id_sym_map = create_id_to_symbol_map()

    progress_recorder.set_progress(90, 100, "Ranking genes")
    relevant_tsvs = list(describe_three_relevant_overlaps(rdf, rdict['phase'], rdict['threshold'])['path'])
    survivors = pervasive_probes(relevant_tsvs, rdict['threshold'])
    all_ranked = ranked_probes(relevant_tsvs, rdict['threshold'])
    all_ranked['rank'] = range(1, len(all_ranked.index) + 1, 1)
    all_ranked['probe_id'] = all_ranked.index

    output = ["<p>Comparison of Mantel correlations between the test half (using probes discovered in the train half) vs connectivity similarity.</p>",
              "<ol>"]
    # Calculate p with a t-test between real and shuffled.
    actuals = rdf[rdf['shuffle'] == 'none']
    for shf in ['be16', 'be08', 'be04', 'edge', 'dist', 'agno']:
        # 'test_score' is in the test set, and is most appropriate and conservative; could do 'train_score', too
        shuffles = rdf[rdf['shuffle'] == shf]
        if len(shuffles) > 0 and len(actuals) > 0:
            t, p = ttest_ind(actuals['test_score'].values, shuffles['test_score'].values)
            output.append("  <li>discovery in real vs discovery in {} (t-test): t = {:0.2f}, p = {:0.5f}</li>".format(shf, t, p))
    output.append("</ol>")

    progress_recorder.set_progress(92, 100, "Calculating p-values per gene")
    # Calculate p for each gene by counting how many times its shuffled rank outperforms its real rank.
    df_gene_ps = rank_genes_respecting_shuffles(list(actuals['path']), list(actuals['path'])).sort_index()
    p_lines = []
    d_lines = []
    for shf in ['be16', 'be08', 'be04', 'edge', 'dist', 'agno']:
        shuffles = rdf[rdf['shuffle'] == shf]
        if len(shuffles) > 0 and len(actuals) > 0:
            tmpdf = rank_genes_respecting_shuffles(list(actuals['path']), list(shuffles['path'])).sort_index()

            df_gene_ps["p_" + shf] = tmpdf['p']
            low_p_indices = all_ranked['probe_id'].isin(tmpdf[tmpdf['p'] < 0.05].index)
            top_by_p = all_ranked[low_p_indices]
            print("{} items in top_by_p (real outperforms shuffled @ p<0.05)".format(len(top_by_p)))

            df_gene_ps["delta_" + shf] = tmpdf['delta']
            high_delta_indices = all_ranked['probe_id'].isin(tmpdf[tmpdf['delta'] > 1000].index)
            top_by_delta = all_ranked[high_delta_indices]
            print("{} items in top_by_delta (real ranks >1000 places better than shuffled)".format(len(top_by_delta)))

            p_lines.append("{}: {:,} {} {}-shuffled data (p<0.05) ({:,} if p<0.01). {} {:,} genes is {:,}".format(
                shf,
                len(tmpdf[tmpdf['p'] < 0.05].index),
                "genes perform better in actual than",
                shf,
                len(tmpdf[tmpdf['p'] < 0.01].index),
                "The average ranking of these",
                len(tmpdf[tmpdf['p'] < 0.05].index),
                int(top_by_p['rank'].mean()),
            ))
            d_lines.append("{}: {:,} {} {}-shuffled data. ({:,} > 2000). {} {:,} genes is {:,}".format(
                shf,
                len(tmpdf[tmpdf['delta'] > 1000].index),
                "genes perform 1000 spots better in actual than",
                shf,
                len(tmpdf[tmpdf['delta'] > 2000].index),
                "The average ranking of these",
                len(tmpdf[tmpdf['delta'] > 1000].index),
                int(top_by_delta['rank'].mean()),
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
    all_ranked = pd.concat([all_ranked, df_gene_ps], axis=1)

    """ Next, for the description file, report top genes. """
    progress_recorder.set_progress(95, 100, "Summarizing genes")
    line1 = "Top {} genes from {} {}, {}-ranked by-{}, mask={}".format(
        '"peak"' if rdict['threshold'] is None else rdict['threshold'],
        len(relevant_tsvs), rdict['phase'], rdict['algo'], rdict['splby'], rdict['mask']
    )
    line2 = "This is a ranking of ALL probes, so the selected threshold does not change it."
    output.append("<p>" + line1 + " " + line2 + "</p>\n  <ol>")
    # print(line)
    all_ranked = all_ranked.sort_values("rank")
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
        output.append("    <li value=\"{}\">{}</li>".format(all_ranked.loc[p, 'rank'], item_string))
        # print("{}. {}".format(all_ranked.loc[p, 'rank'], item_string))
    output.append("  </ol>\n</p>")
    output.append("<div id=\"notes_below\">")
    output.append("    <p>Asterisks indicate probes making the top {} in all 16 splits.</p>".format(rdict['threshold']))
    output.append("</div>")

    # Return the all_ranked dataframe, with two key-like Series as the first columns
    ordered_columns = ["entrez_id", "probe_id", ] + \
                      [item for item in all_ranked.columns if item not in ["entrez_id", "probe_id", ]]
    return all_ranked[ordered_columns], "\n".join(output)
