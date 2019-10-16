from pygest.convenience import create_symbol_to_id_map, create_id_to_symbol_map
from statistics import mean
import pandas as pd
from scipy.stats import ttest_ind

from pygest import algorithms
from pygest.rawdata import miscellaneous

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

    output = ["<p>Comparison of Mantel correlations between the test half (using probes discovered in the train half) vs connectivity similarity.</p>",
              "<ol>"]
    for shf in ['edge', 'dist', 'agno']:
        t, p = ttest_ind(
            # 'test_score' is in the test set, could do 'train_score', too
            rdf[rdf['shuffle'] == 'none']['test_score'].values,
            rdf[rdf['shuffle'] == shf]['test_score'].values,
        )
        output.append("  <li>discovery in real vs discovery in {}: t = {:0.2f}, p = {:0.10f}</li>".format(shf, t, p))
    output.append("</ol>")

    """ Next, for the description file, report top genes. """
    progress_recorder.set_progress(95, 100, "Summarizing genes")
    line1 = "Top {} genes from {}, {}-ranked by-{}, mask={}".format(
        '"peak"' if rdict['threshold'] is None else rdict['threshold'],
        rdict['phase'], rdict['algo'], rdict['splby'], rdict['mask']
    )
    line2 = "This is a ranking of ALL probes, so the selected threshold does not change it."
    output.append("<p>" + line1 + " " + line2 + "</p>\n  <ol>")
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
        output.append("    <li value=\"{}\">{}</li>".format(all_ranked.loc[p, 'rank'], item_string))
        # print("{}. {}".format(all_ranked.loc[p, 'rank'], item_string))
    output.append("  </ol>\n</p>")
    output.append("<div id=\"notes_below\">")
    output.append("    <p>Asterisks indicate probes making the top {} in all 16 splits.</p>".format(rdict['threshold']))
    output.append("</div>")

    return "\n".join(output)


