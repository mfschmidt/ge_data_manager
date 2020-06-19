import numpy as np
from scipy.stats import ttest_ind, tmean, tstd

from matplotlib import pyplot as plt
import matplotlib.patches as patches

import seaborn as sns

from pygest import plot
from pygest.convenience import p_string


def mean_and_sd(numbers):
    """ Report the mean and standard deviation of a list of numbers as text. """

    return "mean {:0.4f} (sd {:0.4f}, n={:,}, range=[{:0.3f} - {:0.3f}])".format(
        tmean(numbers), tstd(numbers), len(numbers), min(numbers), max(numbers)
    )

def calc_hilo(min_val, max_val, df, cols_to_test):
    """ Return lowest and highest values from min_val and max_val if present, or calculate from df. """

    # Calculate (or blindly accept) the range of the y-axis, which must be the same for all four axes.
    if (max_val is None) and (len(df.index) > 0):
        highest_possible_score = max([max(df[col]) for col in cols_to_test])
    else:
        highest_possible_score = max_val
    if (min_val is None) and (len(df.index) > 0):
        lowest_possible_score = min([min(df[col]) for col in cols_to_test])
    else:
        lowest_possible_score = min_val

    return lowest_possible_score, highest_possible_score


def box_and_swarm(figure, placement, label, variable, data, high_score=1.0, orientation="v",
                  lim=None, ps=True, cols=4, push_ordinate_to=None):
    """ Create an axes object with a swarm plot drawn over a box plot of the same data. """

    shuffle_order = ['none', 'be04', 'agno']
    shuffle_color_boxes = sns.color_palette(['gray', 'lightcoral', 'lightblue'])
    shuffle_color_points = sns.color_palette(['black', 'red', 'blue'])
    annot_columns = [
        {'shuf': 'none', 'xo': 0.0, 'xp': 0.0},
        {'shuf': 'be04', 'xo': 1.0, 'xp': 0.5},
        {'shuf': 'agno', 'xo': 2.0, 'xp': 1.0},
    ]
    if cols == 3:
        shuffle_order = shuffle_order[1:]
        shuffle_color_boxes = shuffle_color_boxes[1:]
        shuffle_color_points = shuffle_color_points[1:]
    elif cols == 1:
        shuffle_order = shuffle_order[0:1]
        shuffle_color_boxes = shuffle_color_boxes[0:1]
        shuffle_color_points = shuffle_color_points[0:1]

    if push_ordinate_to is not None:
        data[variable] = data[variable] + push_ordinate_to - data[variable].max()

    ax = figure.add_axes(placement, label=label)
    if orientation == "v":
        sns.swarmplot(data=data, x='shuf', y=variable,
                      order=shuffle_order, palette=shuffle_color_points, size=3.0, ax=ax)
        sns.boxplot(data=data, x='shuf', y=variable, order=shuffle_order, palette=shuffle_color_boxes, ax=ax)
        ax.set_ylabel(None)
        ax.set_xlabel(label)
        if lim is not None:
            ax.set_ylim(lim)
    else:
        sns.swarmplot(data=data, x=variable, y='shuf', order=shuffle_order, palette=shuffle_color_points, ax=ax)
        sns.boxplot(data=data, x=variable, y='shuf', order=shuffle_order, palette=shuffle_color_boxes, ax=ax)
        ax.set_xlabel(None)
        ax.set_ylabel(label)
        if lim is not None:
            ax.set_xlim(lim)

    """ Calculate p-values for each column in the above plots, and annotate accordingly. """
    if ps & (orientation == "v"):
        gap = 0.06
        actual_results = data[data['shuf'] == 'none'][variable].values
        try:
            global_max_y = max(data[variable].values)
        except ValueError:
            global_max_y = high_score
        for i, col in enumerate(annot_columns):
            shuffle_results = data[data['shuf'] == col['shuf']]
            try:
                # max_y = max(data[data['phase'] == 'train'][y].values)
                local_max_y = max(shuffle_results[variable].values)
            except ValueError:
                local_max_y = high_score
            try:
                y_pval = max(max(shuffle_results[variable].values), max(actual_results)) + gap
            except ValueError:
                y_pval = high_score + gap
            try:
                t, p = ttest_ind(actual_results, shuffle_results[variable].values)
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
    elif orientation == "h":
        for i, col in enumerate(annot_columns):
            shuffle_results = data[data['shuf'] == col['shuf']]
            try:
                local_min_x = min(shuffle_results[variable].values)
            except ValueError:
                local_min_x = 0
            try:
                local_mean = np.mean(shuffle_results[variable].values)
                local_n = int(np.mean(shuffle_results['n'].values))
            except ValueError:
                local_mean = 0
                local_n = 0

            s = "mean {:,.0f} (top {:,.0f})".format(local_mean, local_n - local_mean)
            ax.text(local_min_x - 500, i, s, ha='right', va='center')

    return ax


def plot_all_train_vs_test(df, title="Title", fig_size=(8, 8), y_min=None, y_max=None):
    """ Plot everything from initial distributions, through training curves, to training outcomes.
        Then the results of using discovered genes in train and test sets both complete and masked.
        Then even report overlap internal to each cluster of differing gene lists.

        :param pandas.DataFrame df:
        :param str title: The title to put on the top of the whole plot
        :param fig_size: A tuple of inches across x inches high
        :param y_min: Hard code the bottom of the y-axis
        :param y_max: Hard code the top of the y-axis
    """

    lowest_possible_score, highest_possible_score = calc_hilo(
        y_min, y_max, df, ['best', 'train_score', 'test_score', 'masked_train_score', 'masked_test_score', ]
    )

    """ Plot the first pane, rising lines representing rising Mantel correlations as probes are dropped. """
    a = df.loc[df['shuf'] == 'none', 'path']
    b = df.loc[df['shuf'] == 'be04', 'path']
    c = df.loc[df['shuf'] == 'be08', 'path']
    d = df.loc[df['shuf'] == 'be16', 'path']
    e = df.loc[df['shuf'] == 'edge', 'path']
    f = df.loc[df['shuf'] == 'dist', 'path']
    g = df.loc[df['shuf'] == 'smsh', 'path']
    h = df.loc[df['shuf'] == 'agno', 'path']
    fig, ax_curve = plot.push_plot([
        {'files': list(h), 'linestyle': ':', 'color': 'green'},
        {'files': list(g), 'linestyle': ':', 'color': 'blue'},
        {'files': list(f), 'linestyle': ':', 'color': 'red'},
        {'files': list(e), 'linestyle': ':', 'color': 'orchid'},
        {'files': list(d), 'linestyle': ':', 'color': 'orchid'},
        {'files': list(c), 'linestyle': ':', 'color': 'orchid'},
        {'files': list(b), 'linestyle': ':', 'color': 'orchid'},
        {'files': list(a), 'linestyle': '-', 'color': 'black'}, ],
        # title="Split-half train vs test results",
        label_keys=['shuf'],
        fig_size=fig_size,
        title="",
        plot_overlaps=False,
    )
    # The top of the plot must be at least 0.25 higher than the highest value to make room for p-values.
    ax_curve.set_ylim(bottom=lowest_possible_score, top=highest_possible_score + 0.25)

    margin = 0.04
    row_height = 0.34
    peak_box_height = 0.12

    """ Top Row """

    box_width = 0.20
    x_left = margin
    fig.text(x_left, 1.0 - (2 * margin) + 0.01, "A) Training on altered training half", ha='left', va='bottom', fontsize=12)

    """ Horizontal peak plot """
    y_base = 1.0 - margin - peak_box_height - margin
    curve_x = x_left + box_width + margin
    curve_width = 1.0 - (4 * margin) - (2 * box_width)
    ax_peaks = box_and_swarm(
        fig, [curve_x, y_base, curve_width, peak_box_height],
        'Peaks', 'peak', df, orientation="h", lim=ax_curve.get_xlim()
    )
    ax_peaks.set_xticklabels([])

    """ Rising training curve plot """
    y_base = margin + row_height + margin + margin
    ax_curve.set_position([curve_x, y_base, curve_width, row_height])
    ax_curve.set_label('rise')
    ax_curve.set_xlabel('Training')

    """ Initial box and swarm plots """
    ax_pre = box_and_swarm(
        fig, [x_left, y_base, box_width, row_height],
        'Complete Mantel', 'initial', df, high_score=highest_possible_score, lim=ax_curve.get_ylim()
    )
    ax_pre.yaxis.tick_right()
    ax_pre.set_yticklabels([])
    ax_pre.set_ylabel('Mantel Correlation')
    ax_post = box_and_swarm(
        fig, [1.0 - box_width - margin, y_base, box_width, row_height],
        'Peak Mantel', 'best', df, high_score=highest_possible_score, lim=ax_curve.get_ylim()
    )

    """ Bottom Row """

    box_width = 0.20
    x_left = margin
    y_base = margin  # + 0.35 height makes the top at 0.40
    fig.text(x_left, margin + row_height + 0.01, "B) Testing in unshuffled halves", ha='left', va='bottom', fontsize=12)
    """ Train box and swarm plots """
    ax_train_complete = box_and_swarm(
        fig, [x_left + (0 * (margin + box_width)), y_base, box_width, row_height],
        'Train unmasked', 'train_score', df, high_score=highest_possible_score, lim=ax_curve.get_ylim()
    )
    ax_train_complete.yaxis.tick_right()
    ax_train_complete.set_yticklabels([])
    ax_train_complete.set_ylabel('Mantel Correlation')
    ax_train_masked = box_and_swarm(
        fig, [x_left + (1 * (margin + box_width)), y_base, box_width, row_height],
        'Train masked', 'masked_train_score', df, high_score=highest_possible_score, lim=ax_curve.get_ylim()
    )

    """ Test box and swarm plots """
    ax_test_complete = box_and_swarm(
        fig, [x_left + (2 * (margin + box_width)), y_base, box_width, row_height],
        'Test unmasked', 'test_score', df, high_score=highest_possible_score, lim=ax_curve.get_ylim()
    )
    ax_test_complete.yaxis.tick_right()
    ax_test_complete.set_yticklabels([])
    ax_test_complete.set_ylabel('Mantel Correlation')
    ax_test_masked = box_and_swarm(
        fig, [x_left + (3 * (margin + box_width)), y_base, box_width, row_height],
        'Test masked', 'masked_test_score', df, high_score=highest_possible_score, lim=ax_curve.get_ylim()
    )


    fig.text(0.50, 0.99, title, ha='center', va='top', fontsize=14)

    return fig, (ax_peaks, ax_pre, ax_curve, ax_post,
                 ax_train_complete, ax_train_masked, ax_test_complete, ax_test_masked,
                 )


def plot_fig_2(df, title="Title", fig_size=(8, 8), y_min=None, y_max=None):
    """ Plot everything from initial distributions, through training curves, to training outcomes.
        Then the results of using discovered genes in train and test sets both complete and masked.
        Then even report overlap internal to each cluster of differing gene lists.

        :param pandas.DataFrame df:
        :param str title: The title to put on the top of the whole plot
        :param fig_size: A tuple of inches across x inches high
        :param y_min: Hard code the bottom of the y-axis
        :param y_max: Hard code the top of the y-axis
    """

    lowest_possible_score, highest_possible_score = calc_hilo(
        y_min, y_max, df, ['best', 'train_score', 'test_score', 'masked_train_score', 'masked_test_score', ]
    )

    """ Plot the first pane, rising lines representing rising Mantel correlations as probes are dropped. """
    a = df.loc[df['shuf'] == 'none', 'path']
    b = df.loc[df['shuf'] == 'be04', 'path']
    c = df.loc[df['shuf'] == 'agno', 'path']
    fig, ax_curve = plot.push_plot([
        {'files': list(c), 'linestyle': ':', 'color': 'blue'},
        {'files': list(b), 'linestyle': ':', 'color': 'red'},
        {'files': list(a), 'linestyle': '-', 'color': 'black'}, ],
        # title="Split-half train vs test results",
        label_keys=['shuf', ],
        fig_size=fig_size,
        title="",
        plot_overlaps=False,
    )
    # The top of the plot must be at least 0.25 higher than the highest value to make room for p-values.
    ax_curve.set_ylim(bottom=lowest_possible_score, top=highest_possible_score + 0.25)

    margin = 0.05
    main_ratio = 0.60
    alt_ratio = 0.25

    """ Top Row """

    """ Rising training curve plot """
    ax_curve.set_position([margin + 0.01, margin, main_ratio, main_ratio])
    ax_curve.set_label('rise')
    ax_curve.set_xlabel('Training')
    ax_curve.set_ylabel('Mantel r')

    """ Horizontal peak plot """
    ax_peaks = box_and_swarm(
        fig, [margin + 0.01, margin + main_ratio + margin, main_ratio, alt_ratio],
        'Peaks', 'peak', df, orientation="h", lim=ax_curve.get_xlim()
    )
    ax_peaks.set_xticklabels([])

    """ Initial box and swarm plots """
    ax_post = box_and_swarm(
        fig, [margin + main_ratio + margin, margin, alt_ratio, main_ratio],
        'Peak Mantel', 'best', df, high_score=highest_possible_score, lim=ax_curve.get_ylim()
    )

    fig.text(margin + (2.0 * main_ratio / 5.0), margin + main_ratio - 0.01, "A", ha='left', va='top', fontsize=14)
    fig.text(margin + 0.02, 1.0 - margin - 0.01, "B", ha='left', va='top', fontsize=14)
    fig.text(margin + main_ratio + margin + 0.01, margin + main_ratio - 0.01, "C", ha='left', va='top', fontsize=14)

    return fig, (ax_curve, ax_peaks, ax_post)


def plot_fig_3(df, title="Title", fig_size=(8, 8), y_min=None, y_max=None):
    """ Plot Mantel correlation achieved in training by real and shuffled data.
        Then plot same values when trained features are applied to independent test data.

        :param pandas.DataFrame df:
        :param fig_size: A tuple of inches across x inches high
        :param y_min: Hard code the bottom of the y-axis
        :param y_max: Hard code the top of the y-axis
    """

    lowest_possible_score, highest_possible_score = calc_hilo(
        y_min, y_max, df, ['best', 'train_score', 'test_score', 'masked_train_score', 'masked_test_score', ]
    )

    fig = plt.figure(figsize=fig_size)

    margin = 0.05
    ax_height = 0.85
    ax_width = 0.42

    """ Train box and swarm plots """
    ax_a = box_and_swarm(
        fig, [margin, margin * 2, ax_width, ax_height],
        'Train unmasked', 'train_score', df, high_score=highest_possible_score,
    )
    # The top of the plot must be at least 0.25 higher than the highest value to make room for p-values.
    ax_a.set_ylim(bottom=lowest_possible_score, top=highest_possible_score + 0.25)
    ax_a.yaxis.tick_right()
    ax_a.set_yticklabels([])
    ax_a.set_ylabel('Mantel Correlation')

    """ Test box and swarm plots """
    ax_b = box_and_swarm(
        fig, [1.0 - margin - ax_width, margin * 2, ax_width, ax_height],
        'Test unmasked', 'test_score', df, high_score=highest_possible_score, lim=ax_a.get_ylim()
    )
    ax_b.yaxis.tick_left()
    # ax_b.set_ylabel('Mantel Correlation')

    fig.text(margin + 0.01, 1.0 - margin - 0.01, "A", ha='left', va='top', fontsize=14)
    fig.text(1.0 - margin - ax_width + 0.01, 1.0 - margin - 0.01, "B", ha='left', va='top', fontsize=14)

    return fig, (ax_a, ax_b)


def plot_fig_3_over_masks(df, title="Title", fig_size=(8, 8), y_min=None, y_max=None):
    """ Plot Mantel correlation achieved in training by real and shuffled data, over all distance masks.
        Then plot same values when trained features are applied to independent test data, over all distance masks.

        :param pandas.DataFrame df:
        :param fig_size: A tuple of inches across x inches high
        :param y_min: Hard code the bottom of the y-axis
        :param y_max: Hard code the top of the y-axis
    """

    lowest_possible_score, highest_possible_score = calc_hilo(
        y_min, y_max, df, ['best', 'train_score', 'test_score', 'masked_train_score', 'masked_test_score', ]
    )

    fig = plt.figure(figsize=fig_size)

    margin = 0.05
    ax_height = 0.85
    ax_width = 0.42

    """ Train box and swarm plots """
    ax_a = box_and_swarm(
        fig, [margin, margin * 2, ax_width, ax_height],
        'Train unmasked', 'train_score', df, high_score=highest_possible_score,
    )
    # The top of the plot must be at least 0.25 higher than the highest value to make room for p-values.
    ax_a.set_ylim(bottom=lowest_possible_score, top=highest_possible_score + 0.25)
    ax_a.yaxis.tick_right()
    ax_a.set_yticklabels([])
    ax_a.set_ylabel('Mantel Correlation')

    """ Test box and swarm plots """
    ax_b = box_and_swarm(
        fig, [1.0 - margin - ax_width, margin * 2, ax_width, ax_height],
        'Test unmasked', 'test_score', df, high_score=highest_possible_score, lim=ax_a.get_ylim()
    )
    ax_b.yaxis.tick_left()
    # ax_b.set_ylabel('Mantel Correlation')

    fig.text(margin + 0.01, 1.0 - margin - 0.01, "A", ha='left', va='top', fontsize=14)
    fig.text(1.0 - margin - ax_width + 0.01, 1.0 - margin - 0.01, "B", ha='left', va='top', fontsize=14)

    return fig, (ax_a, ax_b)


def describe_mantel(df, descriptor="", title="Title"):
    """ Generate textual descriptions to go along with the plot generated. """

    df['top_n'] = df['n'] - df['peak']

    d = [
        "<h2><span class=\"heavy\">{}</span></h2>".format(title),
        "<h3><span class=\"heavy\">Training peak heights and locations:</span></h3>",
        "<p><img src=\"./{}_fig_2.png\" alt=\"Figure 2. Mantel optimization\" width=\"768\"></p>".format(descriptor),
        "<p><strong>Figure 2. Mantel optimization.</strong> " + \
             "A) The Mantel correlation rises as each 'worst' gene is dropped. " + \
             "B) Few genes remain at peak correlation. " + \
             "C) Higher correlations are attained in real gene expression data than in any shuffled version of it." + \
             "</p>",
    ]
    for shuf in ['none', 'be04', 'be08', 'be16', 'edge', 'dist', 'agno', ]:
        highlighter = ("<mark>", "</mark>") if shuf == "none" else ("", "")
        masked_df = df[df['shuf'] == shuf]
        if len(masked_df) > 0:
            d.append("<p>Mantel peaks with {}-shuffled training sets peaked with {} probes remaining.".format(
                shuf, mean_and_sd(masked_df['top_n'])
            ))
            d.append("{}-shuffled Mantel correlations rose from {} to a peak of {}{}{}.</p>".format(
                shuf, mean_and_sd(masked_df['initial']), highlighter[0], mean_and_sd(masked_df['best']), highlighter[1]
            ))
        else:
            d.append("<p>No {}-shuffles available.".format(shuf))
    d.append("<h3><span class=\"heavy\">Using probes discovered in training to filter original split-half data, and re-Mantel:</span></h3>")
    d.append("<p><img src=\"./{}_fig_3.png\" alt=\"Figure 3. Mantel Optimization\" width=\"384\"></p>".format(descriptor))
    d.append("<p><strong>Figure 3. Gene performance.</strong> " + \
             "A) In original, complete training data, genes discovered in real data still perform better than " + \
             "genes discovered in shuffled data. Expression data shuffled to maintain distance relationships " + \
             "generated genes that performed better than those from agnostic permutations. " + \
             "B) Genes discovered in training data were also used to generate a Mantel correlation in test " + \
             "data, the left-out samples from splitting halves. Patterns are similar to training data, but " + \
             "genes discovered in real training data fall slightly in independent test data." + \
             "</p>")
    for shuf in ['none', 'be04', 'be08', 'be16', 'edge', 'dist', 'agno', ]:
        highlighter = ("<mark>", "</mark>") if shuf == "none" else ("", "")
        masked_df = df[df['shuf'] == shuf]
        if len(masked_df) > 0:
            d.append("<p>Real Mantel correlations with probes discovered in {}-shuffled training sets.".format(shuf))
            d.append("In unmasked train half: {}.".format( mean_and_sd(masked_df['train_score'])))
            d.append("In masked train half: {}.".format(mean_and_sd(masked_df['masked_train_score'])))
            d.append("In unmasked test half: {}{}{}.".format(
                highlighter[0], mean_and_sd(masked_df['test_score']), highlighter[1]
            ))
            d.append("In masked test half: {}.</p>".format(mean_and_sd(masked_df['masked_test_score'])))
        else:
            d.append("No {}-shuffles available.".format(shuf))
    return "\n".join(d)


def plot_overlap(df, title="Title", fig_size=(8, 8), y_min=None, y_max=None):
    """ Plot everything from initial distributions, through training curves, to training outcomes.
        Then the results of using discovered genes in train and test sets both complete and masked.
        Then even report overlap internal to each cluster of differing gene lists.

        :param pandas.DataFrame df:
        :param str title: The title to put on the top of the whole plot
        :param fig_size: A tuple of inches across x inches high
        :param y_min: Hard code the bottom of the y-axis
        :param y_max: Hard code the top of the y-axis
    """

    lowest_possible_score, highest_possible_score = calc_hilo(
        y_min, y_max, df, ['best', 'train_score', 'test_score', 'masked_train_score', 'masked_test_score', ]
    )

    fig = plt.figure(figsize=fig_size)

    margin = 0.04
    box_height = 0.84
    # Four and a half axes get 1.0 - (6 * margin) = 0.76  &  0.76 / 4.5 = 0.17
    box_width = 0.20
    x_left = margin
    bottom = margin * 2

    """ Print titles and subtitles """
    fig.text(0.50, 0.99, title, ha='center', va='top', fontsize=14)

    """ Internal overlap plots """
    fig.text(x_left, 1.0 - (2 * margin) + 0.01, "Overlap between actual training data and shuffles", ha='left', va='bottom', fontsize=12)
    df.loc[df['shuf'] == 'none', 'real_v_shuffle_overlap'] = df.loc[df['shuf'] == 'none', 'overlap_by_seed']
    ax = box_and_swarm(
        fig, [x_left, bottom, box_width, box_height],
        'train vs shuffles', 'real_v_shuffle_overlap', df, orientation="v", ps=True
    )
    ax.set_ylim(bottom=lowest_possible_score, top=highest_possible_score)

    return fig, (ax, )


def plot_fig_4(df, title="Title", fig_size=(8, 5), y_min=None, y_max=None):
    """ Plot everything from initial distributions, through training curves, to training outcomes.
        Then the results of using discovered genes in train and test sets both complete and masked.
        Then even report overlap internal to each cluster of differing gene lists.

        :param pandas.DataFrame df:
        :param fig_size: A tuple of inches across x inches high
        :param y_min: Hard code the bottom of the y-axis
        :param y_max: Hard code the top of the y-axis
    """

    lowest_possible_score, highest_possible_score = calc_hilo(
        y_min, y_max, df, ['overlap_by_seed', 'ktau_by_seed', 'overlap_by_split', 'ktau_by_split', ]
    )
    fig = plt.figure(figsize=fig_size)

    margin = 0.050
    gap = 0.040
    ax_width = 0.190
    ax_height = 0.840

    """ Internal overlap plots """
    # For 'real_v_shuffle_overlap', all unshuffled values are 1.0 because each result matches itself.
    # It needs to be replaced with its internal intra-group overlap for a visual baseline,
    # even though it's not within-split and shouldn't be compared quantitatively against shuffles.
    df.loc[df['shuf'] == 'none', 'real_v_shuffle_overlap'] = df.loc[df['shuf'] == 'none', 'overlap_by_seed']
    df.loc[df['shuf'] == 'none', 'real_v_shuffle_ktau'] = df.loc[df['shuf'] == 'none', 'ktau_by_seed']
    # In only the unshuffled runs, fill in the zeroes (or NaNs) with intra-group data. Unshuffled runs have no seeds.
    # Shuffled runs already have correct calculated overlaps.
    df.loc[df['shuf'] == 'none', 'overlap_by_seed'] = df.loc[df['shuf'] == 'none', 'overlap_by_seed']
    df.loc[df['shuf'] == 'none', 'ktau_by_seed'] = df.loc[df['shuf'] == 'none', 'ktau_by_seed']

    ax_a = box_and_swarm(
        fig, [margin, margin * 2, ax_width, ax_height],
        'intra-shuffle-seed similarity', 'overlap_by_seed', df, orientation="v", ps=False
    )
    ax_a.set_ylim(bottom=lowest_possible_score, top=highest_possible_score)

    ax_b = box_and_swarm(
        fig, [margin + ax_width + gap, margin * 2, ax_width, ax_height],
        'train vs shuffles', 'real_v_shuffle_overlap', df[df['shuf'] != 'none'], orientation="v", ps=False
    )
    ax_b.set_ylim(ax_a.get_ylim())

    ax_c = box_and_swarm(
        fig, [1.0 - margin - ax_width - gap - ax_width, margin * 2, ax_width, ax_height],
        'intra-shuffle-seed similarity', 'ktau_by_seed', df, orientation="v", ps=False
    )
    ax_c.set_ylim(ax_a.get_ylim())

    ax_d = box_and_swarm(
        fig, [1.0 - margin - ax_width, margin * 2, ax_width, ax_height],
        'train vs shuffles', 'real_v_shuffle_ktau', df[df['shuf'] != 'none'], orientation="v", ps=False
    )
    ax_d.set_ylim(ax_a.get_ylim())

    ax_a.yaxis.tick_right()
    ax_a.set_yticklabels([])
    ax_a.set_ylabel('Overlap % (past peak)')
    ax_b.yaxis.tick_left()

    ax_c.yaxis.tick_right()
    ax_c.set_yticklabels([])
    ax_c.set_ylabel('Kendall tau')
    ax_d.yaxis.tick_left()

    fig.text(margin + ax_width + (gap / 2.0), 1.0 - 0.01,
             "Overlap of top genes", ha='center', va='top', fontsize=14
    )
    fig.text(margin + 0.01, 1.0 - margin - 0.02, "A", ha='left', va='top', fontsize=14)
    fig.text(margin + ax_width + gap + 0.01, 1.0 - margin - 0.02, "B", ha='left', va='top', fontsize=14)

    fig.text(1.0 - margin - ax_width - (gap / 2.0), 1.0 - 0.01,
             "Kendall tau of entire list", ha='center', va='top', fontsize=14
    )
    fig.text(1.0 - margin - ax_width - gap - ax_width + 0.01, 1.0 - margin - 0.02, "C", ha='left', va='top', fontsize=14)
    fig.text(1.0 - margin - ax_width + 0.01, 1.0 - margin - 0.02, "D", ha='left', va='top', fontsize=14)

    return fig, (ax_a, ax_b, ax_c, ax_d)


def describe_overlap(df, descriptor="", title="Title"):
    """ Generate textual descriptions to go along with the plot generated. """

    d = [
        "<h2><span class=\"heavy\">{}</span></h2>".format(title),
        "<p><img src=\"./{}_fig_4.png\" alt=\"Figure 4. Overlapping genes\" width=\"768\"></p>".format(descriptor),
        "<p><strong>Figure 4. Overlapping genes.</strong> " + \
            "A) Intra-type consistency. The percent overlap between each past-peak gene list and its comparable, same-split-half, gene lists. " + \
            "Non-shuffled results have only one gene list per split-half and are necessarily plotted across all split-halves. " + \
            "B) Inter-type consistency, or how well a shuffle's top genes approximate those from un-shuffled data. " + \
            ". The percent overlap between the gene list surviving past the peak in each " + \
            "shuffle, and the gene list discovered in the raw data it was shuffled from. " + \
            "C) Intra-type consistency. The kendall tau between each complete ranked gene list and its comparable, same-split-half, gene lists." + \
            "Non-shuffled results have only one gene list per split-half and are necessarily plotted across all split-halves. " + \
            "D) Inter-type consistency, or how well a shuffle approximates un-shuffled data. " + \
            "The kendall tau correlation between the ranked gene list in each " + \
            "shuffle, and the ranked gene list discovered in the raw data it was shuffled from. " + \
            "</p>",
        "<h3><span class=\"heavy\">Internal altogether (not plotted):</span></h3><p>",
    ]
    for shuf in ['none', 'be04', 'be08', 'be16', 'edge', 'dist', 'agno', ]:
        if len(df[df['shuf'] == shuf]) > 0:
            d.append("Overlap within {}-shuffled: {}.<br />".format(
                shuf, mean_and_sd(df[df['shuf'] == shuf]['train_overlap'])
            ))
            d.append("Kendall tau (trimmed) within {}-shuffled: {}.<br />".format(
                shuf, mean_and_sd(df[df['shuf'] == shuf]['train_ktau'])
            ))
    d.append("</p>")
    d.append("<h3><span class=\"heavy\">Internal within a shuffle seed, across splits:</span></h3><p>")
    for shuf in ['none', 'be04', 'be08', 'be16', 'edge', 'dist', 'agno', ]:
        if len(df[df['shuf'] == shuf]) > 0:
            d.append("Overlap within {}-shuffled, within shuffle seed: {}.<br />".format(
                shuf, mean_and_sd(df[df['shuf'] == shuf]['overlap_by_seed'])
            ))
            d.append("Kendall tau (trimmed) within {}-shuffled, within shuffle seed: {}.<br />".format(
                shuf, mean_and_sd(df[df['shuf'] == shuf]['ktau_by_seed'])
            ))
    d.append("</p>")
    d.append("<h3><span class=\"heavy\">Internal within a split, across shuffle seeds (feeds figure 4 B):</span></h3><p>")
    for shuf in ['none', 'be04', 'be08', 'be16', 'edge', 'dist', 'agno', ]:
        if len(df[df['shuf'] == shuf]) > 0:
            d.append("Overlap within {}-shuffled, within split batch: {}.<br />".format(
                shuf, mean_and_sd(df[df['shuf'] == shuf]['overlap_by_split'])
            ))
            d.append("Kendall tau (trimmed) within {}-shuffled, within split batch: {}.<br />".format(
                shuf, mean_and_sd(df[df['shuf'] == shuf]['ktau_by_split'])
            ))
    d.append("</p>")
    d.append("<h3><span class=\"heavy\">Real vs shuffled similarity:</span></h3><p>")
    for shuf in ['none', 'be04', 'be08', 'be16', 'edge', 'dist', 'agno', ]:
        if len(df[df['shuf'] == shuf]) > 0:
            d.append("Overlap between un-shuffled and {}-shuffled: {}.<br />".format(
                shuf, mean_and_sd(df[df['shuf'] == shuf]['real_v_shuffle_overlap'])
            ))
            d.append("Kendall tau (trimmed) between un-shuffled and {}-shuffled: {}.<br />".format(
                shuf, mean_and_sd(df[df['shuf'] == shuf]['real_v_shuffle_ktau'])
            ))
    d.append("</p>")
    d.append("<h3><span class=\"heavy\">Train vs test:</span></h3>")
    d.append("<p>Overlap between top train-discovered probes and what would have been discovered in the other half: ")
    d.append("    {}.</p>".format(mean_and_sd(df[df['shuf'] == 'none']['train_vs_test_overlap'])))
    return "\n".join(d)


def plot_performance_over_thresholds(relevant_results):
    """ Generate a figure with three axes for Mantels, T scores, and Overlaps by threshold. """

    plot_data = relevant_results[relevant_results['threshold'] != 'peak']
    plot_data['threshold'] = plot_data['threshold'].apply(int)

    peak_data = relevant_results[relevant_results['threshold'] == 'peak']
    peak_data['threshold'] = peak_data['n'] - peak_data['peak']

    fig, ax_mantel_scores = plt.subplots(figsize=(10, 12))
    margin = 0.04
    ht = 0.28

    """ Top panel is Mantel correlations. """
    ax_mantel_scores.set_position([margin, 1.0 - margin - ht, 1.0 - (2 * margin), ht])
    sns.lineplot(x="threshold", y="best", data=plot_data, color="gray", ax=ax_mantel_scores, label="peak")
    sns.lineplot(x="threshold", y="train_score", data=plot_data, color="green", ax=ax_mantel_scores, label="train")
    sns.scatterplot(x="threshold", y="train_score", data=peak_data, color="green", ax=ax_mantel_scores)
    sns.lineplot(x="threshold", y="test_score", data=plot_data, color="red", ax=ax_mantel_scores, label="test")
    sns.scatterplot(x="threshold", y="test_score", data=peak_data, color="red", ax=ax_mantel_scores)

    rect = patches.Rectangle((158, -0.3), 5.0, 1.0, facecolor='gray', fill=True, alpha=0.25)
    ax_mantel_scores.add_patch(rect)

    ax_mantel_scores.legend(labels=['peak', 'train', 'test'])
    plt.suptitle("Scores by top probe threshold")
    ax_mantel_scores.set_ylabel('Mantel correlation')

    """ Middle panel is Overlap calculations. """
    ax_overlaps = fig.add_axes([margin, (2 * margin) + ht, 1.0 - (2 * margin), ht], "Real vs Shuffle Overlap Percentages")
    sns.lineplot(x="threshold", y="train_vs_test_overlap", data=plot_data, color="gray", ax=ax_overlaps, label="t-t overlap")
    sns.scatterplot(x="threshold", y="train_vs_test_overlap", data=peak_data, color="black", ax=ax_overlaps)
    sns.lineplot(x="threshold", y="overlap_real_vs_agno", data=plot_data, color="green", ax=ax_overlaps, label="agno")
    sns.scatterplot(x="threshold", y="overlap_real_vs_agno", data=peak_data, color="green", ax=ax_overlaps)
    sns.lineplot(x="threshold", y="overlap_real_vs_dist", data=plot_data, color="red", ax=ax_overlaps, label="dist")
    sns.scatterplot(x="threshold", y="overlap_real_vs_dist", data=peak_data, color="red", ax=ax_overlaps)
    sns.lineplot(x="threshold", y="overlap_real_vs_edge", data=plot_data, color="orchid", ax=ax_overlaps, label="edge")
    sns.scatterplot(x="threshold", y="overlap_real_vs_edge", data=peak_data, color="orchid", ax=ax_overlaps)
    sns.lineplot(x="threshold", y="overlap_real_vs_be04", data=plot_data, color="orchid", ax=ax_overlaps, label="be04")
    sns.scatterplot(x="threshold", y="overlap_real_vs_be04", data=peak_data, color="orchid", ax=ax_overlaps)
    sns.lineplot(x="threshold", y="overlap_real_vs_be08", data=plot_data, color="orchid", ax=ax_overlaps, label="be08")
    sns.scatterplot(x="threshold", y="overlap_real_vs_be08", data=peak_data, color="orchid", ax=ax_overlaps)
    sns.lineplot(x="threshold", y="overlap_real_vs_be16", data=plot_data, color="orchid", ax=ax_overlaps, label="be16")
    sns.scatterplot(x="threshold", y="overlap_real_vs_be16", data=peak_data, color="orchid", ax=ax_overlaps)
    v_rect = patches.Rectangle((158, 0.0), 5.0, 1.0, facecolor='gray', fill=True, alpha=0.25)
    ax_overlaps.add_patch(v_rect)

    """ Bottom panel is t-scores. """
    ax_mantel_ts = fig.add_axes([margin, margin, 1.0 - (2 * margin), ht], "Mantel T Scores")
    sns.lineplot(x="threshold", y="t_mantel_agno", data=plot_data, color="green", ax=ax_mantel_ts, label="agno")
    sns.lineplot(x="threshold", y="t_mantel_dist", data=plot_data, color="red", ax=ax_mantel_ts, label="dist")
    sns.lineplot(x="threshold", y="t_mantel_edge", data=plot_data, color="orchid", ax=ax_mantel_ts, label="edge")
    sns.lineplot(x="threshold", y="t_mantel_be04", data=plot_data, color="orchid", ax=ax_mantel_ts, label="be04")
    sns.lineplot(x="threshold", y="t_mantel_be08", data=plot_data, color="orchid", ax=ax_mantel_ts, label="be08")
    sns.lineplot(x="threshold", y="t_mantel_be16", data=plot_data, color="orchid", ax=ax_mantel_ts, label="be16")

    v_rect = patches.Rectangle((158, -100), 5.0, 200.0, facecolor='gray', fill=True, alpha=0.25)
    ax_mantel_ts.add_patch(v_rect)
    h_rect = patches.Rectangle((0, -2), 1024.0, 2.0, facecolor='gray', fill=True, alpha=0.25)
    ax_mantel_ts.add_patch(h_rect)

    ax_mantel_ts.legend(labels=['agno', 'dist', 'edge', 'be04', 'be08', 'be16', ])
    ax_mantel_ts.set_ylabel('T score')

    return fig, (ax_mantel_scores, ax_mantel_scores, ax_mantel_ts)


