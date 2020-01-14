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


def box_and_swarm(figure, placement, label, variable, data, high_score=1.0, orientation="v", lim=None, ps=True, cols=4):
    """ Create an axes object with a swarm plot draw over a box plot of the same data. """

    shuffle_order = ['none', 'be04', 'be08', 'be16', 'edge', 'dist', 'agno']
    shuffle_color_boxes = sns.color_palette(['gray', 'orchid', 'orchid', 'orchid', 'orchid', 'red', 'green'])
    shuffle_color_points = sns.color_palette(['black', 'orchid', 'orchid', 'orchid', 'orchid', 'red', 'green'])
    if cols == 3:
        shuffle_order = shuffle_order[1:]
        shuffle_color_boxes = shuffle_color_boxes[1:]
        shuffle_color_points = shuffle_color_points[1:]
    elif cols == 1:
        shuffle_order = shuffle_order[0:1]
        shuffle_color_boxes = shuffle_color_boxes[0:1]
        shuffle_color_points = shuffle_color_points[0:1]

    annot_columns = [
        {'shuffle': 'none', 'xo': 0.0, 'xp': 0.0},
        {'shuffle': 'be04', 'xo': 1.0, 'xp': 0.5},
        {'shuffle': 'be08', 'xo': 2.0, 'xp': 1.0},
        {'shuffle': 'be16', 'xo': 3.0, 'xp': 1.5},
        {'shuffle': 'edge', 'xo': 4.0, 'xp': 2.0},
        {'shuffle': 'dist', 'xo': 5.0, 'xp': 2.5},
        {'shuffle': 'agno', 'xo': 6.0, 'xp': 3.0},
    ]

    ax = figure.add_axes(placement, label=label)
    if orientation == "v":
        sns.boxplot(data=data, x='shuffle', y=variable, order=shuffle_order, palette=shuffle_color_boxes, ax=ax)
        sns.swarmplot(data=data, x='shuffle', y=variable, order=shuffle_order, palette=shuffle_color_points, ax=ax)
        ax.set_ylabel(None)
        ax.set_xlabel(label)
        if lim is not None:
            ax.set_ylim(lim)
    else:
        sns.boxplot(data=data, x=variable, y='shuffle', order=shuffle_order, palette=shuffle_color_boxes, ax=ax)
        sns.swarmplot(data=data, x=variable, y='shuffle', order=shuffle_order, palette=shuffle_color_points, ax=ax)
        ax.set_xlabel(None)
        ax.set_ylabel(label)
        if lim is not None:
            ax.set_xlim(lim)

    """ Calculate p-values for each column in the above plots, and annotate accordingly. """
    if ps & (orientation == "v"):
        gap = 0.06
        actual_results = data[data['shuffle'] == 'none'][variable].values
        try:
            global_max_y = max(data[variable].values)
        except ValueError:
            global_max_y = high_score
        for i, col in enumerate(annot_columns):
            shuffle_results = data[data['shuffle'] == col['shuffle']]
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
            shuffle_results = data[data['shuffle'] == col['shuffle']]
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
            max(df['best']),
            max(df['train_score']), max(df['test_score']),
            max(df['masked_train_score']), max(df['masked_test_score']),
        )
    else:
        highest_possible_score = ymax
    if (ymin is None) and (len(df.index) > 0):
        lowest_possible_score = min(
            min(df['best']),
            min(df['train_score']), min(df['test_score']),
            min(df['masked_train_score']), min(df['masked_test_score']),
        )
    else:
        lowest_possible_score = ymin

    """ Plot the first pane, rising lines representing rising Mantel correlations as probes are dropped. """
    a = df.loc[df['shuffle'] == 'none', 'path']
    b = df.loc[df['shuffle'] == 'be04', 'path']
    c = df.loc[df['shuffle'] == 'be08', 'path']
    d = df.loc[df['shuffle'] == 'be16', 'path']
    e = df.loc[df['shuffle'] == 'edge', 'path']
    f = df.loc[df['shuffle'] == 'dist', 'path']
    g = df.loc[df['shuffle'] == 'agno', 'path']
    fig, ax_curve = plot.push_plot([
        {'files': list(g), 'linestyle': ':', 'color': 'green'},
        {'files': list(f), 'linestyle': ':', 'color': 'red'},
        {'files': list(e), 'linestyle': ':', 'color': 'orchid'},
        {'files': list(d), 'linestyle': ':', 'color': 'orchid'},
        {'files': list(c), 'linestyle': ':', 'color': 'orchid'},
        {'files': list(b), 'linestyle': ':', 'color': 'orchid'},
        {'files': list(a), 'linestyle': '-', 'color': 'black'}, ],
        # title="Split-half train vs test results",
        label_keys=['shuffle'],
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


def plot_fig_2(df, title="Title", fig_size=(8, 8), ymin=None, ymax=None):
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
            max(df['best']),
            max(df['train_score']), max(df['test_score']),
            max(df['masked_train_score']), max(df['masked_test_score']),
        )
    else:
        highest_possible_score = ymax
    if (ymin is None) and (len(df.index) > 0):
        lowest_possible_score = min(
            min(df['best']),
            min(df['train_score']), min(df['test_score']),
            min(df['masked_train_score']), min(df['masked_test_score']),
        )
    else:
        lowest_possible_score = ymin

    """ Plot the first pane, rising lines representing rising Mantel correlations as probes are dropped. """
    a = df.loc[df['shuffle'] == 'none', 'path']
    b = df.loc[df['shuffle'] == 'be04', 'path']
    c = df.loc[df['shuffle'] == 'be08', 'path']
    d = df.loc[df['shuffle'] == 'be16', 'path']
    e = df.loc[df['shuffle'] == 'edge', 'path']
    f = df.loc[df['shuffle'] == 'dist', 'path']
    g = df.loc[df['shuffle'] == 'agno', 'path']
    fig, ax_curve = plot.push_plot([
        {'files': list(g), 'linestyle': ':', 'color': 'green'},
        {'files': list(f), 'linestyle': ':', 'color': 'red'},
        {'files': list(e), 'linestyle': ':', 'color': 'orchid'},
        {'files': list(d), 'linestyle': ':', 'color': 'orchid'},
        {'files': list(c), 'linestyle': ':', 'color': 'orchid'},
        {'files': list(b), 'linestyle': ':', 'color': 'orchid'},
        {'files': list(a), 'linestyle': '-', 'color': 'black'}, ],
        # title="Split-half train vs test results",
        label_keys=['shuffle'],
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
    ax_curve.set_position([margin, margin, main_ratio, main_ratio])
    ax_curve.set_label('rise')
    ax_curve.set_xlabel('Training')

    """ Horizontal peak plot """
    ax_peaks = box_and_swarm(
        fig, [margin, margin + main_ratio + margin, main_ratio, alt_ratio],
        'Peaks', 'peak', df, orientation="h", lim=ax_curve.get_xlim()
    )
    ax_peaks.set_xticklabels([])

    """ Initial box and swarm plots """
    ax_post = box_and_swarm(
        fig, [margin + main_ratio + margin, margin, alt_ratio, main_ratio],
        'Peak Mantel', 'best', df, high_score=highest_possible_score, lim=ax_curve.get_ylim()
    )

    fig.text(margin + 0.01, margin + (2 * main_ratio / 3), "A", ha='left', va='top', fontsize=14)
    fig.text(margin + 0.01, 1.0 - margin - 0.01, "B", ha='left', va='top', fontsize=14)
    fig.text(margin + main_ratio + margin + 0.01, margin + main_ratio - 0.01, "C", ha='left', va='top', fontsize=14)

    return fig, (ax_curve, ax_peaks, ax_post)


def plot_fig_3(df, title="Title", fig_size=(8, 8), ymin=None, ymax=None):
    """ Plot everything from initial distributions, through training curves, to training outcomes.
        Then the results of using discovered genes in train and test sets both complete and masked.
        Then even report overlap internal to each cluster of differing gene lists.

        :param pandas.DataFrame df:
        :param fig_size: A tuple of inches across x inches high
        :param ymin: Hard code the bottom of the y-axis
        :param ymax: Hard code the top of the y-axis
    """

    # Calculate (or blindly accept) the range of the y-axis, which must be the same for all four axes.
    if (ymax is None) and (len(df.index) > 0):
        highest_possible_score = max(
            max(df['best']),
            max(df['train_score']), max(df['test_score']),
            max(df['masked_train_score']), max(df['masked_test_score']),
        )
    else:
        highest_possible_score = ymax
    if (ymin is None) and (len(df.index) > 0):
        lowest_possible_score = min(
            min(df['best']),
            min(df['train_score']), min(df['test_score']),
            min(df['masked_train_score']), min(df['masked_test_score']),
        )
    else:
        lowest_possible_score = ymin

    fig = plt.figure(figsize=fig_size)

    margin = 0.05
    ax_height = 0.90
    ax_width = 0.42

    """ Train box and swarm plots """
    ax_a = box_and_swarm(
        fig, [margin, margin, ax_width, ax_height],
        'Train unmasked', 'train_score', df, high_score=highest_possible_score,
    )
    # The top of the plot must be at least 0.25 higher than the highest value to make room for p-values.
    ax_a.set_ylim(bottom=lowest_possible_score, top=highest_possible_score + 0.25)
    ax_a.yaxis.tick_right()
    ax_a.set_yticklabels([])
    ax_a.set_ylabel('Mantel Correlation')

    """ Test box and swarm plots """
    ax_b = box_and_swarm(
        fig, [1.0 - margin - ax_width, margin, ax_width, ax_height],
        'Test unmasked', 'test_score', df, high_score=highest_possible_score, lim=ax_a.get_ylim()
    )
    ax_b.yaxis.tick_left()
    # ax_b.set_ylabel('Mantel Correlation')

    fig.text(margin + 0.01, 1.0 - margin - 0.01, "A", ha='left', va='top', fontsize=14)
    fig.text(1.0 - margin - ax_width + 0.01, 1.0 - margin - 0.01, "B", ha='left', va='top', fontsize=14)

    return fig, (ax_a, ax_b)


def plot_fig_4(df, title="Title", fig_size=(8, 8), y_min=None, y_max=None):
    """ Plot everything from initial distributions, through training curves, to training outcomes.
        Then the results of using discovered genes in train and test sets both complete and masked.
        Then even report overlap internal to each cluster of differing gene lists.

        :param pandas.DataFrame df:
        :param fig_size: A tuple of inches across x inches high
        :param y_min: Hard code the bottom of the y-axis
        :param y_max: Hard code the top of the y-axis
    """

    fig = plt.figure(figsize=fig_size)

    margin = 0.05
    ax_width = 0.42
    ax_height = 0.90

    """ Internal overlap plots """
    df.loc[df['shuffle'] == 'none', 'real_v_shuffle_overlap'] = df.loc[df['shuffle'] == 'none', 'overlap_by_seed']
    ax_a = box_and_swarm(
        fig, [margin, margin, ax_width, ax_height],
        'train vs shuffles', 'real_v_shuffle_overlap', df, orientation="v", ps=True
    )

    ax_b = box_and_swarm(
        fig, [1.0 - margin - ax_width, margin, ax_width, ax_height],
        'intra-list similarity', 'train_overlap', df, orientation="v", ps=True, lim=ax_a.get_ylim()
    )

    if (y_min is not None) and (y_max is not None):
        ax_a.set_ylim(bottom=y_min, top=y_max)
        ax_b.set_ylim(bottom=y_min, top=y_max)
    else:
        ax_a.set_ylim(
            bottom=min(ax_a.get_ylim()[0], ax_b.get_ylim()[0], ax_a.get_ylim()[1], ax_b.get_ylim()[1]),
            top=max(ax_a.get_ylim()[0], ax_b.get_ylim()[0], ax_a.get_ylim()[1], ax_b.get_ylim()[1])
        )
        ax_b.set_ylim(ax_a.get_ylim())

    ax_a.yaxis.tick_right()
    ax_a.set_yticklabels([])
    ax_a.set_ylabel('Overlap % (past peak)')
    ax_b.yaxis.tick_left()
    # ax_b.set_ylabel('Overlap % (past peak)')

    fig.text(margin + 0.01, 1.0 - margin - 0.01, "A", ha='left', va='top', fontsize=14)
    fig.text(margin + ax_width + margin + 0.01, 1.0 - margin - 0.01, "B", ha='left', va='top', fontsize=14)

    return fig, (ax_a, ax_b)


def describe_mantel(df, descriptor="", title="Title"):
    """ Generate textual descriptions to go along with the plot generated. """

    df['top_n'] = df['n'] - df['peak']

    d = ["<p><span class=\"heavy\">{}</span></p>".format(title),
         "<p><span class=\"heavy\">Training peak heights and locations:</span></p>"]
    for shuffle in ['none', 'be04', 'be08', 'be16', 'edge', 'dist', 'agno', ]:
        masked_df = df[df['shuffle'] == shuffle]
        d.append("<p>Mantel peaks with {}-shuffled training sets peaked with {} probes remaining.".format(
            shuffle, mean_and_sd(masked_df['top_n'])
        ))
        d.append("{}-shuffled Mantel correlations rose from {} to a peak of {}.".format(
            shuffle, mean_and_sd(masked_df['initial']), mean_and_sd(masked_df['best'])
        ))
        d.append("Mantel peak locations with {}-shuffled: {}.</p>".format(
            shuffle, mean_and_sd(masked_df['top_n'])
        ))
    d.append("<p><img src=\"./{}_fig_2.png\" alt=\"Figure 2. Mantel optimization\"></p>".format(descriptor))
    d.append("<p><strong>Figure 2. Mantel optimization.</strong> " + \
             "A) The Mantel correlation rises as each 'worst' gene is dropped. " + \
             "B) Few genes remain at peak correlation. " + \
             "C) Higher correlations are attained in real gene expression data than in any shuffled version of it.</p>")
    d.append("<p><span class=\"heavy\">Using probes discovered in training to filter original split-half data, and re-Mantel:</span></p>")
    for shuffle in ['none', 'be04', 'be08', 'be16', 'edge', 'dist', 'agno', ]:
        masked_df = df[df['shuffle'] == shuffle]
        d.append("<p>Real Mantel correlations with probes discovered in {}-shuffled training sets.".format(shuffle))
        d.append("In unmasked train half: {}.".format( mean_and_sd(masked_df['train_score'])))
        d.append("In masked train half: {}.".format(mean_and_sd(masked_df['masked_train_score'])))
        d.append("In unmasked test half: {}.".format(mean_and_sd(masked_df['test_score'])))
        d.append("In masked test half: {}.</p>".format(mean_and_sd(masked_df['masked_test_score'])))
    d.append("<p><img src=\"./{}_fig_3.png\" alt=\"Figure 3. Mantel Optimization\"></p>".format(descriptor))
    d.append("<p><strong>Figure 3. Gene performance.</strong> " + \
             "A) In original, complete training data, genes discovered in real data still perform better than " + \
             "genes discovered in shuffled data. Expression data shuffled to maintain distance relationships " + \
             "generated genes that performed better than those from agnostic permutations. "
             "B) Genes discovered in training data were also used to generate a Mantel correlation in test " + \
             "data, the left-out samples from splitting halves. Patterns are similar to training data, but " + \
             "genes discovered in real training data fall slightly in independent test data.</p>")
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
    df.loc[df['shuffle'] == 'none', 'real_v_shuffle_overlap'] = df.loc[df['shuffle'] == 'none', 'overlap_by_seed']
    ax = box_and_swarm(
        fig, [x_left, bottom, box_width, box_height],
        'train vs shuffles', 'real_v_shuffle_overlap', df, orientation="v", ps=True
    )
    if (y_min is not None) and (y_max is not None):
        ax.set_ylim(bottom=y_min, top=y_max)

    return fig, (ax, )


def describe_overlap(df, descriptor="", title="Title"):
    """ Generate textual descriptions to go along with the plot generated. """

    d = ["<p><span class=\"heavy\">{}</span></p>".format(title),
         "<p><span class=\"heavy\">Internal altogether (not plotted):</span></p><p>"]
    for shuffle in ['none', 'be04', 'be08', 'be16', 'edge', 'dist', 'agno', ]:
        d.append("Overlap within {}-shuffled: {}.".format(
            shuffle, mean_and_sd(df[df['shuffle'] == shuffle]['train_overlap'])
        ))
    d.append("</p>")
    d.append("<p><span class=\"heavy\">Internal by shuffle:</span></p><p>")
    for shuffle in ['none', 'be04', 'be08', 'be16', 'edge', 'dist', 'agno', ]:
        d.append("Overlap within {}-shuffled, by seed: {}.".format(
            shuffle, mean_and_sd(df[df['shuffle'] == shuffle]['overlap_by_seed'])
        ))
    d.append("</p>")
    d.append("<p><span class=\"heavy\">Internal by split:</span></p><p>")
    for shuffle in ['none', 'be04', 'be08', 'be16', 'edge', 'dist', 'agno', ]:
        d.append("Overlap within {}-shuffled, by split: {}.".format(
            shuffle, mean_and_sd(df[df['shuffle'] == shuffle]['overlap_by_split'])
        ))
    d.append("</p>")
    d.append("<p><span class=\"heavy\">Real vs shuffled similarity:</span></p><p>")
    for shuffle in ['none', 'be04', 'be08', 'be16', 'edge', 'dist', 'agno', ]:
        d.append("Overlap between un-shuffled and {}-shuffled: {}.".format(
            shuffle, mean_and_sd(df[df['shuffle'] == shuffle]['real_v_shuffle_overlap'])
        ))
    d.append("<p><img src=\"./{}_fig_4.png\" alt=\"Figure 4. Overlapping genes\"></p>".format(descriptor))
    d.append("<p><strong>Figure 4. Overlapping genes.</strong> " + \
             "A) Inter-type consistency. The percent overlap between the gene list surviving past the peak in each " + \
             "shuffle, and the gene list discovered in the raw data it was shuffled from. " + \
             "B) Intra-type consistency. The percent overlap between each run and its comparable runs.</p>")
    d.append("</p>")
    d.append("<p><span class=\"heavy\">Train vs test:</span></p>")
    d.append("<p>Overlap between top train-discovered probes and what would have been discovered in the other half: ")
    d.append("    {}.</p>".format(mean_and_sd(df[df['shuffle'] == 'none']['train_vs_test_overlap'])))
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


