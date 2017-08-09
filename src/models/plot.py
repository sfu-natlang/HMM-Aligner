# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib import rcParams, rc
rc('pdf', fonttype=42)
__version__ = "0.4a"


figures = []
x = y = 0


def plotAlignmentWithScore(score,
                           align,
                           f=None,
                           e=None,
                           debug=False,
                           dispValue=False,
                           output=None):
    global x, y
    global figures
    figures.append(plt.subplots())
    fig, ax = figures[-1]
    if isinstance(score, np.ndarray):
        x, y = score.shape[:2]
    else:
        x = len(score)
        y = len(score[0])
    # Produce coordinates
    xs = np.tile(np.arange(x), (y, 1)).T.reshape(x * y) + 1
    ys = np.tile(np.arange(y), (x, 1)).reshape(x * y) + 1

    # Process colours
    tmp = score + np.abs(np.min(score, axis=1)[:, None])
    tmp *= tmp
    tmp = tmp / np.max(tmp, axis=1)[:, None]
    colours = 1. - tmp

    # Set origin, move x axis labels
    ax.xaxis.tick_top()
    plt.gca().invert_yaxis()

    # Add words
    if f is not None:
        pyplot.xticks(range(x + 2), rotation=45)
        xlabels = [item.get_text() for item in ax.get_xticklabels()]
        for i in range(len(f)):
            xlabels[i + 1] = f[i][0].decode('utf-8')
        ax.set_xticklabels(xlabels)
    if e is not None:
        pyplot.yticks(range(y + 2))
        ylabels = [item.get_text() for item in ax.get_yticklabels()]
        for i in range(len(e)):
            ylabels[i + 1] = e[i][0].decode('utf-8')
        if y > len(e):
            ylabels[len(e) + 1] = "NULL"
        ax.set_yticklabels(ylabels)

    # Hover effect
    points_with_annotation = []
    for i in range(1, x + 1):
        for j in range(1, y + 1):
            point, = plt.plot(
                i, j, "o", c=str(colours[i - 1][j - 1]), markersize=10)
            annotation = ax.annotate(
                score[i - 1][j - 1],
                xy=(i, j), xycoords='data',
                xytext=(i, j), textcoords='data',
                horizontalalignment="left",
                # arrowprops=dict(arrowstyle="simple",
                #                 connectionstyle="arc3,rad=-0.2"),
                bbox=dict(boxstyle="round", facecolor="w",
                          edgecolor="0.5", alpha=0.9)
            )
            # by default, disable the annotation visibility
            annotation.set_visible(False)
            points_with_annotation.append([point, annotation])

    def on_move(event):
        visibility_changed = False
        for point, annotation in points_with_annotation:
            should_be_visible = (point.contains(event)[0] is True)

            if should_be_visible != annotation.get_visible():
                visibility_changed = True
                annotation.set_visible(should_be_visible)

        if visibility_changed:
            plt.draw()
    on_move_id = fig.canvas.mpl_connect('motion_notify_event', on_move)
    plt.scatter([1, x], [1, y], s=1)

    # Display alignment
    _axs = []
    _ays = []
    count = 1
    for entry in align:
        f = entry[0]
        e = entry[1]
        while f != count:
            _axs.append(count)
            _ays.append(y)
            count += 1
        _axs.append(f)
        _ays.append(e)
        count += 1

    plt.plot(_axs, _ays, 'r', marker='+', markersize=10)

    # Display scores
    if dispValue:
        for i in range(x):
            for j in range(y):
                plt.annotate(score[i][j], xy=(i + 1, j + 1))

    if output is None:
        pyplot.draw()
        return

    if not output.endswith(".pdf"):
        output += ".pdf"

    plt.subplots_adjust(top=0.88)
    fig.savefig(output, bbox_inches='tight', dpi=fig.dpi)
    return


def showPlot():
    pyplot.show()


def addAlignmentToFigure(alignment, figureIndex, colour='#FFA500'):
    if figureIndex >= len(figures):
        print "duang", figureIndex, len(figures)
        return
    fig, ax = figures[figureIndex]
    _axs = [entry[0] for entry in alignment]
    _ays = [entry[1] for entry in alignment]

    ax.scatter(_axs, _ays, s=200, marker='s', facecolors='none',
               edgecolors=colour, linewidth='3')


if __name__ == '__main__':
    alignment = [
        (1, 2),
        (2, 3),
        (3, 4)
    ]
    alignment2 = [
        (4, 7),
        (5, 8),
        (6, 9)
    ]

    f = (u"记者 赶到 时 ， 不少 村民 正聚集 在 一 起 ， 对 这起 案件 议论 纷纷 。"
         ).encode("utf-8").split()
    e = (
        u"when our journalist arrived , many villagers had gathered . " +
        u"they were talking widely about the case .").encode("utf-8").split()
    f = [(word,) for word in f]
    e = [(word,) for word in e]

    score = 1. / (np.arange(len(f) * len(e)).reshape((len(e), len(f))) + 1).T
    plotAlignmentWithScore(score, alignment, f, e)
    plotAlignmentWithScore(score, alignment, f, e)

    addAlignmentToFigure(alignment2, 0)
    addAlignmentToFigure(alignment2, 1)
    showPlot()
