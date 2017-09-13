# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib import rcParams, rc, font_manager
rc('pdf', fonttype=42)
__version__ = "0.5a"


prop = None
# In case the characters are not display correctly, one can use the following
# two lines of code to manually load fonts.
# font_path = "/System/Library/Fonts/STHeiti Light.ttc"
# prop = font_manager.FontProperties(fname=font_path)


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
        if prop is not None:
            ax.set_xticklabels(xlabels, fontproperties=prop)
        else:
            ax.set_xticklabels(xlabels)
    if e is not None:
        pyplot.yticks(range(y + 2))
        ylabels = [item.get_text() for item in ax.get_yticklabels()]
        for i in range(len(e)):
            ylabels[i + 1] = e[i][0].decode('utf-8')
        if y > len(e):
            ylabels[len(e) + 1] = "NULL"
        if prop is not None:
            ax.set_yticklabels(ylabels, fontproperties=prop)
        else:
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
    print "This run is for debug purpose only"
    alignment = [
        (1, 10), (2, 7), (3, 3), (4, 1), (5, 2), (6, 3)
    ]
    goldAlignment = [
        (1, 10), (1, 9), (2, 7), (2, 6), (2, 8), (3, 4), (3, 5), (4, 1),
        (5, 2), (6, 3)
    ]
    score = [[-69.83854259, -50.24906734, -40.05389257, -52.78227564,
              -48.95559616, -83.05203389, -41.84190419, -36.80988323,
              -53.95177903, -5.51836396, -52.21216642],
             [-52.62003578, -48.66259046, -49.63266598, -48.3515764,
              -47.78903018, -47.54357859, -20.80823706, -23.82287451,
              -46.78370645, -44.94213171, -54.38097957],
             [-67.21835653, -65.00201479, -62.33380346, -64.35327028,
              -63.36921806, -63.823067, -61.77243324, -61.50220974,
              -63.21514471, -63.30335522, -69.67085268],
             [-71.601932, -103.69747858, -107.72672434, -130.7129619,
              -124.35902727, -139.03331623, -101.91164878, -116.38019098,
              -129.80343778, -102.51292713, -110.36482535],
             [-144.83428264, -111.05359273, -117.6295708, -113.49649321,
              -125.38998938, -114.11012588, -114.2928265, -126.00118102,
              -115.6176449, -117.29842781, -120.46454762],
             [-154.63301046, -150.83650201, -151.92970335, -152.40304942,
              -152.70031894, -153.19412, -153.31395162, -153.80551227,
              -154.39353667, -154.82147064, -159.91620834]]
    f = (u"阿富汗 阅兵典礼 遭攻击 三名 激进分子 被击毙 。"
         ).encode("utf-8").split()
    e = (u"three militants killed in attack on troop parade in afghanistan" +
         u" . NULL").encode("utf-8").split()
    f = [(word,) for word in f]
    e = [(word,) for word in e]

    alignment2 = [(1, 1), (2, 2)]
    goldAlignment2 = [(1, 1), (2, 2)]
    score2 = [[-2.14112664, -38.80694869, -50.13921307],
              [-36.93361547, -3.30257487, -51.00374225]]
    f2 = (u"对 。").encode("utf-8").split()
    e2 = (u"right . NULL").encode("utf-8").split()
    f2 = [(word,) for word in f2]
    e2 = [(word,) for word in e2]

    plotAlignmentWithScore(score, alignment, f, e)
    plotAlignmentWithScore(score2, alignment2, f2, e2)

    addAlignmentToFigure(goldAlignment, 0)
    addAlignmentToFigure(goldAlignment2, 1)
    showPlot()
