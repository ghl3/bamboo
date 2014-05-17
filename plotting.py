from __future__ import division

import pandas as pd
from pandas import DataFrame

import matplotlib.pyplot as plt
from math import floor, ceil

import numpy as np


class SkipPlot(Exception):
    pass


def hist(grouped, var=None, *args, **kwargs):
    """
    Takes a data frame (grouped by a variable)
    and plots histograms of the variable 'var'
    for each of the groups.
    """

    if isinstance(grouped, pd.core.groupby.SeriesGroupBy):
        _series_hist(grouped, *args, **kwargs)

    elif isinstance(grouped, pd.core.groupby.DataFrameGroupBy):
        if var is not None:
            _series_hist(grouped[var], *args, **kwargs)
        else:
            for (var, series) in grouped._iterate_column_groupbys():
                plt.figure()
                try:
                    _series_hist(series, *args, **kwargs)
                    plt.xlabel(var)
                except TypeError as e:
                    print "Failed to plot %s" % var
                    print e
                finally:
                    pass

    else:
        pass


def _series_hist(grouped, ax=None, normed=False, normalize=False, autobin=False, *args, **kwargs):
    """
    Takes a pandas.SeriesGroupBy
    and plots histograms of the variable 'var'
    for each of the groups.
    """
    if ax is None:
        ax = plt.gca()

    normed_or_normalize = normed or normalize

    if grouped.obj.dtype in NUMERIC_TYPES:
        _series_hist_float(grouped, ax, normed=normed_or_normalize, autobin=autobin, *args, **kwargs)
    else:
        _series_hist_nominal(grouped, ax, normalize=normed_or_normalize, *args, **kwargs)

    plt.legend(loc='best', fancybox=True)


def _series_hist_float(grouped, ax, autobin=False, normed=False, normalize=False,
                       stacked=False, *args, **kwargs):
    """
    Takes a pandas.SeriesGroupBy
    and plots histograms of the variable 'var'
    for each of the groups.
    """

    if autobin and 'bins' not in kwargs:
        kwargs['bins'] = get_variable_binning(grouped.obj)

    color_cycle = ax._get_lines.color_cycle

    for (color, (key, srs)) in zip(color_cycle, grouped):

        if 'label' in kwargs.keys():
            label = kwargs['label']
        else:
            label = key

        if 'color' in kwargs.keys():
            color = kwargs['color']

        srs.hist(ax=ax, color=color, label=label, normed=normed, **kwargs)


def _series_hist_nominal(grouped, ax=None, normalize=False, *args, **kwargs):
    """
    Takes a pandas.SeriesGroupBy
    and plots histograms of the variable 'var'
    for each of the groups.
    """

    color_cycle = ax._get_lines.color_cycle

    for (color, (key, srs)) in zip(color_cycle, grouped):

        if 'label' in kwargs.keys():
            label = kwargs['label']
        else:
            label = key

        if 'color' in kwargs.keys():
            color = kwargs['color']

        srs.value_counts(normalize=normalize).plot(kind='bar', ax=ax, color=color, label=label, **kwargs)


def scatter(grouped, x, y, **kwargs):
    """
    Takes a grouped data frame and draws a scatter
    plot of the suppied variables wtih a different
    color for each group
    """
    ax = plt.gca()
    color_cycle = ax._get_lines.color_cycle
    for (color, (key, grp)) in zip(color_cycle, grouped):
        plt.scatter(grp[x], grp[y], color=color, label=key, **kwargs)
    plt.legend(loc='best')
    plt.xlabel(x)
    plt.ylabel(y)


def binner(divisor):
    """
    Returns a function that rounds a number
    down to the nearst factor of the divisor
    """
    def round_down(num):
        return num - (num % divisor)
    return round_down


def binned_ratio(seriesA, seriesB, bin_width):
    numer = seriesA.map(binner(bin_width)).value_counts()
    denom = seriesB.map(binner(bin_width)).value_counts()
    return numer / denom


def vals_in_range(vals, bins):
    return len([val for val in vals
                if val >= bins[0]
                and val <= bins[-1]])


def add_label(labels):
    xy = (0.55, 0.95)
    for name, val in labels.iteritems():
        plt.annotate("{0}: {1}".format(name, val),
                     xy=xy, xycoords='axes fraction')
        xy = (xy[0], xy[1]-0.04)


def get_variable_binning(var, nbins=10, int_bound=40):
    """
    Get the binning of a variable.
    Deals with a number of special cases.
    For smooth distributions most of the time the maximum (minimum)
    occurs within 15% of the 98th (2nd) percentile, so we define
    extreme outliers as:

    if ( (var_max > p_50 + (p_98 - p_50) * 1.15) or
         (var_min < p_50 - (p_50 - p_02) * 1.15) )

    If an outlier is present, then use the expanded (by 15%) 98th
    (or 2nd) percentile as the bin edge. Otherwise we use the actual extremum.
    """

    var_min = min(var)
    var_max = max(var)

    if var_min == var_max:
        return np.array([var_min-0.5, var_max+0.5])

    # If all values are integers (not necessarily by type) between
    # -int_bound and +int_bound, then use unit spacing centered on
    # integers.
    if var_min > -int_bound and var_max < int_bound:

        if all(np.equal(np.mod(var, 1), 0)):
            return np.arange(var_min-0.5, var_max+1.5, 1)

    # Detect extreme outliers by the following heuristic.
    p_02, p_50, p_98 = np.percentile(var, (2, 50, 98))

    p_02_exp = p_50 - (p_50 - p_02) * 1.15
    p_98_exp = p_50 + (p_98 - p_50) * 1.15

    if (var_max > p_98_exp):
        var_max = p_98_exp

    if (var_min < p_02_exp):
        var_min = p_02_exp

    bins = np.arange(nbins+1)/nbins * (var_max - var_min) + var_min
    return bins


def boxplot(df, xaxis, yaxis):
    """ Draw a boxplot using the
    variables xaxis and yaxis.
    Include ticks
    """

    if xaxis not in df:
        print "x axis " + xaxis + " not in DataFrame: ", df
        print df.columns
        raise Exception()

    if yaxis not in df:
        print "y axis " + yaxis + " not in DataFrame: ", df
        print df.columns
        raise Exception()

    from pandas.tools.plotting import boxplot_frame_groupby
    boxplot_frame_groupby(df.groupby(xaxis)[yaxis], subplots=False)


def stacked_counts_plot(df, xaxis, categories, ratio=False, **kwargs):
    """
    Draw a plot of the counts per category in categories
    per bin of x-axis.

    Convert this:
    class  day
    A      1
    A      1
    A      2
    B      1
    B      2

    Into this:
    A A B
    A B
    as a barplot
    """

    counts = df.groupby(xaxis)[categories].value_counts().unstack().fillna(0.)

    if ratio:
        denom = counts.sum(axis=1)
        counts = counts.divide(denom, axis='index')

    ax = plt.gca()
    plot = counts.plot(kind="bar", stacked=True, subplots=False, ax=ax, **kwargs)
    for container in plot.containers:
        plt.setp(container, width=1)


def _draw_stacked_plot(grouped, **kwargs):
    """
    Draw a vertical bar plot of multiple series
    stacked on top of each other

    Deals with some annoying pandas issues when
    drawing a DataFrame
    """

    #color_cycle = ax._get_lines.color_cycle

    series_dict = {}

    for (key, srs) in grouped:
        series_dict[key] = srs

    df_for_plotting = pd.DataFrame(series_dict)
    df_for_plotting.plot(kind="bar", stacked=True, ax=ax, **kwargs)


NUMERIC_TYPES = ('bool_', 'int_', 'intc', 'intp', 'int8',
                 'int16', 'int32', 'int64', 'uint8',
                 'uint16', 'uint32', 'uint64', 'float_',
                 'float16', 'float32', 'float64')
