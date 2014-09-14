from __future__ import division

import math

import pandas as pd
from pandas import DataFrame

import matplotlib.pyplot as plt
from math import floor, ceil

import numpy as np

from subplots import PdfSubplots
from matplotlib.backends.backend_pdf import PdfPages

from frames import group_by_binning
from groups import map_groups
from helpers import NUMERIC_TYPES


def hexbin(df, x, y, **kwargs):
    plt.hexbin(df[x].values, df[y].values)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.colorbar()


def boxplot(df, x, y, bins=None):
    """ Draw a boxplot using the
    variables x and y.
    Include ticks
    If 'bins' is not none, group the x-axis
    variable into the supplied bins
    """

    from pandas.tools.plotting import boxplot_frame_groupby
    if bins is None:
        boxplot_frame_groupby(df.groupby(x)[y], subplots=False, return_type='dict')
    else:
        boxplot_frame_groupby(group_by_binning(df, x, bins)[y], subplots=False, return_type='dict')


def hist(dfgb, var=None, *args, **kwargs):
    """
    Takes a data frame (grouped by a variable)
    and plots histograms of the variable 'var'
    for each of the groups.
    """

    if isinstance(dfgb, pd.core.groupby.SeriesGroupBy):
        _series_hist(dfgb, *args, **kwargs)

    elif isinstance(dfgb, pd.core.groupby.DataFrameGroupBy):
        if var is not None:
            _series_hist(dfgb[var], *args, **kwargs)
        else:
            for (var, series) in dfgb._iterate_column_groupbys():
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


def _series_hist(dfgb, ax=None, normed=False, normalize=False, autobin=False, *args, **kwargs):
    """
    Takes a pandas.SeriesGroupBy
    and plots histograms of the variable 'var'
    for each of the groups.
    """
    if ax is None:
        ax = plt.gca()

    normed_or_normalize = normed or normalize

    if dfgb.obj.dtype in NUMERIC_TYPES:
        _series_hist_float(dfgb, ax, normed=normed_or_normalize, autobin=autobin, *args, **kwargs)
    else:
        _series_hist_nominal(dfgb, ax, normalize=normed_or_normalize, *args, **kwargs)

    plt.legend(loc='best', fancybox=True)


def _series_hist_float(dfgb, ax=plt.gca(), autobin=False, normed=False, normalize=False,
                       stacked=False, *args, **kwargs):
    """
    Takes a pandas.SeriesGroupBy
    and plots histograms of the variable 'var'
    for each of the groups.
    """

    if autobin and 'bins' not in kwargs:
        kwargs['bins'] = get_variable_binning(dfgb.obj)

    color_cycle = ax._get_lines.color_cycle

    for (color, (key, srs)) in zip(color_cycle, dfgb):

        if 'label' in kwargs.keys():
            label = kwargs['label']
        else:
            label = key

        if 'color' in kwargs.keys():
            color = kwargs['color']

        srs.hist(ax=ax, color=color, label=str(label), normed=normed, **kwargs)


def _series_hist_nominal(dfgb, ax=None, normalize=False, *args, **kwargs):
    """
    Takes a pandas.SeriesGroupBy
    and plots histograms of the variable 'var'
    for each of the groups.
    """

    color_cycle = ax._get_lines.color_cycle

    for (color, (key, srs)) in zip(color_cycle, dfgb):

        if 'label' in kwargs.keys():
            label = kwargs['label']
        else:
            label = key

        if 'color' in kwargs.keys():
            color = kwargs['color']

        value_counts = srs.value_counts(normalize=normalize).sort_index()
        value_counts.plot(kind='bar', ax=ax, color=color, label=label, **kwargs)


def hist_functions(dfgb, functions, cols=2, **kwargs):
    """
    Take a DataFrameGroupBy and a list of functions and make a
    grid of plots showing a histogram of the values of each
    function applied to each group individually.
    Each function should take a row in the dataframe and
    can reference columns using dict-like access on that row.
    Useful for exploring new features in classification
    """

    rows = math.ceil(len(functions) / cols)

    for i, function in enumerate(functions):
        plt.subplot(rows, cols, i+1)
        hist(map_groups(dfgb, function), **kwargs)
        plt.xlabel(function.__name__)


def scatter(dfgb, x=None, y=None, **kwargs):
    """
    Takes a grouped data frame and draws a scatter
    plot of the suppied variables wtih a different
    color for each group
    """
    ax = plt.gca()
    color_cycle = ax._get_lines.color_cycle
    for (color, (key, grp)) in zip(color_cycle, dfgb):
        plt.scatter(grp[x], grp[y], color=color, label=key, **kwargs)
    plt.legend(loc='best')
    plt.xlabel(x)
    plt.ylabel(y)


def stacked_counts_plot(dfgb, category, ratio=False, **kwargs):
    """
    Takes a dataframe that has been grouped and
    plot a stacked bar-plot representing the makeup of
    the input category per group.
    Optionally, show the ratio of the category in each group.

    Convert this:
    category  group
    A         1
    A         1
    A         2
    B         1
    B         2

    Into this:
    1: A A B
    2: A B
    as a barplot
    """

    counts = dfgb[category].value_counts().unstack().fillna(0.)

    if ratio:
        denom = counts.sum(axis=1)
        counts = counts.divide(denom, axis='index')

    ax = plt.gca()
    plot = counts.plot(kind="bar", stacked=True, subplots=False, ax=ax, **kwargs)
    for container in plot.containers:
        plt.setp(container, width=1)


def _draw_stacked_plot(dfgb, **kwargs):
    """
    Draw a vertical bar plot of multiple series
    stacked on top of each other

    Deals with some annoying pandas issues when
    drawing a DataFrame
    """

    #color_cycle = ax._get_lines.color_cycle

    series_dict = {}

    for (key, srs) in dfgb:
        series_dict[key] = srs

    df_for_plotting = pd.DataFrame(series_dict)
    df_for_plotting.plot(kind="bar", stacked=True, ax=ax, **kwargs)


def _save_plots(dfgb, plot_func, output_file, title=None, *args, **kwargs):
    """
    Take a grouped dataframe and save a pdf of
    the histogrammed variables in that dataframe.
    TODO: Can we abstract this behavior...?
    """

    pdf = PdfPages(output_file)

    if title:
        helpers.title_page(pdf, title)

    subplots = PdfSubplots(pdf, 3, 3)

    for (var, series) in dfgb._iterate_column_groupbys():

        subplots.next_subplot()
        try:
            plot_func(series, *args, **kwargs)
            plt.xlabel(var)
            subplots.end_iteration()
        except :
            subplots.skip_subplot()

    subplots.finalize()

    pdf.close()


def save_grouped_hists(dfgb, output_file, title=None, *args, **kwargs):
    _save_plots(dfgb, _series_hist, output_file, title, *args, **kwargs)


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