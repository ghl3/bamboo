from __future__ import division

from collections import OrderedDict

import pandas as pd

from bamboo import plotting
from helpers import combine_data_frames

import matplotlib.pyplot as plt
from math import ceil


"""
Functions that act on data frame group by objects
"""


def filter_groups(dfgb, filter_function, on_index=False):
    """
    Filter the groups of a DataFrameGroupBy
    and return a DataFrameGroupBy containing
    only the groups that return true.
    In pseudocode:
    return [group for group in dfgb.groups
            if filter_function(group)]
    If on_index is true, the filter function is applied
    to the group value (the index)
    """

    return_groups = []

    for val, group in dfgb:
        if on_index and filter_function(val):
            return_groups.append(group)
        if not on_index and filter_function(group):
            return_groups.append(group)

    return combine_data_frames(return_groups).groupby(dfgb.keys)


def sorted_groups(dfgb, key):
    """
    Return a sorted version of the
    grouped dataframe.  The required
    key function is a function of each
    group and determins the sort order
    """

    sort_list = []

    for name, group in dfgb:

        sort_val = key(group)
        sort_list.append((sort_val, group))

    sorted_groups = [group for val, group
                     in sorted(sort_list, key=lambda x: x[0])]

    return combine_data_frames(sorted_groups).groupby(dfgb.keys, sort=False)


def map_groups(dfgb, func, name=None):
    """
    Take a DataFrameGroupBy and a function that takes a DataFrame
    and returns a Series.  Apply that function to the DataFrame
    of each group and return a SeriesGroupBy with the same grouping
    as the original DataFrameGroupBy but where the value of
    each group is the value of the function applied to the corresponding
    dataframe in the original group.
    """

    if name is None:
        name = func.__name__

    transformed = dfgb.obj.apply(func, axis=1, reduce=False)
    return pd.DataFrame({name: transformed}).groupby(dfgb.grouper)[name]


def take_groups(dfgb, n):
    """
    Return a DataFrameGroupBy holding the
    up to the first n groups
    """

    return_groups = []

    for i, (key, group) in enumerate(dfgb):
        if i >= n:
            break
        return_groups.append(group)

    return combine_data_frames(return_groups).groupby(dfgb.keys)


def pivot_groups(mdfgb, func, **kwargs):
    """
    Pivot a DataFrameGroupBy that is grouped by
    exactly 2 groups (hierarical index)
    and return a new DataFrame that is pivoted by
    those two groups and whose cell value is the
    value of the given function.

    Example:

    df =
    group1 group2  val
       A      X    1
       B      Y    2
       C      Z    3

    pivot_groups(df.groupby(['group1', 'group2']), lambda x: x.val*2)

    group2   X   Y   Z
    group1
    A        2 NaN NaN
    B      NaN   4 NaN
    C      NaN NaN   6
    """

    assert(mdfgb.first().index.nlevels == 2)

    x, y = mdfgb.first().index.names

    mdfgb_applied = mdfgb.apply(func).reset_index(level=[x, y])

    c = [column for column in mdfgb_applied if column not in (x, y)][0]

    return mdfgb_applied.pivot_table(index=x, columns=y, values=c, **kwargs)


def apply_groups(dfgb, *args, **kwargs):
    """
    Return a DataFrameGroupBy object with each
    group of the original DataFrameGroupBy
    transformed into a new dataframe, where
    each column of these new dataframes is the
    result of one of the arg functions supplied.
    """

    res = OrderedDict()
    for idx, arg in enumerate(args):
        name = arg.__name__
        if name == '<lambda>':
            name += '_{}'.format(idx)
        res[name] = dfgb.obj.apply(arg, axis=1, **kwargs)

    return pd.DataFrame(res).groupby(dfgb.grouper)


#
# Plotting
#


def hist(dfgb, *args, **kwargs):
    return plotting._grouped_hist(dfgb, *args, **kwargs)


def scatter(dfgb, *args, **kwargs):
    return plotting._grouped_scatter(dfgb, *args, **kwargs)


def hist_functions(dfgb, *args, **kwargs):
    """
    Take a DataFrameGroupBy and a list of functions and make a
    grid of plots showing a histogram of the values of each
    function applied to each group individually.
    Each function should take a row in the dataframe and
    can reference columns using dict-like access on that row.
    Useful for exploring new features in classification
    """

    if 'cols' in kwargs:
        cols = kwargs['cols']
    else:
        cols = 2

    functions = args

    rows = ceil(len(functions) / cols)

    for i, function in enumerate(functions):
        plt.subplot(rows, cols, i + 1)
        plotting._series_hist(map_groups(dfgb, function), **kwargs)
        plt.xlabel(function.__name__)


def stacked_counts_plot(dfgb, category, ratio=False, **kwargs):
    """
    Takes a dataframe that has been grouped and
    plot a stacked bar-plot representing the makeup of
    the input category per group.
    Optionally, show the ratio of the category in each group.

    Convert this:
    group   category
    1       A
    1       A
    2       A
    1       B
    2       B

    Into this:
    1: A A B
    2: A B
    as a barplot

    Motivating Example:
    - group = year
    - category = product name

    This would then put the years on the
    x axis and would show a stack of the number
    of each product sold in that year
    """

    counts = dfgb[category].value_counts().unstack().fillna(0.)

    if ratio:
        denom = counts.sum(axis=1)
        counts = counts.divide(denom, axis='index')

    ax = plt.gca()
    plot = counts.plot(kind="bar", stacked=True, subplots=False, ax=ax, **kwargs)
    for container in plot.containers:
        plt.setp(container, width=1)


def save_grouped_hists(dfgb, output_file, title=None, *args, **kwargs):
    plotting._save_plots(dfgb, plotting._series_hist, output_file, title, *args, **kwargs)


def hist_all(dfgb, n_columns=3, figsize=(12, 4), plot_func=None,
             *args, **kwargs):
    """
    dfgb - A grouped DataFrame.  Every column in the group will be plotted

    n_columns - The number of columns to show (the number of rows is determined
      by the number of columns variables in the input DataFrameGroupBy

    figsize - The size of each figure

    plot_func - A function that takes a SeriesGroupBy object and plots it.
      One can optionally supply a specific function to plot as one pleases.
      All additional args and kwargs for hist_all will be passed to plot_func.

      If one chooses to not supply a specific function, the default _plot_and_decorate
      function will be used.  The default _plot_and_decorate function accepts
      the following arguments (in addition to others):

      - binning_map: A dictionary of {name: [binning]}, where the 'name' is the name
        of each feature and the given binning array is used as the binning of that feature
      - title_map: A dictionary of {name: title} for each feature
      - ylabel_map: A dictionary of {name: ylabel} for each feature
    """

    columns = dfgb.obj.columns

    if plot_func is None:
        plot_func = plotting._plot_and_decorate

    for i, feature in enumerate(columns):

        if i % n_columns == 0:
            plt.figure(figsize=figsize)

        plt.subplot(1, n_columns, (i % n_columns) + 1)

        try:
            sgb = dfgb[feature]
            plot_func(sgb, *args, **kwargs)
        except Exception as e:
            print e


