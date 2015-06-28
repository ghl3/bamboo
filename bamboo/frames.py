from __future__ import division

import pandas as pd

from helpers import NUMERIC_TYPES, convert_nominal_to_int

from collections import OrderedDict
import matplotlib.pyplot as plt

from bamboo import plotting

"""
Functions that act on data frames
"""


def exclude(df, exclude):
    """
    Return a DataFrame with a set of feature excluded
    """
    keep = [column for column in df.columns
            if column not in exclude]
    return df[keep]


def partition(df, partition_function):
    """
    Group a DataFrame based on the value of a function
    evaluated over the rows of the dataframe.
    """

    partition_values = df.apply(partition_function, axis=1)
    return df.groupby(partition_values)


def split(df, columns, exclude=None):
    """
    Separate a dataframe column-wise and turn a tuple
    of all the given columns on one side and the remaining
    columns on the other side
    """

    if exclude is None:
        exclude = []

    left = columns
    right = [column for column in df.columns
             if column not in left
             and column not in exclude]

    return (df[left], df[right])


def take(df, var, exclude=None):
    """
    Separate a dataframe column-wise and turn a tuple
    of a single column on one side and the remaining columns
    on the other side.
    """

    if exclude is None:
        exclude = []

    rest = [column for column in df.columns
            if column != var
            and column not in exclude]

    return (df[var], df[rest])


def sort_rows(df, key=None):
    """
    Sort a dataframe's rows.
    If required, a key function should be
    a function that acts on a row and
    orders the dataframe by the value
    of the function evaluated on every row.
    """

    if not key:
        return df.sort(inplace=False)

    indices = df.apply(key, axis=1).sort(inplace=False).index
    return df.ix[indices]


def sort_columns(df, key, by_name=False):
    """
    Sort the columns of a DataFrame by the given key function
    - If by_name is true, the key function must take the column
      names as an argument
    - if by_name is false, the key function must take the
      columns themselves as an argument (and must return a sortable object)
    DataFrame by its columns, either by
    functions of the column names or by functions
    of the values of the columns themselves.
    """

    items = []

    for idx, colname in enumerate(df):

        if by_name:
            val = key(colname)
        else:
            val = key(df[colname])

        items.append((idx, val))

    items.sort(key=lambda x: x[1])

    return df.icol([idx for idx, _ in items])


def apply_all(df, *args, **kwargs):
    """
    Return a new dataframe consisting of
    a number of functions applied to the
    current dataframe

    If the input object is a DataFrameGroupBy,
    the args functions should (but don't have to)
    be aggregations on the columns that return
    single variables (such that each group has
    only one value)
    """

    if isinstance(df, pd.DataFrame) and 'axis' not in kwargs:
        kwargs['axis'] = 1

    res = OrderedDict()
    for idx, arg in enumerate(args):
        name = arg.__name__
        if name == '<lambda>':
            name += '_{}'.format(idx)
        res[name] = df.apply(arg, **kwargs)
    return pd.DataFrame(res)


def map_functions(df, functions):
    """
    Takes an input data frame and a map of
    {function names: functions} that each are
    to act row-wise on the input dataframe.

    Return a DataFrame whose columns are
    the result of those functions on the
    input dataframe.
    functions - either a list of functions or
    a map of names to functions
    """

    column_map = {}

    try:
        # If it's dict like
        for name, func in functions.iteritems():
            column_map[name] = df.apply(func, axis=1)
    except AttributeError:
            # Assume it's list like
        for func in functions:
            column_map[func.__name__] = df.apply(func, axis=1)

    return pd.DataFrame(column_map, index=df.index)


def with_new_columns(df, functions):
    """
    Take an input DataFrame and a set of functions
    whose values represent new columns and return
    the original DataFrame with the new columns
    added based on mapping those functions over
    the rows of the input DataFrame
    """
    return df.join(map_functions(df, functions))


def get_numeric_features(df):
    float_feature_names = [feature for feature in df.columns
                           if df[feature].dtype in NUMERIC_TYPES]
    return df[float_feature_names]


def convert_nominals_to_int(df):
    ret = pd.DataFrame()
    for column_name in df:
        column = df[column_name]
        if column.dtype not in NUMERIC_TYPES:
            ret[column_name] = convert_nominal_to_int(column)
        else:
            ret[column_name] = column
    return ret


def get_index_rows(df, indices):
    """
    Given a dataframe and a list of indices,
    return a list of row numbers corresponding
    to the supplied indices (maintaining order)
    """
    rows = []
    for i, (index, row) in enumerate(df.iteritems()):
        if index in indices:
            rows.append(i)
    return rows


def group_by_binning(df, column, bins):
    """
    Return a dataframe grouped by the given
    column being within the bins.
    """

    def grouper(x):
        for i in range(len(bins[:-1])):
            left, right = bins[i], bins[i + 1]
            if x >= left and x < right:
                return "({}, {})".format(left, right)
        return None

    return df.groupby(df[column].map(grouper))


def get_columns_with_null(df):
    null_counts = pd.isnull(df).sum()
    return null_counts[null_counts > 0]


def get_rows_with_null(df):
    """
    Return a DataFrame of all rows that
    contain a null value in ANY column
    """
    null_row_counts = pd.isnull(df).sum(axis=1)
    return df[null_row_counts > 0]


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


def hist(df, *args, **kwargs):
    return df.hist(*args, **kwargs)


def scatter(df, *args, **kwargs):
    return plotting._frame_scatter(*args, **kwargs)
