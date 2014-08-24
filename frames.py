from __future__ import division

import pandas as pd

from helpers import NUMERIC_TYPES

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
    Separate a dataframe row-wise based on the
    value of the supplied partition function
    """

    partition_values = df.apply(partition_function, axis=1)
    return df.groupby(partition_values)


def split(df, columns, exclude=None):
    """
    Separate a dataframe column-wise and turn a tuple
    of all the given columns on one side and the remaining
    columns on the other side
    """

    if exclude==None:
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

    if exclude==None:
        exclude = []

    rest = [column for column in df.columns
             if column != var
             and column not in exclude]

    return (df[var], df[rest])


def apply_all(df, *args, **kwargs):
    """
    Return a new dataframe consisting of
    a number of functions applied to the
    current dataframe
    """

    if 'axis' not in kwargs:
        kwargs['axis']=1

    res = {}
    for arg in args:
        res[arg.__name__] = df.apply(arg, **kwargs)
    return pd.DataFrame(res)


def map_functions(df, functions):
    """
    Takes an input data frame and a map of
    function namesto functions that each are
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


def get_nominal_integer_dict(nominal_vals):
    """
    Takes a set of nominal values (non-numeric)
    and returns a dictionary that maps each value
    to a unique integer
    """
    d = {}
    for val in nominal_vals:
        if val not in d:
            current_max = max(d.values()) if len(d) > 0 else -1
            d[val] = current_max+1
    return d


def convert_to_integer(srs, return_dict=False):
    """
    Convert a series of nominal values
    into a series of integers
    """
    d = get_nominal_integer_dict(srs)
    result =  srs.map(lambda x: d[x])
    if return_dict:
        return (result, d)
    else:
        return result


def convert_strings_to_integer(df):
    ret = pd.DataFrame()
    for column_name in df:
        column = df[column_name]
        if column.dtype not in NUMERIC_TYPES:
            ret[column_name] = convert_to_integer(column)
        else:
            ret[column_name] = column
    return ret


def get_index_rows(srs, indices):
    """
    Given a dataframe and a list of indices,
    return a list of row numbers corresponding
    to the supplied indices (maintaining order)
    """
    rows = []
    for i, (index, row) in enumerate(srs.iteritems()):
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
            left, right = bins[i], bins[i+1]
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



def combine_data_frames(frames):
    """
    Helper function to combine multiple data frames
    into a single data frame
    """

    if len(frames)==0:
        return pd.DataFrame()
    elif len(frames)==1:
        return frames[0]
    else:
        first = frames[0]
        return first.append(list(frames[1:]))


