
from collections import OrderedDict

import pandas as pd

from helpers import combine_data_frames

from types import *
import inspect

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
    Take a DataFrameGroupBy and apply a function
    to the DataFrames, returning a seriesgroupby
    of the values
    """

    if name is None:
        name = func.__name__

    transformed = dfgb.obj.apply(func, axis=1, reduce=False)
    return pd.DataFrame({name: transformed}).groupby(dfgb.grouper)


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


def pivot_groups(dfgb, **kwargs):
    """
    Pivot a DataFrameGroupBy.
    Takes all the normal keyword args of a pivot
    """
    return dfgb.reset_index().pivot(**kwargs)


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

