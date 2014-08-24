

import pandas


from bamboo import combine_data_frames

"""

Functions that act on data frame group by objects

"""

def filter_groups(dfgb, filter_function):
    """
    Filter the groups of a DataFrameGroupBy
    and return a DataFrameGroupBy containing
    only the groups that return true.
    In pseudocode:
    return [group for group in dfgb.groups
            if filter_function(group)]
    """

    return_groups = []

    for i in range(0, dfgb.ngroups):
        group = dfgb.get_group(i)
        if filter_function(group):
            return_groups.append(group)

    return combine_data_frames(return_groups).groupby(dfgb.keys)


def sort(grouped, key):

    sort_list = []

    for name, group in grouped:
        key_val = key(group)
        sort_list.append((key_val, (name, group)))

    df = pandas.DataFrame

    return sorted(sort_list, key=lambda x: x[1])


def groupmap(grouped, func):
    """
    Take a DataFrameGroupBy and apply a function
    to the DataFrames, returning a seriesgroupby
    of the values
    """
    # grouped.index.name
    transformed = grouped.obj.apply(func, axis=1)
    return transformed.groupby(grouped.grouper)


def take_groups(dfgb, n):
    """
    Return a DataFrameGroupBy holding the
    up to the first n groups
    """

    return_groups = []

    for i in range(0, min(dfgb.ngroups, n)):
        return_groups.append(dfgb.get_group(i))

    return combine_data_frames(return_groups).groupby(dfgb.keys)


