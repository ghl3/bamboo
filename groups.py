
import pandas

from helpers import combine_data_frames

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
        print name, sort_val
        sort_list.append((sort_val, group))

    sorted_groups = [group for val, group
                     in sorted(sort_list, key=lambda x: x[0])]

    return combine_data_frames(sorted_groups).groupby(dfgb.keys, sort=False)


def groupmap(grouped, func):
    """
    Take a DataFrameGroupBy and apply a function
    to the DataFrames, returning a seriesgroupby
    of the values
    """

    transformed = grouped.obj.apply(func, axis=1)
    return transformed.groupby(grouped.grouper)


def take_groups(dfgb, n):
    """
    Return a DataFrameGroupBy holding the
    up to the first n groups
    """

    return_groups = []

    for i, (key, group) in enumerate(dfgb):
    #for i in range(0, min(dfgb.ngroups, n)):
        if i >= n:
            break
        return_groups.append(group) #dfgb.get_group(i))

    return combine_data_frames(return_groups).groupby(dfgb.keys)


