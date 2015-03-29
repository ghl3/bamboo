
from pandas.core.groupby import DataFrame, GroupBy
from pandas.core.groupby import DataFrameGroupBy
from pandas.core.groupby import SeriesGroupBy

from singledispatch import singledispatch
from inspect import getmembers, isfunction


from bamboo import groups
from bamboo import plotting


# This module defines the core API and handles
# polymorphic dispatch routing to various
# functions throughout the bamboo code

@singledispatch
def head(df, n=5):
    """
    A more powerful version of pandas 'head'
    that works nicely with DataFrameGroupBy objects
    """

    return df.head(n)


@head.register(GroupBy)
def _(dfgb, n=5, ngroups=5):
    return groups.take_groups(dfgb, ngroups).apply(lambda x: x.head(n))


@singledispatch
def hist(df, *args, **kwargs):
    return df.hist(*args, **kwargs)


@hist.register(SeriesGroupBy)
def _(sgb, *args, **kwargs):
    plotting._series_hist(sgb, *args, **kwargs)


@hist.register(DataFrameGroupBy)
def _(sgb, *args, **kwargs):
    plotting._frame_hist(sgb, *args, **kwargs)


@singledispatch
def scatter(df, x, y, **kwargs):
    pass


@scatter.register(DataFrame)
def _(df, x, y, **kwargs):
    return frames.scatter(df, x, y, **kwargs)


@scatter.register(GroupBy)
def _(df, x, y, **kwargs):
    return groups.scatter(df, x, y, **kwargs)

