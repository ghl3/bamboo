
from pandas.core.groupby import DataFrame, GroupBy

from singledispatch import singledispatch
from inspect import getmembers, isfunction

from bamboo import groups
from BambooObjects import _wrap_with_bamboo


def wrap(obj):
    return _wrap_with_bamboo(obj)


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


@hist.register(GroupBy)
def _(dfgb, *args, **kwargs):
    return groups.hist(dfgb, *args, **kwargs)


@singledispatch
def scatter(df, x, y, **kwargs):
    pass


@scatter.register(DataFrame)
def _(df, x, y, **kwargs):
    return frames.scatter(df, x, y, **kwargs)


@scatter.register(GroupBy)
def _(df, x, y, **kwargs):
    return groups.scatter(df, x, y, **kwargs)
