
from pandas.core.groupby import DataFrame, GroupBy
from pandas.core.groupby import DataFrameGroupBy
from pandas.core.groupby import SeriesGroupBy

from singledispatch import singledispatch

from bamboo import frames
from bamboo import groups
from bamboo import series_groups

from bamboo import reports

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
    pass


@hist.register(DataFrame)
def _(df, *args, **kwargs):
    return frames.hist(df, *args, **kwargs)


@hist.register(SeriesGroupBy)
def _(sgb, *args, **kwargs):
    return series_groups.hist(sgb, *args, **kwargs)


@hist.register(DataFrameGroupBy)
def _(dfgb, *args, **kwargs):
    return groups.hist(dfgb, *args, **kwargs)


@singledispatch
def scatter(df, x, y, **kwargs):
    pass


@scatter.register(DataFrame)
def _(df, x, y, **kwargs):
    return frames.scatter(df, x, y, **kwargs)


@scatter.register(DataFrameGroupBy)
def _(dfgb, x, y, **kwargs):
    return groups.scatter(dfgb, x, y, **kwargs)


def hist_all(*args, **kwargs):
    return groups.hist_all(*args, **kwargs)


def save_report(*args, **kwargs):
    return reports.save_report(*args, **kwargs)
