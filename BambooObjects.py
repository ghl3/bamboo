
import pandas as pd
import pandas
from pandas.core.groupby import GroupBy

from inspect import getmembers, isfunction

import bamboo, groups, frames, plotting


_bamboo_methods = {name:func  for name, func in getmembers(bamboo) if isfunction(func)}
_frames_methods = {name:func  for name, func in getmembers(frames) if isfunction(func)}
_groups_methods = {name:func  for name, func in getmembers(groups) if isfunction(func)}
_plot_methods = {name:func  for name, func in getmembers(plotting) if isfunction(func)}


def _wrap_with_bamboo(obj):

    print "Insider the wrapper: %s" % type(obj)

    if isinstance(obj, pandas.DataFrame):
        print "Making it a BambooDataFrame"
        return BambooDataFrame(obj)

    elif isinstance(obj, pandas.core.groupby.SeriesGroupBy):
        return BambooSeriesGroupBy(obj)

    elif isinstance(obj, pandas.core.groupby.DataFrameGroupBy):
        return BambooDataFrameGroupBy(obj)

    else:
        print "Using native method"
        return obj


def _create_bamboo_wrapper(obj, func):
    print "Wrapping with bamboo"
    def callable(*args, **kwargs):
        res = func(obj, *args, **kwargs)
        return _wrap_with_bamboo(res)
    return callable

# def _create_bamboo_wrapper_2(func):
#     print "Wrapping with bamboo"
#     def callable(*args, **kwargs):
#         res = func(*args, **kwargs)
#         return _wrap_with_bamboo(res)
#     return callable


class BambooDataFrame(pandas.DataFrame):

    def __init__(self, *args, **kwargs):
        super(BambooDataFrame, self).__init__(*args, **kwargs)

#    def __getattribute__(self, name):
#        return  _create_bamboo_wrapper(self, object.__getattribute__(self, name))

    def __getattr__(self, *args, **kwargs):

        name, pargs = args[0], args[1:]

        print "Getting function: %s" % name

        if name in _bamboo_methods:
            return _create_bamboo_wrapper(self, _bamboo_methods[name])

        if name in _plot_methods:
            return _create_bamboo_wrapper(self, _plot_methods[name])

        if name in _frames_methods:
            return _create_bamboo_wrapper(self, _frames_methods[name])

        print "Using native method"
        return _create_bamboo_wrapper(self, super(BambooDataFrame, self).__getattribute__(*args, **kwargs))
#        return super(BambooDataFrame, self).__getattribute__(*args, **kwargs)


class BambooDataFrameGroupBy(pandas.core.groupby.DataFrameGroupBy):

    def __init__(self, other):
        if not isinstance(other, pandas.core.groupby.DataFrameGroupBy):
            raise TypeError()

        super(BambooDataFrameGroupBy, self).__init__(
                                             other.obj,
                                             other.keys,
                                             other.axis,
                                             other.level,
                                             other.grouper,
                                             other.exclusions,
                                             other._selection,
                                             other.as_index,
                                             other.sort,
                                             other.group_keys,
                                             other.squeeze)

    def head(self, *args, **kwargs):
        return bamboo.head(self, *args, **kwargs)

    def __getattr__(self, *args, **kwargs):

        name, pargs = args[0], args[1:]

        if name in _bamboo_methods:
            return _create_bamboo_wrapper(self, _bamboo_methods[name])

        if name in _plot_methods:
            return _create_bamboo_wrapper(self, _plot_methods[name])

        if name in _groups_methods:
            return _create_bamboo_wrapper(self, _groups_methods[name])

        return _create_bamboo_wrapper(self, super(BambooDataFrameGroupBy, self).__getattribute__(*args, **kwargs))
#        return super(BambooDataFrameGroupBy, self).__getattribute__(*args, **kwargs)


class BambooSeriesGroupBy(pandas.core.groupby.SeriesGroupBy):

    def __init__(self, other):
        if not isinstance(other, pandas.core.groupby.SeriesGroupBy):
            raise TypeError()
        super(BambooSeriesGroupBy, self).__init__(
                                             other.obj,
                                             other.keys,
                                             other.axis,
                                             other.level,
                                             other.grouper,
                                             other.exclusions,
                                             other._selection,
                                             other.as_index,
                                             other.sort,
                                             other.group_keys,
                                             other.squeeze)

    def head(self, *args, **kwargs):
        return bamboo.head(self, *args, **kwargs)

    def __getattr__(self, *args, **kwargs):

        name, pargs = args[0], args[1:]

        if name in _bamboo_methods:
            return _create_bamboo_wrapper(self, _bamboo_methods[name])

        if name in _plot_methods:
            return _create_bamboo_wrapper(self, _plot_methods[name])

        if name in _groups_methods:
            return _create_bamboo_wrapper(self, _groups_methods[name])

        return _create_bamboo_wrapper(self, super(BambooSeriesGroupBy, self).__getattribute__(*args, **kwargs))

