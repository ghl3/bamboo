

import pandas as pd
from pandas.core.groupby import GroupBy

from inspect import getmembers, isfunction

from groups import take_groups


"""
Miscellaneous functions acting on various objects
"""


def head(frame, n=5, ngroups=5):
    """
    A more powerful version of pandas 'head'
    that works nicely with DataFrameGroupBy objects
    """

    if isinstance(frame, GroupBy):
        return take_groups(frame, ngroups).apply(lambda x: x.head(n))
    else:
        return frame.head(n)


def threading(df, *args):
    """
    A function that mimics Clojure's threading macro
    Apply a series of transformations to the input
    DataFrame and return the fully transformed
    DataFrame at the end
    """
    if len(args) > 0 and args[0] != ():
        first_func = args[0]
        return threading(first_func(df), *args[1:])
    else:
        return df


_bamboo_methods = {name:func  for name, func in getmembers(bamboo.bamboo) if isfunction(func)}
_frames_methods = {name:func  for name, func in getmembers(bamboo.frames) if isfunction(func)}
_groups_methods = {name:func  for name, func in getmembers(bamboo.groups) if isfunction(func)}



class Bamboo(object):
    """
    A class designed to wrap a Pandas object and dispatch
    bamboo functions (when available).  Else, it falls
    back to the object's original methods
    """

    def __init__(self, obj):
        self.obj = obj
        self._is_data_frame = isinstance(self.obj, pandas.DataFrame)
        self._is_group_by = isinstance(self.obj, pandas.core.groupby.GroupBy)


    def create_callable(self, func):
        def callable(*args, **kwargs):
            return func(self.obj, *args, **kwargs)
        return callable

    def __getattr__(self, *args, **kwargs):

        name, args = args[0], args[1:]

        isinstance(self.obj, pandas.DataFrame)

        if (self._is_data_frame or self._is_group_by ) and name in _bamboo_methods:
            return self.create_callable(_bamboo_methods[name])
            #return lambda : _bamboo_methods[name](self.obj, *args, **kwargs)

        if self._is_data_frame and name in _frames_methods:            
            return self.create_callable(_frames_methods[name])

            #print "Found frames func"
            #func = _frames_methods[name]
            #def callable(*args, **kwargs):
            #    return func(self.obj, *args, **kwargs)
            #
            #return callable
            #return lambda: _frames_methods[name](self.obj, *args, **kwargs)

        if self._is_group_by and name in _groups_methods:
            return self.create_callable(_groups_methods[name])
            #return lambda: _groups_methods[name](self.obj, *args, **kwargs)

        return self.obj.__getattribute__(*args, **kwargs)


# class Bamboo(object):
#     """
#     A class designed to wrap a Pandas object and dispatch
#     bamboo functions (when available).  Else, it falls
#     back to the object's original methods
#     """

#     def __init__(self, obj):
#         self.obj = obj

#     def foo(self):
#         print "Foo"

#     def __getattr__(self, *args, **kwargs):

#         name = args[0]

#         if isinstance(obj, pandas.DataFrame) and name in _frames_methods:
#             return _frames_methods[name]

#         if isinstance(obj, pandas.GroupBy) and name in _groupes_methods:
#             return _groupes_methods[name]

#         return self.obj.__getattribute__(*args, **kwargs)


# #        if name in ['head']:
# #            return lambda : head(self.obj)
# #        else:

