

import pandas as pd
from pandas.core.groupby import GroupBy

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



class Bamboo(Object):

    def __init__(self, obj):
        self.obj = obj

    

