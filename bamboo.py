

"""
Miscellaneous functions acting on various objects
"""


import pandas as pd
from pandas.core.GroupBy import GroupBy

def head(frame, n=5):
    """
    A more powerful version of pandas 'head'
    that works nicely with DataFrameGroupBy objects
    """

    if isinstance(frame, GroupBy):
        return gropuped.apply(lambda x: x.head(n))
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


def combine_data_frames(frames):
    """
    A helper function to combine multiple
    compatable data frames into a single
    data frame (creating a fresh copy, of course)
    """
    if len(frames)==0:
        return pd.DataFrame()
    elif len(frames)==1:
        return frames[0]
    else:
        first = frames[0]
        return first.append(list(frames[1:]))


NUMERIC_TYPES = ('bool_', 'int_', 'intc', 'intp', 'int8',
                 'int16', 'int32', 'int64', 'uint8',
                 'uint16', 'uint32', 'uint64', 'float_',
                 'float16', 'float32', 'float64')

