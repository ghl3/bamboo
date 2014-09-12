

import pandas as pd

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

