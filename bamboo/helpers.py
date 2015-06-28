
import pandas as pd


NUMERIC_TYPES = ('bool_', 'int_', 'intc', 'intp', 'int8',
                 'int16', 'int32', 'int64', 'uint8',
                 'uint16', 'uint32', 'uint64', 'float_',
                 'float16', 'float32', 'float64')


def combine_data_frames(frames):
    """
    A helper function to combine multiple
    compatable data frames into a single
    data frame (creating a fresh copy, of course)
    """
    if len(frames) == 0:
        return pd.DataFrame()
    elif len(frames) == 1:
        return frames[0]
    else:
        first = frames[0]
        return first.append(list(frames[1:]))


def get_nominal_integer_dict(nominal_vals):
    """
    Takes a set of nominal values (non-numeric)
    and returns a dictionary that maps each value
    to a unique integer
    """
    d = {}
    for val in nominal_vals:
        if val not in d:
            current_max = max(d.values()) if len(d) > 0 else -1
            d[val] = current_max + 1
    return d


def convert_nominal_to_int(srs, return_dict=False):
    """
    Convert a series of nominal values
    into a series of integers
    """
    d = get_nominal_integer_dict(srs)
    result = srs.map(lambda x: d[x])
    if return_dict:
        return (result, d)
    else:
        return result
