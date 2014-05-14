from __future__ import division

import pandas as pd

from plotting import NUMERIC_TYPES



def get_numeric_features(df):
    float_feature_names = [feature for feature in df.columns
                           if df[feature].dtype in NUMERIC_TYPES]
    return df[float_feature_names]


def arff_to_df(arff):
    rows = []
    for row in arff[0]:
        rows.append(list(row))
    attributes = [x for x in arff[1]]
    return pd.DataFrame(rows, columns=attributes)


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
            d[val] = current_max+1
    return d


def convert_to_integer(srs, return_dict=False):
    """
    Convert a series of nominal values
    into a series of integers
    """
    d = get_nominal_integer_dict(srs)
    result =  srs.map(lambda x: d[x])
    if return_dict:
        return (result, d)
    else:
        return result


def convert_strings_to_integer(df):
    ret = pd.DataFrame()
    for column_name in df:
        column = df[column_name]
        if column.dtype not in NUMERIC_TYPES:
            ret[column_name] = convert_to_integer(column)
        else:
            ret[column_name] = column
    return ret


def get_index_rows(srs, indices):
    """
    Given a dataframe and a list of indices,
    return a list of row indices corresponding
    to the supplied indices (maintaining order)
    """
    rows = []
    for i, (index, row) in enumerate(srs.iteritems()):
        if index in indices:
            rows.append(i)
    return rows

