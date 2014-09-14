
import numpy as np
import pandas

from bamboo.frames import *

from helpers import *


group = [0, 0, 0, 0,
         1, 1,
         0, 1]

feature1 = [1, 1, 1, 1,
            2, 2,
            3, 4]

feature2 = [10.0, 10.5, 9.5, 11.0,
            20.0, 20.0,
            0.0, 200.0]

df = pandas.DataFrame({'group':group,
                       'feature1':feature1,
                       'feature2':feature2})

def test_exclude():
    exclude(df, [0])


def test_partition():
    partition(df, lambda x: x['feature1'] > 1)


def test_split():
    split(df, ['feature1'])


def test_take():
    take(df, ['feature1'])


def test_apply_all():

    def sum(x):
        return x.sum()

    print apply_all(df, sum)


def test_sort_rows():

    df = create_test_df_v2()
    sort_rows(df, lambda x: x.mean())


def test_sort_columns():

    df = create_test_df_v2()
    sort_columns(df, lambda x: x.mean())


def test_map_functions():

    df = create_test_df_v2()
    map_functions(df, [lambda x: x.feature1, lambda x: x.feature1 + x.feature2])


def test_with_new_columns():

    df = create_test_df_v2()
    with_new_columns(df, [lambda x: x.feature1 + x.feature2])

def test_get_numeric_features():

    df = create_test_df_v3()
    get_numeric_features(df)


def test_convert_nonminals_to_int():

    df = create_test_df_v3()
    convert_nominals_to_int(df)


def test_get_index_rows():

    df = create_test_df_v2()
    get_index_rows(df, [0, 2, 4])


def test_group_by_binning():

    df = create_test_df_v2()
    group_by_binning(df, 'feature2', np.arange(0, 20, 10))
