
import numpy as np
import pandas

from bamboo import frames

from helpers import *

from numpy.testing import assert_array_equal

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
    frames.exclude(df, [0])


def test_partition():
    frames.partition(df, lambda x: x['feature1'] > 1)


def test_split():
    frames.split(df, ['feature1'])


def test_take():
    frames.take(df, ['feature1'])


def test_apply_all():

    def sum(x):
        return x.sum()

    print frames.apply_all(df, sum)


def test_sort_rows():

    df = create_test_df_v2()
    sorted = frames.sort_rows(df, lambda x: x.mean())
    assert_array_equal(sorted.index, [6,8, 9, 2, 0, 1, 3, 4, 5, 10, 11, 7])


def test_sort_columns():

    df = create_test_df_v2()
    frames.sort_columns(df, lambda x: x.mean())


def test_map_functions():

    df = create_test_df_v2()
    frames.map_functions(df, [lambda x: x.feature1, lambda x: x.feature1 + x.feature2])


def test_with_new_columns():

    df = create_test_df_v2()
    frames.with_new_columns(df, [lambda x: x.feature1 + x.feature2])

def test_get_numeric_features():

    df = create_test_df_v3()
    frames.get_numeric_features(df)


def test_convert_nonminals_to_int():

    df = create_test_df_v3()
    frames.convert_nominals_to_int(df)


def test_get_index_rows():

    df = create_test_df_v2()
    frames.get_index_rows(df, [0, 2, 4])


def test_group_by_binning():

    df = create_test_df_v2()
    frames.group_by_binning(df, 'feature2', np.arange(0, 20, 10))
