

from helpers import *
from bamboo.core import wrap

import numpy as np


def test_wrap():

    df = create_test_df_v2()

    return wrap(df).groupby('group') \
        .filter_groups(lambda x: x['group'].mean() > 0) \
        .sorted_groups(lambda x: x['feature2'].mean()) \
        .map_groups(lambda x: x['feature1'].mean()) \
        .sum() \
        .hist()


def test_filter_groups():

    df = create_test_df_v2()

    return wrap(df)\
        .groupby('group')\
        .filter_groups(lambda x: x in (0, 1), on_index=True)\
        .feature1\
        .value_counts()


def test_filter_groups_hist():

    df = create_test_df_v2()

    return wrap(df)\
        .groupby('group')\
        .filter_groups(lambda x: x in (0, 1, 2), on_index=True)\
        .feature1\
        .hist(alpha=0.5, normed=True, bins=np.arange(0, 20, 2))


def test_sorted_groups():

    df = create_test_df_v2()

    return wrap(df)\
        .groupby(df['group'])\
        .sorted_groups(lambda x: x.feature2.mean())\
        .feature1.mean()\
                 .head()


def test_map_groups():

    df = create_test_df_v2() 

    return wrap(df)\
        .groupby(df['group'])\
        .map_groups(lambda x: x.feature1 - x.feature2, name='feature1_minus_feature2')\
        .mean()\
        .sort('feature1_minus_feature2')


def test_apply_groups():

    df = create_test_df_v2()

    def thing(x):
        return x.feature1 + x.feature2

    return wrap(df)\
        .groupby('group')\
        .apply_groups(thing, lambda x: x.feature1 - x.feature2)\
        .mean()\
        .scatter('thing', '<lambda>_1')
