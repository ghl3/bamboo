

from helpers import *
from bamboo.core import wrap


def test_wrap():

    df = create_test_df_v2()

    wrap(df).groupby('group') \
        .filter_groups(lambda x: x['group'].mean() > 0) \
        .sorted_groups(lambda x: x['feature2'].mean()) \
        .map_groups(lambda x: x['feature1'].mean()) \
        .sum() \
        .hist()
