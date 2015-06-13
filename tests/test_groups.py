
from nose.tools import *
from bamboo.groups import *

from helpers import *

from pandas.util.testing import assert_frame_equal, assert_panelnd_equal, assert_series_equal


def test_filter_groups():
    """
    Return only groups with size > 3
    """

    dfgb = create_test_df().groupby('group')

    filtered = filter_groups(dfgb, lambda x: len(x) > 3)

    should_be = pandas.DataFrame({
        'group': [0, 0, 0, 0, 0],
        'feature1' : [1, 1, 1, 1, 3],
        'feature2' : [10.0, 10.5, 9.5, 11.0, 0.0]},
                                 index=[0, 1, 2, 3, 6]).groupby('group')

    assert_equals(filtered, should_be)


def test_take_groups():
    """
    Get the first group
    """

    dfgb = create_test_df().groupby('group')

    filtered = take_groups(dfgb, 1)

    should_be = pandas.DataFrame({
        'group': [0, 0, 0, 0, 0],
        'feature1' : [1, 1, 1, 1, 3],
        'feature2' : [10.0, 10.5, 9.5, 11.0, 0.0]},
                                 index=[0, 1, 2, 3, 6]).groupby('group')

    assert_equals(filtered, should_be)


def test_map_groups():
    """
    Map a function onto a DataFrameGroupBy
    """

    dfgb = create_test_df().groupby('group')

    mapped = map_groups(dfgb, lambda x: x.feature1 + x.feature2, name='sum')

    should_be = pandas.Series([11.0, 11.5, 10.5, 12.0, 22.0, 22.0, 3.0, 204.0], name='sum')\
                      .groupby([0, 0, 0, 0, 1, 1, 0, 1])

    assert_series_equal(mapped.obj, should_be.obj)


def test_sort_v2():
    """
    Sort the order of the groups
    by the size of the group
    """
    dfgb = create_test_df_v2().groupby('group')

    dfgb_sorted = sorted_groups(dfgb, lambda x: x['feature2'].mean())

    should_be = pandas.DataFrame({
        'group': [3, 3, 0, 0, 0, 0, 0, 4, 4, 1, 1, 1],
        'feature1' : [10, 10, 1, 1, 1, 1, 3, 12, 18, 2, 2, 4],
        'feature2' : [-10.0, -5.0,10.0, 10.5, 9.5, 11.0, 0.0,10.0, 20.0, 20.0, 20.0, 200]},
                                 index=[8, 9, 0, 1, 2, 3, 6, 10, 11,4, 5, 7]).groupby('group', sort=False)

    assert_equals(dfgb_sorted, should_be)

    group_order = [key for key, group in dfgb_sorted]
    eq_(group_order, [3, 0, 4, 1])




@plotting
def test_hist_functions():
    dfgb = create_test_df_v2().groupby('group')
    hist_functions(dfgb, lambda x: x.feature1+x.feature2, lambda x: x.feature1-x.feature2)


@plotting
def test_scatter():

    dfgb = create_test_df_v2().groupby('group')
    scatter(dfgb, 'feature1', 'feature2')

@plotting
def test_stacked_counts_plot():

    dfgb = create_test_df_v2().groupby('group')
    stacked_counts_plot(dfgb, 'feature1')

@plotting
def test_stacked_counts_plot_ratio():

    dfgb = create_test_df_v2().groupby('group')
    stacked_counts_plot(dfgb, 'feature1', ratio=True)


def test_pivot_groups():
    dfgb = create_test_df_v2().groupby(['group', 'feature1'])
    pivot_groups(dfgb, lambda x: x.feature2)


def test_pivot_groups_2():
    df = pandas.DataFrame({'group1': ['A', 'B', 'C'], 'group2': ['X', 'Y', 'Z'], 'val': [1, 2, 3]}).groupby(['group1', 'group2'])
    pivoted = pivot_groups(df, lambda x: x.val*2)

    idx = pd.Series(['A', 'B', 'C'], name='group1')

    assert_series_equal(pivoted['X'], pd.Series([2, None, None], index=idx, name='X'))
    assert_series_equal(pivoted['Y'], pd.Series([None, 4, None], index=idx, name='Y'))
    assert_series_equal(pivoted['Z'], pd.Series([None, None, 6], index=idx, name='Z'))

    #should_be = pandas.DataFrame({
    #    'group1': [3, 3, 0, 0, 0, 0, 0, 4, 4, 1, 1, 1],
    #    'group2' : [10, 10, 1, 1, 1, 1, 3, 12, 18, 2, 2, 4],
    #    'feature2' : [-10.0, -5.0,10.0, 10.5, 9.5, 11.0, 0.0,10.0, 20.0, 20.0, 20.0, 200]},
    #                             index=[8, 9, 0, 1, 2, 3, 6, 10, 11,4, 5, 7]).groupby('group', sort=False)

    #assert_equals(dfgb_sorted, should_be)

