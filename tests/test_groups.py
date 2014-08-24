
from nose.tools import *
from bamboo.groups import *

from helpers import *

from pandas.util.testing import assert_frame_equal, assert_panelnd_equal


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
    Return only groups with size > 3
    """

    dfgb = create_test_df().groupby('group')

    filtered = take_groups(dfgb, 1)

    should_be = pandas.DataFrame({
        'group': [0, 0, 0, 0, 0],
        'feature1' : [1, 1, 1, 1, 3],
        'feature2' : [10.0, 10.5, 9.5, 11.0, 0.0]},
                                 index=[0, 1, 2, 3, 6]).groupby('group')

    assert_equals(filtered, should_be)


def test_sort():
    """
    Sort the order of the groups
    by the size of the group
    """
    dfgb = create_test_df().groupby('group')

    sort(dfgb, lambda x: len(x))
