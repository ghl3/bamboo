
import matplotlib.pyplot as plt

import pandas

from pandas.util.testing import assert_frame_equal, assert_panelnd_equal

from bamboo.core import head

from functools import wraps
from nose.plugins.attrib import attr

def assert_equals(obj1, obj2):

    print "Object of type {}:\n {}\nShould equal Object of type {}:\n{}\n".format(
        type(obj1), head(obj1), type(obj2), head(obj2))

    assert(type(obj1)==type(obj2))

    if isinstance(obj1, pandas.DataFrame):
        assert_frame_equal(obj1, obj2)
    elif isinstance(obj1, pandas.core.groupby.GroupBy):
        assert_groupby_equal(obj1, obj2)
    else:
        assert(obj1==obj2)


def assert_groupby_equal(groupby, test, **kwargs):
    for ((keyA, groupA), (keyB, groupB)) in zip(groupby, test):
        assert(keyA==keyB)
        assert_frame_equal(groupA, groupB, **kwargs)


def create_test_df():
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

    return df


def create_test_df_v2():
    group = [0, 0, 0, 0,
             1, 1,
             0, 1,
             3, 3,
             4, 4]

    feature1 = [1, 1, 1, 1,
                2, 2,
                3, 4,
                10, 10,
                12, 18]

    feature2 = [10.0, 10.5, 9.5, 11.0,
                20.0, 20.0,
                0.0, 200.0,
                -10.0, -5.0,
                10.0, 20.0]

    df = pandas.DataFrame({'group':group,
                           'feature1':feature1,
                           'feature2':feature2})
    return df


def create_test_df_v3():
    group = ['A', 'A', 'A',
             'B', 'B', 'B']

    feature1 = [1, 1, 1,
                2, 2, 2]

    feature2 = [10.0, 10.5, 9.5,
                11.0, 20.0, 20.0]

    df = pandas.DataFrame({'group':group,
                           'feature1':feature1,
                           'feature2':feature2})
    return df


def plotting(test):

    @wraps(test)
    def test_wrapper(*args, **kwargs):
        plt.clf()
        res = test(*args, **kwargs)

        file_name = 'tests/images/{}_{}.png'.format(test.__module__, test.__name__)
        plt.savefig(file_name)
        return res

    return test_wrapper
