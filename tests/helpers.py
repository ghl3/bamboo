
import pandas

from pandas.util.testing import assert_frame_equal, assert_panelnd_equal


def assert_equals(obj1, obj2):

    print "Object of type {}:\n {}\nShould equal Object of type {}:\n{}\n".format(
        type(obj1), obj1, type(obj2), obj2)

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
