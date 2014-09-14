
import pandas
from nose.tools import *
from bamboo.core import *
from bamboo.helpers import *

from pandas.util.testing import assert_frame_equal

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

dfgb = df.groupby('group')


def test_combine_data_frames():
    dfA = pandas.DataFrame({'val' : [1, 4, 9]},
                           index=[1, 2, 3])

    dfB = pandas.DataFrame({'val' : [49, 64]},
                           index=[7, 8])

    combined = combine_data_frames([dfA, dfB])

    should_be = pandas.DataFrame({'val' : [1, 4, 9, 49, 64]},
                                 index=[1, 2, 3, 7, 8])

    print "Combined: "
    print combined.head()
    print "Should be: "
    print should_be.head()

    assert_frame_equal(combined, should_be)


def test_head_groups():
    head(dfgb, n=5, ngroups=5)
