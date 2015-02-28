
import pandas
from nose.tools import *
from bamboo.core import *
from bamboo.helpers import *
import bamboo.ml

from pandas.util.testing import assert_frame_equal


group = [0, 0, 0, 0,
         1, 1, 1, 1,
         0, 0, 0, 1]

feature1 = [1, 1, 1, 1,
            2, 2, 3, 4,
            1, 2, 3, 4]

feature2 = [10.0, 10.5, 9.5, 11.0,
            20.0, 20.0, 35.0, -10.0,
            0.0, 200.0, 150.0, -30.0]

df = pandas.DataFrame({'group':group,
                       'feature1':feature1,
                       'feature2':feature2})

def test_balance_truncation():

    data = bamboo.ml.ModelingData.from_dataframe(df, target='group')

    eq_(len(data), 12)
    
    value_counts = data.targets.value_counts()
    eq_(value_counts[0], 7)
    eq_(value_counts[1], 5)

    # Now, balance the data
    balanced = data._balance_by_truncation()

    eq_(len(balanced), 10)

    value_counts = balanced.targets.value_counts()
    eq_(value_counts[0], 5)
    eq_(value_counts[1], 5)
