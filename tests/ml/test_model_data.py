
import pandas
from nose.tools import *
from bamboo.core import *
from bamboo.helpers import *
import bamboo.ml

from pandas.util.testing import assert_frame_equal

from numpy.random import RandomState

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


def test_split():

    random_state = RandomState(12345)

    data = bamboo.ml.ModelingData.from_dataframe(df, target='group')

    train, test = data.train_test_split(random_state=random_state)

    print train, test

    eq_(train.shape(), (9,3))
    eq_(test.shape(), (3,3))

    assert(train.is_orthogonal(test))
    

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



def test_orthogonal():

    dataA = bamboo.ml.ModelingData.from_dataframe(df.iloc[0:6], target='group')
    dataB = bamboo.ml.ModelingData.from_dataframe(df.iloc[3:9], target='group')
    dataC = bamboo.ml.ModelingData.from_dataframe(df.iloc[6:12], target='group')

    assert(dataA.is_orthogonal(dataC))
    assert(not dataA.is_orthogonal(dataB))
    assert(not dataB.is_orthogonal(dataC))
