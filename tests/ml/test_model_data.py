
import numpy as np

import pandas
from nose.tools import *
from bamboo.core import *
from bamboo.helpers import *
import bamboo.ml

from pandas.util.testing import assert_frame_equal

from numpy.random import RandomState

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

group = [0, 0, 0, 0,
         1, 1, 1, 1,
         0, 0, 0, 1]

feature1 = [1, 1, 1, 1,
            2, 2, 3, 4,
            1, 2, 3, 4]

feature2 = [10.0, 10.5, 9.5, 11.0,
            20.0, 20.0, 35.0, -10.0,
            0.0, 200.0, 150.0, -30.0]


feature3 = ['A', 'A', 'B', 'C',
            'C', 'B', 'D', 'B',
            'A', 'B', 'C', 'D']

df = pandas.DataFrame({'group':group,
                       'feature1':feature1,
                       'feature2':feature2})

df2 = pandas.DataFrame({'group':group,
                        'feature1':feature1,
                        'feature2':feature2,
                        'feature3':feature3})


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


def test_balance_sample():

    data = bamboo.ml.ModelingData.from_dataframe(df, target='group')

    eq_(len(data), 12)

    value_counts = data.targets.value_counts()
    eq_(value_counts[0], 7)
    eq_(value_counts[1], 5)

    # Now, balance the data
    np.random.seed(42)
    balanced = data._balance_by_sample_with_replace(size=20)

    eq_(len(balanced), 20)

    value_counts = balanced.targets.value_counts()
    eq_(value_counts[0], 10)
    eq_(value_counts[1], 10)


def test_orthogonal():

    dataA = bamboo.ml.ModelingData.from_dataframe(df.iloc[0:6], target='group')
    dataB = bamboo.ml.ModelingData.from_dataframe(df.iloc[3:9], target='group')
    dataC = bamboo.ml.ModelingData.from_dataframe(df.iloc[6:12], target='group')

    assert(dataA.is_orthogonal(dataC))
    assert(not dataA.is_orthogonal(dataB))
    assert(not dataB.is_orthogonal(dataC))


def test_numeric_features():

    data = bamboo.ml.ModelingData.from_dataframe(df2, target='group')

    eq_(data.shape(), (12,4))
    assert('feature3' in data.features.columns)

    numeric_data = data.numeric_features()

    eq_(numeric_data.shape(), (12,3))
    assert('feature3' not in numeric_data.features.columns)


def test_probas():

    clf = LogisticRegression()

    data = bamboo.ml.ModelingData.from_dataframe(df, target='group')

    data.fit(clf)

    probas = data.predict_proba(clf)

    eq_(dict(probas.irow(0)), {'index': 0, 'proba_0': 0.64009602726273496, 'proba_1': 0.35990397273726504, 'target': 0})


def test_predict():

    reg = LinearRegression()

    data = bamboo.ml.ModelingData.from_dataframe(df, target='group')

    data.fit(reg)

    predictions = data.predict(reg)

    eq_(dict(predictions.irow(0)), {'predict': 5.8180999081003382e-16, 'index': 0, 'target': 0})
    eq_(dict(predictions.irow(-1)), {'predict': 1.0000000000000016, 'index': 11, 'target': 1})


def test_summary():

    clf = LogisticRegression()

    data = bamboo.ml.ModelingData.from_dataframe(df, target='group')
    data.fit(clf)

    probas = data.predict_proba(clf)
    summary = bamboo.ml.ModelingData.get_threshold_summary(probas, 1)

    eq_(dict(summary), {'sensiticity': 0.5, 'false_positives': 0, 'f1': 0.6666666666666666, 'precision': 1.0, 'false_negatives': 5, 'true_positive_rate': 0.0, 'specificity': 1.0, 'threshold': 0.5, 'target': 1, 'true_negatives': 0, 'recall': 0.5, 'false_positive_rate': 0.4166666666666667, 'true_positives': 5, 'accuracy': 0.4166666666666667})


def test_classifier_performance_summary():

    clf = LogisticRegression()

    data = bamboo.ml.ModelingData.from_dataframe(df, target='group')
    data.fit(clf)

    summary = data.get_classifier_performance_summary(clf, 0, thresholds=np.arange(0.0, 1.0, 0.1))

    eq_(dict(summary.irow(0)), {'f1': 0.53846153846153844, 'target': 0.0, 'sensiticity': 0.5, 'recall': 0.5, 'false_positive_rate': 0.58333333333333337, 'false_positives': 5.0, 'precision': 0.58333333333333337, 'true_positives': 7.0, 'false_negatives': 7.0, 'true_positive_rate': 0.41666666666666669, 'specificity': 0.5, 'threshold': 0.0, 'true_negatives': 5.0, 'accuracy': 1.0})

