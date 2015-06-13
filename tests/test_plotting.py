
import numpy as np

import pandas
import bamboo.groups
import bamboo.frames
import bamboo.addons

from helpers import *

from bamboo import hist

group = ['A', 'A', 'A', 'A',
         'B', 'B',
         'C']

feature1 = [1, 1, 1, 1,
            2, 2,
            3]

feature2 = [10.0, 10.5, 9.5, 11.0,
            20.0, 20.0,
            0.0]

df = pandas.DataFrame({'group':group,
                       'feature1':feature1,
                       'feature2':feature2})

@plotting
def test_boxplot():
    bamboo.frames.boxplot(df, 'feature1', 'feature2')

@plotting
def test_hexbin():
    try:
        from sklearn.datasets import make_blobs
    except:
        assert(False)

    X1, Y1 = make_blobs(n_features=2, centers=3, n_samples=10000)
    df = pandas.DataFrame(X1, columns=['x', 'y'])
    bamboo.frames.hexbin(df, 'x', 'y')

@plotting
def test_hist_df():
    bamboo.core.hist(df.groupby('group'))

@plotting
def test_hist_var():
    bamboo.core.hist(df.groupby('group'), 'feature1')

@plotting
def test_hist():
    dfgb = create_test_df_v3().groupby('group')
    hist(dfgb['feature1'])
    #hist(dfgb['feature2'])


@plotting
def test_summary_table():
    df = pandas.DataFrame({'group': ['GOOD', 'GOOD', 'GOOD', 'GOOD', 'BAD', 'BAD', 'BAD'],
                           'x': [1, 2, 1, 3.4, 2, 5.6, 3],
                           'y':[10, 50, 10, 20, 20, 40, -10]})
    bamboo.hist(df.groupby('group'), 'x',
                addons=[bamboo.addons.summary_table],
                bins=np.arange(0, 5, 1),
                alpha=0.5)

