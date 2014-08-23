

import pandas

from bamboo.frames import *


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

def test_exclude():
    exclude(df, [0])


def test_partition():
    partition(df, lambda x: x['feature1'] > 1)


def test_split():
    split(df, ['feature1'])


def test_take():
    take(df, ['feature1'])


def test_apply_all():

    def sum(x):
        return x.sum()

    print apply_all(df, sum)
