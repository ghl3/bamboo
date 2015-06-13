
import pandas
import bamboo.groups
import bamboo.frames

from helpers import *

import bamboo


def create_df():

    group = ['A', 'A', 'A', 'A',
             'B', 'B',
             'C']

    feature1 = [1, 1, 1, 1,
                2, 2,
                3]

    feature2 = [10.0, 10.5, 9.5, 11.0,
                20.0, 20.0,
                0.0]

    feature3 = ['Red', 'Red', 'Green', 'Yellow',
                'Red', 'Green', 'Yellow']

    return pandas.DataFrame({'group':group,
                             'feature1':feature1,
                             'feature2':feature2,
                             'feature3':feature3})


def test_standard_plot():

    df = create_df()

    fig = plt.figure(figsize=(12,8))

    plt.subplot(2, 2, 1)
    bamboo.hist(df.groupby('group')['feature1'])

    plt.subplot(2, 2, 2)
    bamboo.hist(df.groupby('group'), 'feature2')

    plt.subplot(2, 2, 3)
    bamboo.hist(df.groupby('group')['feature3'])

    plt.subplot(2, 2, 4)
    bamboo.scatter(df.groupby('group'), 'feature1', 'feature2')

    plt.savefig("tests/images/standard.pdf")

