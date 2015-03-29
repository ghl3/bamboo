
import numpy as np
import matplotlib.pyplot as plt

from itertools import cycle, islice

import pandas as pd

from sklearn.datasets import make_classification


def take(iterable, n):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

# Tests to run the documentation (README) and to create the
# corresponding images

def create_dataset(n_rows=1000):

    features, classes= make_classification(n_rows, n_features=4, n_informative=2, n_classes=2, random_state=1)

    df = pd.DataFrame(features, columns=['feature1', 'feature2', 'feature3', 'feature4'])
    df['class'] = classes
    df['feature5'] = take(cycle(['A', 'B', 'C', 'D']), n_rows)

    return df



def test_hist_float():

    df = create_dataset()

    plt.clf()

    from bamboo import hist
    hist(df.groupby('class').feature1,
         ax=plt.gca(), bins=np.arange(-5, 5, 0.5), alpha=0.5)

    plt.savefig('tests/images/readme_hist_float.png')


def test_hist_nominal():

    df = create_dataset()

    plt.clf()

    from bamboo import hist
    hist(df.groupby('class').feature5,
         ax=plt.gca(), alpha=0.5)

    plt.savefig('tests/images/readme_hist_nominal.png')


def test_hist_all():

    df = create_dataset()

    plt.clf()
    plt.figure()

    from bamboo import hist
    hist(df.groupby('class'), alpha=0.5, autobin=True)

    plt.savefig('tests/images/readme_hist_all.png')


def test_scatter():

    df = create_dataset()

    plt.clf()

    from bamboo import scatter
    scatter(df.groupby('class'), 'feature1', 'feature2', alpha=0.5)

    plt.savefig('tests/images/readme_scatter.png')




def test_manipulation():

    df = create_dataset()

    plt.clf()

    from bamboo import wrap

    wrap(df) \
        .groupby('class') \
        .map_groups(lambda x: x.feature1 + x.feature1, name='feature_sum') \
        .hist(ax=plt.gca(), bins=np.arange(-5, 5, 0.5), alpha=0.5)

    plt.savefig('tests/images/readme_manipulation_hist.png')

