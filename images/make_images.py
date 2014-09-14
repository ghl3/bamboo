#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bamboo import wrap


def create_df():
    group = [0, 0,
             1, 1,
             2, 2]

    feature1 = [10.0, 10.0,
                25.0, 25.0,
                -15.0, -25.0]

    feature2 = [100, 200,
                150, 250,
                0, 20]

    df = pd.DataFrame({'group':group,
                       'feature1':feature1,
                       'feature2':feature2})
    return df



def main():

    df = create_df()

    fig = plt.figure(figsize=(12,8))
    wrap(df) \
        .groupby('group') \
        .feature1 \
        .hist(ax=plt.gca(), bins=np.arange(-50, 60, 10), alpha=0.5)
    fig.savefig('image1.png')


    fig = plt.figure(figsize=(12,8))
    wrap(df) \
        .groupby('group') \
        .map_groups(lambda x: x.feature1 + x.feature1, name='mean') \
        .hist(ax=plt.gca(), bins=np.arange(-100, 100, 10), alpha=0.5)
    fig.savefig('image2.png')


if __name__ == '__main__':
    main()
