
from bamboo.groups import *

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


def test_filter_groups():
    """
    Return only groups with size > 3
    """
    filter_groups(dfgb, lambda x: len(x) > 3)


def test_sort():
    """
    Sort the order of the groups
    by the size of the group
    """
    sort(dfgb, lambda x: len(x))
