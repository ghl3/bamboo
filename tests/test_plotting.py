
import pandas
import bamboo.groups
import bamboo.frames

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

def test_boxplot():
    bamboo.frames.boxplot(df, 'feature1', 'feature2')

def test_hexbin():
    bamboo.frames.hexbin(df, 'feature1', 'feature2')

def test_hist_df():
    bamboo.groups.hist(df.groupby('group'))

def test_hist_var():
    bamboo.groups.hist(df.groupby('group'), 'feature1')


