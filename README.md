bamboo
======

Bamboo is a library that provides a set of tools intended to be used with Pandas.  It's primary goal is to make it easier and more intuitive to handle and visualize data.  Its design has an emphasis on immutable and composable functions that can be chained together to form more complicated data transformations.


Example
-------
  
  
For these examples, assume that we have a Pandas DataFrame structured similarly to the following:
  
| id | class | feature1 | feature2 |
|----|-------|----------|----------|
| 1  | 0     | 10.0     | 100      |
| 2  | 0     | 10.0     | 200      |
| 3  | 1     | 20.0     | 150      |
| 4  | 1     | 25.0     | 250      |
| 5  | 2     | -15.0    | 0        |
| 6  | 2     | -25.0    | 20       |


Much of bamboo's functionality involves manipulating grouped data frames, something that proves to be extremely useful in common data analyses.

Imagine that we want to group our data by the class column but only return those groups that have an average of feature1 > 0:

    filter_groups(df.groupby('group'), lambda x: x['feature1'].mean() > 0)
    
|       |   | group | feature1 | feature2 |
|-------|---|-------|----------|----------|
| **group** |   |       |          |          |
| **0**     |   | 0     | 10       | 100      |
| **0**     |   | 0     | 10.0     | 200      |
| **1**     |   | 1     | 20.0     | 150      |
| **1**     |   | 1     | 25.0     | 250      |


The return value is a DataFrameGroupBy with only two of the three groups remaining (those that satisfy the given predicate).