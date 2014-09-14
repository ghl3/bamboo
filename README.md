bamboo
======

Bamboo is a library that provides a set of tools intended to be used with Pandas.  It's primary goal is to make it easier and more intuitive to handle and visualize data.  Its design has an emphasis on immutable and composable functions that can be chained together to form more complicated data transformations.


Examples
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


A common practice is to create overlapping plots of data broken up by a class.  This is easy with bamboo.

    hist(df.groupby('group')['feature1'])


Bamboo functions are meant to be composable.  So, let's say that I want to group by the group, filter out groups with a small mean of feature1, and then sort the reamining groups by their mean of feature 2.  One can simply do:

    sorted_groups(filter_groups(df.groupby('group'), lambda x: x['feature1'].mean() > 0), lambda x: x['feature2'].mean())

While this works, composing functions like this can make them become less and less readable (and makes it harder to write).  This is a familiar issue with lisp languages.

Fortunately, bamboo exposes a solution to this.  Bamboo uses subclasses of common Pandas classes that have many of bamboo's helper functions available as methods.  One can create these classes by wrapping a Pandas object with the 'wrap' function:

    from bamboo import wrap

    wrap(df) \
        .groupby('group') \
        .filter_groups(lambda x: x['feature1'].mean() > 0) \
        .sorted_groups(lambda x: x['feature2'].mean())

 Notice that the result of each transformation is automatically passed to the next transformation.  This allows one to do more complicated transformations by chaining methods in succession:


    wrap(df).groupby('group') \
        .filter_groups(lambda x: x['group'].mean() > 0) \
        .sorted_groups(lambda x: x['feature2'].mean()) \
        .map_groups(lambda x: x['feature1'].mean()) \
        .sum() \
        .hist()

Let's describe what's going here for the sake of completeness.  We start by taking a normal Pandas DataFrame and convert it into a BambooDataFrame using 'wrap'.  This allows us to do in line processing on it.  We group it by the 'group' column, turning it into a DataFrameGroupBy.  Next, we apply a filter that requires that all remaining groups have a mean of their group column greater than 0.  We then sort the groups, ordering them by the mean of their 'feature2' column.  Then, we map a function over each row in each group, taking the mean of 'feature2'.  This turns the underlying object from a multi-column data frame into a data frame of only one column (the result of our mapping function).  Finally, we take the sum of that value within each group.  We end by making a histogram of the resulting object, which has a single value for each group.