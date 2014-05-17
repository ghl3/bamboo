

def groupmap(grouped, func):
    """
    Take a DataFrameGroupBy and apply a function
    to the DataFrame, returning a seriesgroupby
    of the values
    """
    # grouped.index.name
    transformed = grouped.obj.apply(func, axis=1)
    return transformed.groupby(grouped.grouper)

