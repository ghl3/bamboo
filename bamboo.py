


def threading(df, *args):
    """
    A function that mimics Clojure's threading macro
    Apply a series of transformations to the input
    DataFrame and return the fully transformed
    DataFrame at the end
    """
    if len(args) > 0 and args[0] != ():
        first_func = args[0]
        return threading(first_func(df), *args[1:])
    else:
        return df
