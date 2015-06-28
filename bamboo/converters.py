
import pandas as pd


def arff_to_df(arff):
    rows = []
    for row in arff[0]:
        rows.append(list(row))
    attributes = [x for x in arff[1]]

    # Create the DataFrame
    return pd.DataFrame(rows, columns=attributes)
