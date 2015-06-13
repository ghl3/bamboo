from __future__ import division

import matplotlib.pyplot as plt
import pandas as pd


def summary_table(type, series_map, bins=None):
    """
    Draw a table describing the number of
    elements of each group for either nominal
    or floating point valued groups
    """

    if type == 'FLOAT':
        return _summary_table_float(series_map, bins)
    elif type == 'NOMINAL':
        return _summary_table_nominal(series_map)
    else:
        raise NotImplementedError()


def _make_table_pretty(table):

    cells = table.properties()['child_artists']
    for cell in cells:
        cell.set_height(cell.get_height()*1.4)

    row_label_cells = [cell for (x, y), cell in table.properties()['celld'].iteritems()
                       if y==-1]

    print row_label_cells
    for cell in row_label_cells:
        cell.set_width(cell.get_width()*5)

    table.auto_set_font_size(False)
    table.set_fontsize(8)


def _summary_table_float(series_map, bins):

    print bins, min(bins), max(bins)

    rows = []
    row_labels = []
    column_labels = ['Total', 'Not Null', '% Shown']
    for group, srs in series_map.iteritems():

        total_num = len(srs)
        not_null = len(srs[pd.notnull(srs)])
        not_shown = len(srs[(pd.isnull(srs)) | (srs > max(bins)) | (srs < min(bins))])

        pct_shown = "{}%".format((total_num - not_shown) / total_num * 100.0)

        row_labels.append(group)
        rows.append([total_num, not_null, pct_shown])

    table = plt.table(cellText=rows,
                      rowLabels=row_labels,
                      colLabels=column_labels,
                      colWidths = [0.08]*3,
                      loc='upper right')

    _make_table_pretty(table)
    plt.show()

    return table
