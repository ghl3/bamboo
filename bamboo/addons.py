from __future__ import division

import matplotlib.pyplot as plt
import pandas as pd


def summary_table(type, series_map, bins=None, **kwargs):
    """
    Draw a table describing the number of
    elements of each group for either nominal
    or floating point valued groups
    """

    if type == 'FLOAT':
        return _create_summary_table(series_map, bins, **kwargs)
    elif type == 'NOMINAL':
        return _create_summary_table(series_map, **kwargs)
    else:
        raise NotImplementedError()


def _make_table_pretty(table, fontsize=8, height_factor=1.0, width_factor=1.4, **kwargs):

    cells = table.properties()['child_artists']
    for cell in cells:
        cell.set_height(cell.get_height() * height_factor)
        cell.set_width(cell.get_width() * width_factor)

    row_label_cells = [cell for (x, y), cell in table.properties()['celld'].iteritems()
                       if y == -1]

    for cell in row_label_cells:
        cell.set_width(cell.get_width() * 5)

    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)


def _create_summary_table(series_map, bins=None, **kwargs):

    rows = []
    row_labels = []
    column_labels = ['Total', 'Not Null', '% Shown']
    for group, srs in series_map.iteritems():

        total_num = len(srs)
        not_null = len(srs[pd.notnull(srs)])

        if bins is not None:
            not_shown = len(srs[(pd.isnull(srs)) | (srs > max(bins)) | (srs < min(bins))])
        else:
            not_shown = len(srs[(pd.isnull(srs))])

        pct_shown = "{number:.{digits}f}%".format(number=(total_num - not_shown) / total_num * 100.0, digits=1)

        row_labels.append(group)
        rows.append([total_num, not_null, pct_shown])

    table = plt.table(cellText=rows,
                      rowLabels=row_labels,
                      colLabels=column_labels,
                      colWidths=[0.08] * 3,
                      loc='upper center')

    _make_table_pretty(table, **kwargs)

    return table
