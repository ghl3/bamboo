
import matplotlib.pyplot as plt

from subplots import PdfSubplots

import bamboo.plotting

def add_title_page_to_pdf(pdf, title, figsize=(30, 20),
                     end_page=True):
    fig = plt.figure(figsize=figsize)
    plt.axis('off')
    plt.text(0.5, 0.5, title, ha='center', va='center',
             size=48)

    if end_page:
        pdf.savefig()

def add_subplots_to_pdf(pdf, dfgb, plot_func,
                        var_func=None,
                        nrows=3, ncols=3, figsize=(30, 20),
                        end_page=True,
                        *args, **kwargs):
    """
    Take a grouped dataframe and save a pdf of
    the histogrammed variables in that dataframe.
    TODO: Can we abstract this behavior...?
    """

    subplots = PdfSubplots(pdf, nrows, ncols, figsize=figsize)

    for (var, series) in dfgb._iterate_column_groupbys():

        subplots.next_subplot()
        try:
            plot_func(series, *args, **kwargs)
            if var_func:
                var_func(var)
            plt.xlabel(var)
            subplots.end_iteration()
        except:
            subplots.skip_subplot()

    if end_page:
        subplots.finalize()


def add_table_to_pdf(pdf, table, figsize=(30, 20),
                     end_page=True,
                     *args, **kwargs):

    fig = plt.figure(figsize=figsize)

    bamboo.plotting.plot_table(table, *args, **kwargs)
    plt.axis('off')

    if end_page:
        pdf.savefig(fig)
