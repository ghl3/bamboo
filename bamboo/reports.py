
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from subplots import PdfSubplots

import bamboo.plotting


def save_report(dfgb, name, title=None, plot_func=None,
                *args, **kwargs):

    with PdfPages(name) as pdf:
        if title:
            add_title_page_to_pdf(pdf, title)

        add_subplots_to_pdf(pdf, dfgb, plot_func=plot_func,
                            *args, **kwargs)


def add_title_page_to_pdf(pdf, title, figsize=(30, 20),
                          end_page=True,
                          *args, **kwargs):

    fig = plt.figure(figsize=figsize)
    plt.axis('off')
    bamboo.plotting.plot_title(title, *args, **kwargs)

    if end_page:
        pdf.savefig()


def add_subplots_to_pdf(pdf, dfgb, plot_func=None,
                        nrows=3, ncols=3, figsize=(30, 20),
                        end_page=True,
                        skip_if_exception=False,
                        *args, **kwargs):
    """
    Take a pdf and a grouped dataframe and save
    Take a grouped dataframe and save a pdf of
    the histogrammed variables in that dataframe.

    plot_func - A function that takes the series and a list of
    arguments and kwargs and plots the given function

    TODO: Can we abstract this behavior...?
    """

    subplots = PdfSubplots(pdf, nrows, ncols, figsize=figsize)

    if plot_func is None:
        plot_func = bamboo.plotting._plot_and_decorate

    for (var, series) in dfgb._iterate_column_groupbys():

        subplots.next_subplot()
        try:
            plot_func(series, *args, **kwargs)
            subplots.end_iteration()
        except Exception as e:
            if skip_if_exception:
                subplots.skip_subplot()
            else:
                raise e

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
