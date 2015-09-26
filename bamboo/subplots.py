
import matplotlib.pyplot as plt


class PdfSubplots:

    def __init__(self, pdf, rows, cols, figsize=(30, 20)):
        self.pdf = pdf
        self.rows = rows
        self.cols = cols
        self.idx = -1
        self.num_subplots = rows * cols
        self.skip_set = False
        self.figsize = figsize

    def next_subplot(self):

        if self.skip_set:
            self.skip_set = False
            return

        self.idx += 1

        # Start a new page if necessary
        if self.idx % self.num_subplots == 0:
            plt.figure(figsize=self.figsize)
            plt.subplot(self.rows, self.cols, 1)
        else:
            get_next_subplot()

    def skip_subplot(self):
        self.skip_set = True

    def end_iteration(self):
        if self.idx % self.num_subplots == self.num_subplots - 1:
            plt.tight_layout()
            self.pdf.savefig()

    def finalize(self):
        if self.idx % self.num_subplots != self.num_subplots - 1:
            plt.tight_layout()
            self.pdf.savefig()


def print_subplot_info():
    ax = get_current_subplot()
    print "Current subplot %s (%s, %s) out of (%s, %s)" % \
        (id(ax), ax.rowNum, ax.colNum, ax.numRows, ax.numCols)


def get_current_subplot():
    """
    Return the current subplot
    Throw an exception if not a subplot
    """

    # Get the current plot
    ax = plt.gca()

    # Check if the current plot is a subplot
    if ax.__class__.__name__ != "AxesSubplot":
        print "Cannot get next subplot, current axis is not a subplot"
        raise Exception()

    return ax


def get_next_subplot(wrap=True):
    """
    Create and move to the next subplot
    in a grid of matplotlib subplots
    """

    # Get the current plot
    ax = get_current_subplot()

    if ax.is_last_col() and ax.is_last_row():
        if wrap:
            print "Wrapping: rows: %s cols: %s (last row: %s last col: %s" % \
                (ax.numRows, ax.numCols, ax.is_last_col(), ax.is_last_row())
            plt.subplot(ax.numRows, ax.numCols, 1)
            return
        else:
            print "Cannot get next subplot, on last row and column"
            raise Exception()

    # Do some magic to get the current subplot index
    row, col = ax.rowNum, ax.colNum
    ax.numRows, ax.numCols

    idx = ax.numCols * row + col + 1
    next_idx = idx + 1
    plt.subplot(ax.numRows, ax.numCols, next_idx)


def append_to_title(text):
    title = plt.gca().get_title()
    title = title + text
    plt.title(title)
