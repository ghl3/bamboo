

from bamboo import plotting


def hist(sgb, *args, **kwargs):
    return plotting._series_hist(sgb, *args, **kwargs)
