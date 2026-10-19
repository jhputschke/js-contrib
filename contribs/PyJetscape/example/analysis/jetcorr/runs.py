"""The analyses' data and figures, outside the code: example/analysis/runs/<run>/data and
runs/<run>/figures, one run per campaign or test set (``zjet_setup``: the Z+jet test and the
campaign set-up checks; ``zjet_c00-01``: campaigns c00 + c01).  runs/ is git-ignored: the
hadron files are large, and the results are not public yet."""
import os

#: example/analysis/runs
RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")


def data(run, *parts):
    """runs/<run>/data[/parts]"""
    return os.path.join(RUNS, run, "data", *parts)


def figures(run, *parts):
    """runs/<run>/figures[/parts]"""
    return os.path.join(RUNS, run, "figures", *parts)
