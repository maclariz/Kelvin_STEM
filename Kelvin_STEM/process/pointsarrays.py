import numpy as np

"""
Functions to allow operations on pointsarrays that do not fall easily into another category 
(c.f. polarpoints for polar coordinates functions and cluster for clustering)
For example, area selection in a pointsarray is defined here.
"""


def rowlims(pointsarray, toplimit, bottomlimit):
    """
    finds the first and last rows in the points array where the vertical real space indices
    are the top and bottom of the rectangle drawn on the image
    """
    firstrow = np.where(pointsarray[:, 3] == toplimit)[0][0]
    lastrow = np.where(pointsarray[:, 3] == bottomlimit)[0][-1]
    return (firstrow, lastrow)
