import numpy as np
import string


"""
This module provides functions for cluster analysis of pointsarrays using sklearn functions.  Whilst the sklearn functions
are not directly quoted in here, it explicitly uses the attributes of sklearn outputs.  Thus, at least one clustering 
method from sklearn.cluster has to be imported in your notebook to produce outputs that can be analysed by these tools.
"""


def COM_cluster_R(pointsarray, clusterresult, clusterlabel, weighted=False):
    """
    Calculates either real space centre of mass (weighted by intensity) or a simplified version with
    no intensity from a specific cluster after running cluster analysis
    with scikit.learn on a pointsarray

    Parameters
    ----------
    pointsarray: np.ndarray
        A pointsarray that has been run through cluster analysis
    clusterresult: complex object
        The full output from a cluster analysis
    label: int
        A specific label that is present in clusterlabels
    weighted: bool
        True weights by intensity
        False has no weighting and treats all pixels in cluster as having equal weight

    Returns
    -------
    COM: np.ndarray
        [COMx,COMy]
    """

    clusterpoints = pointsarray[clusterresult.labels_ == clusterlabel]
    if weighted:
        COM = (clusterpoints.T[2] * clusterpoints.T[3:5]).sum(axis=1) / clusterpoints.T[
            2
        ].sum()
    else:
        COM = clusterpoints.T[3:5].sum(axis=1) / clusterpoints.shape[0]
    return COM


def COMs_R(pointsarray, clusterresult, weighted=True):
    """
    Calculates either the real space centre of mass (weighted by intensity) or a simplified version with
    no intensity for all clusters in the output of running cluster analysis
    with scikit.learn on a pointsarray

    Parameters
    ----------
    pointsarray: np.ndarray
        A pointsarray that has been run through cluster analysis
    clusterresult: complex object
        The full output from a cluster analysis
    weighted: bool
        True weights by intensity
        False has no weighting and treats all pixels in cluster as having equal weight

    Returns
    -------
    COMs: np.ndarray
        An array of [COMx,COMy] values for each cluster
    """
    clusterlabels = np.unique(clusterresult.labels_)[1:]
    COMs = np.zeros((clusterlabels.shape[0], 2))
    for n, clusterlabel in enumerate(clusterlabels):
        COM = COM_cluster_R(pointsarray, clusterresult, clusterlabel, weighted=weighted)
        COMs[n] = COM
    return COMs


def COM_cluster_Q(pointsarray, clusterresult, clusterlabel, weighted=False):
    """
    Calculates either reciprocal space centre of mass (weighted by intensity) or a simplified version with
    no intensity from a specific cluster after running cluster analysis
    with scikit.learn on a pointsarray

    Parameters
    ----------
    pointsarray: np.ndarray
        A pointsarray that has been run through cluster analysis
    clusterresult: complex object
        The full output from a cluster analysis
    label: int
        A specific label that is present in clusterlabels
    weighted: bool
        True weights by intensity
        False has no weighting and treats all pixels in cluster as having equal weight

    Returns
    -------
    COM: np.ndarray
        [COMx,COMy]
    """

    clusterpoints = pointsarray[clusterresult.labels_ == clusterlabel]
    if weighted:
        COM = (clusterpoints.T[2] * clusterpoints.T[:2]).sum(axis=1) / clusterpoints.T[
            2
        ].sum()
    else:
        COM = clusterpoints.T[:2].sum(axis=1) / clusterpoints.shape[0]
    return COM


def COMs_Q(pointsarray, clusterresult, weighted=True):
    """
    Calculates either the reciprocal space centre of mass (weighted by intensity) or a simplified version with
    no intensity for all clusters in the output of running cluster analysis
    with scikit.learn on a pointsarray

    Parameters
    ----------
    pointsarray: np.ndarray
        A pointsarray that has been run through cluster analysis
    clusterresult: complex object
        The full output from a cluster analysis
    weighted: bool
        True weights by intensity
        False has no weighting and treats all pixels in cluster as having equal weight

    Returns
    -------
    COMs: np.ndarray
        An array of [COMx,COMy] values for each cluster
    """
    clusterlabels = np.unique(clusterresult.labels_)
    COMs = np.zeros((clusterlabels.shape[0], 2))
    for n, clusterlabel in enumerate(clusterlabels):
        COM = COM_cluster_Q(pointsarray, clusterresult, clusterlabel, weighted=weighted)
        COMs[n] = COM
    return COMs


def letterselectedpoints(pointsarray, L2key, letter, L1clusterresult, L2clusterresult):
    """
    Select points in the original pointsarray from level 2 real-space cluster labels
    (each of which clusters several of the level 1 clusters).  So, the L1 clusters are
    found from the L2 clusters, and then the L1 cluster labels are used to select the points.

    Parameters
    ----------
    pointsarray: np.ndarray
        A pointsarray that has been run through cluster analysis
    L2 key: dict
        A dictionary that maps the letter labels onto the numbers that were the actual outputs
        of L2 clustering
    letter: str
        A single character, usually a lower or upper case letter denoting an L2 cluster
    L1clusterresult: complex output
        Output of the level 1 clustering
    L2clusterresult: complex output
        Output of the level 2 clustering

    Returns
    -------
    selectedpoints: np.ndarray
        A Nx7 array of selected points from the pointsarray

    """
    L1clusters = np.argwhere(L2clusterresult.labels_ == L2key[letter]).T[0]
    selectedpoints = np.empty(shape=(0, 7))
    for L1cluster in L1clusters:
        selectedpoints = np.vstack(
            (selectedpoints, pointsarray[L1clusterresult.labels_ == L1cluster])
        )
    return selectedpoints


def DDFfromcluster(pointsarray, clusterlabels, label, Rshape):
    """
    Makes a Digital Dark Field image from a specific cluster after running cluster analysis
    with scikit.learn on a pointsarray

    Parameters
    ----------
    pointsarray: np.ndarray
        A pointsarray that has been run through cluster analysis
    clusterlabels: np.ndarray
        An array of cluster labels, usually from .labels_, of same length as pointsarray
    label: int
        A specific label that is present in clusterlabels
    Rshape: tuple (int,int)
        Real space shape of the image of (Rx,Ry)

    Returns
    -------
    DDFim: np.ndarray
        A 2D image
    """
    assert label in clusterlabels, f"{label} is not found in the list of clusters"
    clusterpoints = pointsarray[clusterlabels == label]
    DDFim = DDFimagefromselectedpoints(clusterpoints, Rshape)
    # DDFstack[
    #     clusterpoints.T[3].astype(int),
    #     clusterpoints.T[4].astype(int),
    #     np.arange(clusterpoints.shape[0])
    # ] = clusterpoints.T[2]
    # DDFim = DDFstack.sum(axis=2)
    return DDFim


def DDFimagefromselectedpoints(selectedpoints, Rshape):
    """
    Similar to DDFfromcluster, but makes a Digital Dark Field image from a cut down points array,
    produced after running cluster analysis with scikit.learn on a pointsarray

    Parameters
    ----------
    selectedpoints: np.ndarray
        A pointsarray that has been run through cluster analysis (usually at least 2 levels)
    Rshape: tuple (int,int)
        Real space shape of the image of (Rx,Ry)

    Returns
    -------
    DDFim: np.ndarray
        A 2D image
    """

    DDFim = np.zeros(shape=Rshape)
    for point in selectedpoints:
        DDFim[point[3].astype(int), point[4].astype(int)] += point[2]
    return DDFim


def letters():
    """
    Produces a list of letters, uppercase first, for labelling level 2 (L2) clusters

    Returns
    -------
    letters: list
        list of strings [A,B,C....x,y,z]
    """
    return string.ascii_uppercase + string.ascii_lowercase
