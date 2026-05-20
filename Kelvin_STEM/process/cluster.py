import numpy as np
import string
import matplotlib.pyplot as plt
import matplotlib.gridspec as GridSpec
import matplotlib.colors as colors
from colorsys import hsv_to_rgb

"""
This module provides functions for cluster analysis of pointsarrays using sklearn functions.  Whilst the sklearn functions
are not directly quoted in here, it explicitly uses the attributes of sklearn outputs.  Thus, at least one clustering 
method from sklearn.cluster has to be imported in your notebook to produce outputs that can be analysed by these tools.
"""
# General COM functions


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


def COM_cluster_Q(
    pointsarray, clusterresult, clusterlabel, weighted=False, returnint=False
):
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
    returnint: bool
        If true, returns total intensity for the cluster too

    Returns
    -------
    COM: np.ndarray
        [COMx,COMy,(intensity)]
    """

    clusterpoints = pointsarray[clusterresult.labels_ == clusterlabel]
    intensity = clusterpoints.T[2].sum()
    if weighted:
        weights = clusterpoints.T[2]
    else:
        weights = np.ones(shape=clusterpoints.shape[0])
    if returnint:
        COM = np.append(
            (clusterpoints.T[2] * clusterpoints.T[:2]).sum(axis=1) / weights.sum(),
            intensity,
        )
    else:
        COM = (clusterpoints.T[2] * clusterpoints.T[:2]).sum(axis=1) / weights.sum()
    return COM


def COMs_Q(pointsarray, clusterresult, weighted=True, returnint=False):
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
    returnint: bool
        If true, returns total intensity for the cluster too

    Returns
    -------
    COMs: np.ndarray
        An array of [COMx,COMyCOMy,(intensity)] values for each cluster
    """
    clusterlabels = np.unique(clusterresult.labels_)
    if returnint:
        i = 3
    else:
        i = 2
    COMs = np.zeros((clusterlabels.shape[0], i))
    for n, clusterlabel in enumerate(clusterlabels):
        COM = COM_cluster_Q(
            pointsarray,
            clusterresult,
            clusterlabel,
            weighted=weighted,
            returnint=returnint,
        )
        COMs[n, :i] = COM
    return COMs


# Basic functions


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


# Level 1 clustering - working on raw pointsarray data (possibly after some radial filtering)


def plot_L1_4D_clusters(L1cluster_result, pointsarray, QRmax, returnfig=False):
    """
    Takes a L1 cluster result of running some cluster algorithm in Scikit-Learn (e.g. DBSCAN)
    on 4D data in a points array and plots the results in reciprocal and real space.  Everything
    is plotted in uncalibrated pixels, since this is just about seeing the results.
    It expects you have applied some radial filtering (although this is not necessary)
    and sets a maximum radius in reciprocal space, purely for visualisation (not calculation)
    purposes.  It also plots the unclustered points in pale grey.
    You can use it for simple inline visualisation in a notebook, or can return a figure for
    saving.

    Parameters
    ----------
    L1cluster_result: complex object
        Result of a scikit-learn clustering algorithm (all have the same attributes).  Ultimately,
        it is the output of .labels_ that is used
    pointsarray: np.ndarray
        A points array, as defined in py4DSTEM.process.diffraction.digital_dark_field
    QRmax: int, float
        maximum radius for the reciprocal space plot
    returnfig: bool
        Tells whether to give a return.  If false, only displays a plot.
    Returns
    -------
    figure: matplotlib figure
        Only if returnfig==True
    """
    figure, axs = plt.subplots(1, 2, figsize=(12, 5.5))
    Rx_1, Rx_2 = int(pointsarray.T[3].min()), int(pointsarray.T[3].max())
    Ry_1, Ry_2 = int(pointsarray.T[4].min()), int(pointsarray.T[4].max())

    axs[0].set_title("DBscan, Qx, Qy, Rx, Ry")
    axs[0].set_xlabel("Qx (pix)", fontsize=24)
    axs[0].set_ylabel("Qy (pix)", fontsize=24)
    axs[0].set_ylim(QRmax, -QRmax)
    axs[0].set_xlim(-QRmax, QRmax)

    axs[1].set_title("DBscan, Qx, Qy, Rx, Ry")
    axs[1].set_xlabel("Ry (pix)", fontsize=24)
    axs[1].set_ylabel("Rx (pix)", fontsize=24)
    axs[1].set_ylim(Rx_2, Rx_1)
    cmap = "rainbow"

    uniquelabels = np.unique(L1cluster_result.labels_)
    COMs = COMs_R(pointsarray, L1cluster_result, weighted=False)

    for n, clusterlabel in enumerate(uniquelabels):
        cindex = n / uniquelabels[1:].shape[0] * 5 % 1
        if clusterlabel == -1:
            c = "lightgrey"
        else:
            c = plt.colormaps[cmap](cindex)

        points = pointsarray[L1cluster_result.labels_ == clusterlabel]

        axs[0].scatter(points.T[1], points.T[0], label=n, s=0.1, alpha=0.2, color=c)
        maxint = np.argmax(points.T[2])
        r, ang = points[maxint][5] + 8, np.radians(points[maxint][6])
        labx, laby = np.sin(ang) * r, np.cos(ang) * r
        axs[0].annotate(
            n,
            (points[maxint, 1], points[maxint, 0]),
            (laby, -labx),
            horizontalalignment="center",
            verticalalignment="center",
            size=7,
        )

        axs[1].scatter(points.T[4], points.T[3], label=n, s=1, alpha=0.5, color=c)
        if clusterlabel != -1:
            COM = COMs[clusterlabel]
            axs[1].text(COM[1], COM[0], clusterlabel)

    if returnfig:
        return figure


def show_L1_clusters_in_real_space(
    pointsarray, L1cluster_result, cluster_list=None, col=3, gamma=0.25, returnfig=False
):
    """
    Function to show real space plots of L1 clustering outputs

    Parameters
    ----------
    L1cluster_result: complex object
        Result of a scikit-learn clustering algorithm (all have the same attributes).  Ultimately,
        it is the output of .labels_ that is used
    pointsarray: np.ndarray
        A points array, as defined in py4DSTEM.process.diffraction.digital_dark_field
    cluster_list: np.ndarray (optional)
        You can pass a list of the clusters you wish to plot as an ndarray here.  If you don't, it
        plots everything except those in cluster -1 (unclustered points)
    cols: int
        number of columns to be used
    returnfig: bool
        Tells whether to give a return.  If false, only displays a plot.
    Returns
    -------
    figure: matplotlib figure
        Only if returnfig==True
    """
    shape = (int(pointsarray.T[3].max()) + 1, int(pointsarray.T[4].max()) + 1)
    if isinstance(cluster_list, (np.ndarray)):
        pass
    else:
        cluster_list = np.unique(L1cluster_result.labels_)[1:]
    l = cluster_list.shape[0]
    ar = shape[1] / shape[0]
    w = 10
    row = int(np.ceil(l / col))
    fig = plt.figure(figsize=(w, w * row / col / ar))
    gs = GridSpec.GridSpec(row, col)
    for n, cluster_label in enumerate(cluster_list):
        i, j = int(n / col), n % col
        ax = plt.subplot(gs[i, j])
        ax.set_axis_off()
        selpoints = pointsarray[L1cluster_result.labels_ == cluster_label]
        im = DDFimagefromselectedpoints(selpoints, shape)
        ax.imshow(im, norm=colors.PowerNorm(gamma=gamma), cmap="inferno")
        ax.text(
            5,
            5,
            cluster_label,
            color="w",
            size=14,
            fontweight="bold",
            verticalalalignment=top,
        )

    if returnfig:
        return fig


# Level 2 clustering - grouping L1 clusters that come from the same spatial locations
# these are referred to by letters, to avoid confusion with the numbered L1 clusters


def letters():
    """
    Produces a list of letters, uppercase first, for labelling level 2 (L2) clusters

    Returns
    -------
    letters: list
        list of strings [A,B,C....a,b,c....Da,Db,Dc....Dz]
    """
    l1 = [letter for letter in string.ascii_uppercase] + [
        letter for letter in string.ascii_lowercase
    ]
    l2 = ["A" + letter for letter in string.ascii_lowercase] + [
        "B" + letter for letter in string.ascii_lowercase
    ]
    l3 = ["C" + letter for letter in string.ascii_lowercase] + [
        "D" + letter for letter in string.ascii_lowercase
    ]
    return l1 + l2 + l3


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


def show_L2_clusters_in_real_space(
    pointsarray,
    L1cluster_result,
    L2cluster_result,
    L2key,
    letters,
    gamma=0.5,
    col=3,
    returnfig=False,
):
    """
    Function to show real space plots of L1 clustering outputs

     Parameters
    ----------
    pointsarray: np.ndarray
        A points array, as defined in py4DSTEM.process.diffraction.digital_dark_field
    L2cluster_result: complex object
        Result of a scikit-learn clustering algorithm (all have the same attributes).  Ultimately,
        it is the output of .labels_ that is used
    L1cluster_result: complex object
        Result of a scikit-learn clustering algorithm (all have the same attributes).  Ultimately,
        it is the output of .labels_ that is used
    gamma: float
        A gamma for the intensity normalisation (Power normalisation), <1 flattens contrast
    cols: int
        number of columns to be used
    returnfig: bool
        Tells whether to give a return.  If false, only displays a plot.
    Returns
    -------
    figure: matplotlib figure
        Only if returnfig==True
    """
    shape = (int(pointsarray.T[3].max()) + 1, int(pointsarray.T[4].max()) + 1)
    l = len(letters)
    ar = shape[1] / shape[0]
    w = 10
    row = int(np.ceil(l / col))
    fig = plt.figure(figsize=(w, w * row / col / ar))
    gs = GridSpec.GridSpec(row, col)

    ims = np.empty(shape=(*shape, l))
    for n, letter in enumerate(letters):
        selpoints = letterselectedpoints(
            pointsarray, L2key, letter, L1cluster_result, L2cluster_result
        )
        im = DDFimagefromselectedpoints(selpoints, shape)
        i, j = int(n / col), n % col
        ax = plt.subplot(gs[i, j])
        ax.set_axis_off()
        ax.imshow(im, cmap="inferno", norm=colors.PowerNorm(gamma=gamma))
        plt.text(
            5, 5, letter, color="w", size=14, fontweight="bold", verticalalalignment=top
        )

    if returnfig:
        return fig


def phimean_min_L2_cluster(
    letter, L2key, pointsarray, L1clusterresult, L2clusterresult
):
    """
    Finds the lowest mean phi value for any of the L1 clusters including in an L2 cluster
    (i.e. the lowest angle diffraction spot in this cluster of spots for a given crystal).
    Angle, as ever, measured ACW from horizontal right.

    Parameters
    ----------
    letter: str
        A letter in the list from letters() denoting one cluster in the L2 cluster result
    L2key: dict
        A dictionary relating letters to numbers which are the labels of the L2clusterresult
    pointsarray: np.ndarray
        The original array of diffraction peaks that was run through L1 clustering
    L1clusterresult: complex object
        The output of L1 clustering, either 4D (Qx,Qy,Rx,Ry) or 2D (Qx,Qy) on the pointsarray
    L2clusterresult: complex object
        The output of L2 clustering on COMs of each L1 result, calculated in (Rx,Ry)

    Returns
    -------
    phimeans.min(): float
        The lowest mean phi value
    """
    uniquelabels = np.unique(L1clusterresult.labels_)
    labels = uniquelabels[1:][L2clusterresult.labels_ == L2key[letter]]
    phimeans = np.empty(shape=labels.shape)
    for n, label in enumerate(labels):
        points = pointsarray[L1clusterresult.labels_ == label]
        phimeans[n] = points.T[6].mean()
    phimeans = np.where(phimeans < 0, phimeans + 180, phimeans)

    return phimeans.min()


def make_lettercolkey_from_phimean_min(
    pointsarray, L1clusterresult, L2clusterresult, L2key, saturation
):
    """
    makes a dictionary that converts a letter to a colour, based on the lowest angle phi in that L2 cluster

    Parameters
    ----------
    pointsarray: np.ndarray
        The original array of diffraction peaks that was run through L1 clustering
    L1clusterresult: complex object
        The output of L1 clustering, either 4D (Qx,Qy,Rx,Ry) or 2D (Qx,Qy) on the pointsarray
    L2clusterresult: complex object
        The output of L2 clustering on COMs of each L1 result, calculated in (Rx,Ry)
    L2key: dict
        A dictionary relating letters to numbers which are the labels of the L2clusterresult
    saturation: float
        A value from 0-1 for the saturation of colour in the HSV model

    Returns
    -------
    lettercolkey: dict
        Converts letters (str) to colours (as RGB in 0-1 range) as a 3-tuple
    """
    lettercolkey = {}
    for letter in L2key:
        phimean = phimean_min_L2_cluster(
            letter, L2key, pointsarray, L1clusterresult, L2clusterresult
        )
        h = phimean / 180
        col = hsv_to_rgb(h, saturation, 1)
        lettercolkey.update({letter: col})

    return lettercolkey


def filter_pointsarray_from_L2_cluster(
    pointsarray, L1cluster_result, L2cluster_result, L2key, letter
):
    """
    Function to show real space plots of L1 clustering outputs

     Parameters
    ----------
    pointsarray: np.ndarray
        A points array, as defined in py4DSTEM.process.diffraction.digital_dark_field
    L2cluster_result: complex object
        Result of a scikit-learn clustering algorithm (all have the same attributes).  Ultimately,
        it is the output of .labels_ that is used
    L1cluster_result: complex object
        Result of a scikit-learn clustering algorithm (all have the same attributes).  Ultimately,
        it is the output of .labels_ that is used
    L2key: dict
        A dictionary relating letters to numbers which are the labels of the L2clusterresult
    letter: str
        A letter denoting one of the clusters (must be in L2key)
    Returns
    -------
    selpoints: np.ndarray
        A points array, only containing points in the chosen cluster
    """
    selpoints = np.vstack(
        selpoints,
        letterselectedpoints(
            pointsarray, L2key, letter, L1cluster_result, L2cluster_result
        ),
    )

    return selpoints


def threecol_im_from_letters(
    pointsarray,
    L1clusterresult,
    L2clusterresult,
    imshape,
    L2key,
    letterlist,
    gamma=0.5,
    saturation=0.8,
):
    """
    makes a three colour image from a number of L2 clusters, denoted by a list of letters
    Uses a "lighten" algorithm where the lightest colour in each colour channel is chosen
    (much like an earlier idea implemented in Photoshop and similar).  Implementing a mid colour
    would be harder since you have to ignore all the zeros in every channel.

    Parameters
    ----------
    pointsarray: np.ndarray
        The original array of diffraction peaks that was run through L1 clustering
    L1clusterresult: complex object
        The output of L1 clustering, either 4D (Qx,Qy,Rx,Ry) or 2D (Qx,Qy) on the pointsarray
    L2clusterresult: complex object
        The output of L2 clustering on COMs of each L1 result, calculated in (Rx,Ry)
    imshape: tuple of ints
        shape of the images as a standard 2-tuple (of ints)
    L2key: dict
        A dictionary relating letters to numbers which are the labels of the L2clusterresult
    letterlist: str
        A list of letters in the list from letters() denoting one cluster in the L2 cluster result
    lettercolkey: dict
        Defined colours (as 3-tuples of (R,G,B) [in range 0-1]) for each letter in the letterlist
    gamma: float
        A number used for setting the power norm for display (<1 flattens contrast, 1 does nothing)

    Returns
    -------
    threecol_im: np.ndarray
        Image of shape (*imshape,3) that will plot in plt.imshow()
    stackmax: float
        The maximum intensity in the stack (for normalising against another image, if needed)
    """
    stack = np.zeros(shape=(imshape[0], imshape[1], 3, len(letterlist)))
    lettercolkey = make_lettercolkey_from_phimean_min(
        pointsarray, L1clusterresult, L2clusterresult, L2key, saturation
    )
    for n, letter in enumerate(letterlist):
        selpoints = letterselectedpoints(
            pointsarray, L2key, letter, L1clusterresult, L2clusterresult
        )
        im = DDFimagefromselectedpoints(selpoints, imshape) ** gamma
        stack[:, :, :, n] = im[:, :, np.newaxis]
    stackmax = stack.max()
    stack /= stackmax
    for n, letter in enumerate(letterlist):
        col = np.array(lettercolkey[letter])
        stack[:, :, :, n] *= col[np.newaxis, np.newaxis, :]
    threecol_im = stack.max(axis=3)

    return threecol_im, stackmax


# L3 clustering - remaking diffraction patterns from the data in L2 clusters - especially for ACOM


def plot_L3_2D_clusters(L3cluster_result, pointsarray, QRmax, s=100, returnfig=False):
    """
    Takes a L3 cluster result of running some cluster algorithm in Scikit-Learn (e.g. DBSCAN)
    on 2D data in a points array that relates to one crystal only and plots the results in
    reciprocal space only.  Everything is plotted in uncalibrated pixels, since this is just
    about seeing the results.
    No unclustered points are expected as everything in the cluster at L3 should be relevant
    You can use it for simple inline visualisation in a notebook, or can return a figure for
    saving.

    Parameters
    ----------
    L3cluster_result: complex object
        Result of a scikit-learn clustering algorithm (all have the same attributes).  Ultimately,
        it is the output of .labels_ that is used
    pointsarray: np.ndarray
        A points array, as defined in py4DSTEM.process.diffraction.digital_dark_field, expected to
        be heavily cut from the original via L1 and L2 clustering
    QRmax: int, float
        maximum radius for the reciprocal space plot
    s: int, float
        Multiplier for the size of the points in the plot (multiplies intensity to determine
        scatter marker size)
    returnfig: bool
        Tells whether to give a return.  If false, only displays a plot.
    Returns
    -------
    figure: matplotlib figure
        Only if returnfig==True
    """
    figure, ax = plt.subplots(1, 1, figsize=(2, 2))

    ax.set_title("DBscan, Qx, Qy")
    ax.set_xlabel("Qx (pix)", fontsize=24)
    ax.set_ylabel("Qy (pix)", fontsize=24)
    ax.set_ylim(QRmax, -QRmax)
    ax.set_xlim(-QRmax, QRmax)

    uniquelabels = np.unique(L3cluster_result.labels_)
    COMs3 = COMs_Q(pointsarray, L3cluster_result, weighted=True)

    for n, clusterlabel in enumerate(uniquelabels):
        cindex = n / uniquelabels[1:].shape[0] * 5 % 1
        c = plt.colormaps[cmap](cindex)

        points = pointsarray[L3cluster_result.labels_ == clusterlabel]

        ax.scatter(
            points.T[1], points.T[0], label=n, s=s * points.T[2], alpha=1, c="navy"
        )
        maxint = np.argmax(points.T[2])
        labx, laby = COMs3[n]
        ax.annotate(
            n,
            (laby, labx),
            (laby - 2, labx + 2),
            horizontalalignment="center",
            verticalalignment="center",
            color="w",
            size=7,
        )

    if returnfig:
        return figure
