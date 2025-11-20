import numpy as np

"""
This module provides functions for rapid cartesian-polar transformations of either a single image or a 4D dataset
"""


def discfloat(ci, cj, rmin, rmax, segments):
    """
    This is a backend function for the other functions below which generates
    a meshgrid of values corresponding to a regular array r and phi values back
    in a cartesian i, j frame, which can then be used to lookup all the right pixels
    in a diffraction pattern to allow quick transformation from cartesian to polar
    representations of the data I have chosen to use i and j to refer to vertical and
    horizontal axes in data (rather than x and y, because some packages use x for
    horizontal (e.g. matplotlib) and some use x for axis 0, which is vertical in numpy arrays)

    It is calculated for a finite range of radii which need not be the full range in the
    cartesian data, this may help speed and reduce memory requirements when used in
    the calculation of large datasets (i.e. only calculate the radii you care about)

    Parameters
    ----------
    ci: int, float
        the centre position in the cartesian array along axis 0 in pixels
    cj: int, float
        the centre position in the cartesian array along axis 1 in pixels
    rmin: int
        the minimum radius desired in the polar transform dataset in pixels
    rmax: int
        the maximum radius desired in the polar transform dataset in pixels
    segments: int
        the number of angular segments in the transformed dataset (360 might be
        a common choice but no need to stick with this)

    Returns
    -------
    meshgrids: np.ndarray
        datatype is float
        Dimensions:
        0: 0 gives the array for i positions, 1 gives the array for j positions
        1: gives the radial array
        2: gives the azimuth array

    """
    phi = np.arange(0, 2 * np.pi, 2 * np.pi / segments)
    r = np.arange(rmin, rmax)
    r_phi_mesh = np.meshgrid(r, phi)
    i = -r_phi_mesh[0] * np.sin(r_phi_mesh[1]) + ci
    j = r_phi_mesh[0] * np.cos(r_phi_mesh[1]) + cj
    meshgrids = np.array([i, j])
    return meshgrids


def polarttransform(DP, ci, cj, rmin, rmax, segments, simple=True):
    """
    This is a function to polar transform a single diffraction pattern only covering a limited
    radial range
    Parameters
    ----------
    DP: np.ndarray
        the diffraction pattern to be transformed (must be 2D)
    ci: int, float
        the centre position in the cartesian array along axis 0 in pixels
    cj: int, float
        the centre position in the cartesian array along axis 1 in pixels
    rmin: int
        the minimum radius desired in the polar transform dataset in pixels
    rmax: int
        the maximum radius desired in the polar transform dataset in pixels
    segments: int
        the number of angular segments in the transformed dataset
        Advisable to use  an appropriate number of segments to approximately match 2*pi*r at
        the largest radius of interest in your analysis to get a good sampling of the
        original data in your transform
    simple: bool
        True: just calculates intensity from nearest pixel to every cartesian grid reference
        from the polar (r,phi) grid
        False: calculates a weighted average of the four nearest pixels to that grid reference
        (slower but more robust to single pixel glitches)

    Returns
    -------
    meshgrids: np.ndarray
        datatype is float
        Dimensions:
        0: 0 gives the array for i positions, 1 gives the array for j positions
        1: gives the radial array
        2: gives the azimuth array


    """

    disc = discfloat(
        ci, cj, rmin, rmax, segments
    )  # get basic disc of all transform positions

    if simple:
        pos = np.round(disc, 0).astype(
            "int16"
        )  # round the disc array to nearest integer
        pos2 = pos.reshape((2, pos.shape[1] * pos.shape[2]))  # turn to a 1D list
        pt = DP[pos2[0], pos2[1]].reshape(pos.shape[1], pos.shape[2]).T
        # calculate PT by using the pos2 array to slice the original array

    else:
        shape = disc[0].shape[0] * disc[0].shape[1]
        disc0 = disc[0].reshape(shape)  # turn into linear array of i positions
        disc1 = disc[1].reshape(shape)  # turn into linear array of j positions
        ui = np.floor(disc0).astype("int16")  # find upper i pixel array
        li = np.ceil(disc0).astype("int16")  # find lower i pixel array
        li = np.where(
            li == ui, li + 1, li
        )  # deals with the case of an exact hit on an i position
        lj = np.floor(disc1).astype("int16")  # find left j pixel array
        rj = np.ceil(disc1).astype("int16")  # find right j pixel array
        rj = np.where(
            rj == lj, lj + 1, rj
        )  # deals with the case of an exact hit on a j position
        wul = (1 - (disc0 - ui)) * (1 - (disc1 - lj))  # weighting parameter upper left
        wur = (1 - (disc0 - ui)) * (1 - (rj - disc1))  # weighting parameter upper right
        wll = (1 - (li - disc0)) * (1 - (disc1 - lj))  # weighting parameter lower left
        wlr = (1 - (li - disc0)) * (1 - (rj - disc1))  # weighting parameter lower right
        pt = (
            (DP[ui, lj] * wul + DP[ui, rj] * wur + DP[li, lj] * wll + DP[li, rj] * wlr)
            .reshape(disc[0].shape[0], disc[0].shape[1])
            .T
        )

    # Now weight result by pixel area in transform image
    radweight = np.arange(rmin, rmax) * 2 * np.pi / segments
    azi = np.ones(shape=(segments))
    rweighting = np.meshgrid(azi, radweight)[1]
    PTDP = pt * rweighting

    return PTDP
