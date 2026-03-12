import numpy as np
from emdfile import tqdmnd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit


"""
This module provides functions for rapid cartesian-polar transformations of either a single image or a 4D dataset
and fitting of simple periodic functions to the polar transformed data
"""


def discfloat(ci, cj, Qrmin, Qrmax, segments):
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
    Qrmin: int
        the minimum radius desired in the polar transform dataset in pixels
    Qrmax: int
        the maximum radius desired in the polar transform dataset in pixels
    segments: int
        the number of angular segments in the transformed dataset (360 might be
        a common choice but no need to stick with this)

    Returns
    -------
    meshgrids: np.ndarray
        datatype is float
        Dimensions:
        0: 0 gives the meshgrid array for Qi positions, 1 gives the meshgrid array for Qj positions
        1: gives the Qr array
        2: gives the Qphi array

    """
    Qphi = np.arange(0, 2 * np.pi, 2 * np.pi / segments)
    Qr = np.arange(Qrmin, Qrmax)
    r_phi_mesh = np.meshgrid(Qr, Qphi)
    Qi = -r_phi_mesh[0] * np.sin(r_phi_mesh[1]) + ci
    Qj = r_phi_mesh[0] * np.cos(r_phi_mesh[1]) + cj
    meshgrids = np.array([Qi, Qj])
    return meshgrids


def Qrmax_test_adjust(Qrmax, Qshape, ci, cj):
    if isinstance(ci, np.ndarray):
        cimax = ci.max()
        cimin = ci.min()
        cjmax = cj.max()
        cjmin = cj.min()
    else:
        cimax = ci
        cimin = ci
        cjmax = cj
        cjmin = cj
    if cimax + Qrmax >= Qshape[0]:
        Qrmax = Qshape[0] - cimax - 1
        print(
            f"Qrmax adjusted to {Qrmax} as currently set bigger than the diffraction pattern"
        )
    elif cimin - Qrmax < 0:
        Qrmax = cimin
        print(
            f"Qrmax adjusted to {Qrmax} as currently set bigger than the diffraction pattern"
        )
    if cjmax + Qrmax >= Qshape[1]:
        Qrmax = Qshape[1] - cjmax - 1
        print(
            f"Qrmax adjusted to {Qrmax} as currently set bigger than the diffraction pattern"
        )
    elif cjmin - Qrmax < 0:
        Qrmax = cjmin
        print(
            f"Qrmax adjusted to {Qrmax} as currently set bigger than the diffraction pattern"
        )
    return Qrmax


def polarttransform(DP, ci, cj, Qrmin, Qrmax, segments, simple=True):
    """
    This is a function to polar transform a single diffraction pattern in a defined
    radial range and is the base function that does the calculations, which is then called
    by the functions for a DP or a 4DSTEM dataset
    Parameters
    ----------
    DP: np.ndarray
        the diffraction pattern to be transformed (must be 2D)
    ci: int, float
        the centre position in the cartesian array along axis 0 in pixels
    cj: int, float
        the centre position in the cartesian array along axis 1 in pixels
    Qrmin: int
        the minimum radius desired in the polar transform dataset in pixels
    Qrmax: int
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
    PT: np.ndarray
        datatype is float
        Dimensions:
        0: Qr (radial)
        1: Qphi (azimuthal starting horizontal right and proceeding ACW)
    """
    disc = discfloat(
        ci, cj, Qrmin, Qrmax, segments
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
    radweight = np.arange(Qrmin, Qrmax) * 2 * np.pi / segments
    azi = np.ones(shape=(segments))
    rweighting = np.meshgrid(azi, radweight)[1]
    PT = pt * rweighting

    return PT


def PTDP(DP, ci, cj, Qrmin, Qrmax, segments, simple=True):
    """
    This is a function to polar transform a single diffraction pattern in a defined
    radial range and is the function to call from a notebook, as it checks and
    adjusts (if required) the maximum radius used in the calculation before running
    Parameters
    ----------
    DP: np.ndarray
        the diffraction pattern to be transformed (must be 2D)
    ci: int, float
        the centre position in the cartesian array along axis 0 in pixels
    cj: int, float
        the centre position in the cartesian array along axis 1 in pixels
    Qrmin: int
        the minimum radius desired in the polar transform dataset in pixels
    Qrmax: int
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
    PTDP: np.ndarray
        datatype is float
        Dimensions:
        0: Qr (radial)
        1: Qphi (azimuthal starting horizontal right and proceeding ACW)
    """
    Qrmax = Qrmax_test_adjust(Qrmax, DP.shape, ci, cj)
    PTDP = polarttransform(DP, ci, cj, Qrmin, Qrmax, segments, simple)
    return PTDP


def PT4D(dataset, ci, cj, Qrmin, Qrmax, segments, simple=True):
    """
    This is a function to polar transform the Q dimensions of a 4DSTEM dataset only covering a limited
    radial range
    Parameters
    ----------

    dataset: np.ndarray
        the 4DSTEM dataset to be transformed (must be 4D, Ri, Rj, Qi, Qj)
    ci: int, float
        the centre position in the cartesian array along axis 0 in pixels
    cj: int, float
        the centre position in the cartesian array along axis 1 in pixels
    Qrmin: int
        the minimum radius desired in the polar transform dataset in pixels
    Qrmax: int
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
    PTDP: np.ndarray
        datatype is float
        Dimensions:
        0: Ri (vertical)
        1: Rj (horizontal)
        2: Qr (radial)
        3: Qphi (azimuthal, starting horizontal right and proceeding ACW)

    """

    Ri_max, Rj_max = dataset.shape[0], dataset.shape[1]
    Qrmax = Qrmax_test_adjust(Qrmax, (dataset.shape[2], dataset.shape[3]), ci, cj)
    PT4D = np.zeros(shape=(Ri_max, Rj_max, Qrmax - Qrmin, segments))

    # version of calculation for a single value for pattern centre
    if isinstance(ci, int):
        for Ri, Rj in tqdmnd(Ri_max, Rj_max):
            PT4D[Ri, Rj, :, :] = polarttransform(
                dataset[Ri, Rj, :, :], ci, cj, Qrmin, Qrmax, segments, simple=simple
            )

    # version of calculation for an array of pattern centres
    elif isinstance(ci, np.ndarray):
        assert (
            ci.shape[0] == Ri_max and cj.shape[1] == Rj_max
        ), "The array size for the pattern centres does not match the dataset"
        for Ri, Rj in tqdmnd(Ri_max, Rj_max):
            PT4D[Ri, Rj, :, :] = polarttransform(
                dataset[Ri, Rj, :, :],
                ci[Ri, Rj],
                cj[Ri, Rj],
                Qrmin,
                Qrmax,
                segments,
                simple=simple,
            )
    return PT4D


def plotpolar(polarDP, Qrmin, Qrmax, lines, title):
    """
    A convenience plotting function for plotting polar transformed data with labelled axes and lines delineating
    features such as HOLZ rings

    Parameters
    ----------
    polarDP: np.ndarray
        a single polar transformed diffraction pattern (2D array)
    Qrmin: int
        minimum radius of the transform in pixels
    Qrmax: int
        maximum radius of the transform in pixels
    lines: list
        a set of five line positions to delineate the Laue zone,
        between 2 and 3 is used as standard in defining the HOLZ ring
    title: str
        any chosen title for the plot

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(
        polarDP,
        vmin=np.percentile(polarDP, 5),
        vmax=np.percentile(polarDP, 98),
        cmap="turbo",
        extent=[0, 360, Qrmax, Qrmin],
        aspect=3,
    )
    ax.set_xlabel(r"$\phi,$ deg")
    ax.set_ylabel("radius, pixels")
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.xaxis.set_major_locator(MultipleLocator(60))

    ax.hlines(
        lines + Qrmin, 0, 359, color=["m", "k", "w", "w", "m"]
    )  # adding lines to mark out position of HOLZ ring
    ax.set_title(title)


def fun_cos_sq(phi, A2, phi2, A1, phi1, B):
    """
    Simple function for fitting azimuthal data as a sum of one-fold (cos) and two-fold (cos2) functions
    according to
    f = A2 cos^2(phi-phi2) + A1 cos(phi-phi1) + B

    Parameters
    ----------
    phi: float
        angle in radians
    A2: float
        amplitude of the 2-fold function
    phi2: float
        angular shift of the 2-fold function
    A1: float
        amplitude of the 1-fold function
    phi1: angular shift of the 1-fold function
    B: float
        baseline offset of the whole function from zero

    Returns
    -------
    f: float
        as defined above

    """
    return (
        A2 * np.cos(np.radians(phi - phi2)) ** 2
        + A1 * np.cos(np.radians(phi - phi1))
        + B
    )


def fitIntensity(
    data,
    func=fun_cos_sq,
    p0=[2000, 90, 2000, 0, 1000],
    bounds=([0, 0, 0, -180, 0], [np.inf, 180, np.inf, 180, np.inf]),
):
    """
    fit HOLZ ring intensity to a periodic function using scipy.optimize.curve_fit
    documentation for that function at:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

    Parameters
    ----------
    data: np.ndarray
        3D numpy array of intensity values corresponding to each detector pixel
        dim 0: Ri (real space vertical, down)
        dim 1: Rj (real space horizontal, right)
        dim 2: Rphi segments, (azimuthal angle from 0-360)
    func: function
        periodic function to fit data to
    p0: list, array, tuple
        length 5 list of initial values for use by curve_fit
    bounds: tuple
        tuple of two length 5 lists of parameters for the lower and upper bounds
        of the optimization
        typically, the amplitudes should simply be positive numbers
        the range over which the angle shifts should be defined depends on the dataset
        (e.g. a function with a peak angle close to 0 degrees is best defined with 0 in
        the middle of the angular range)
        whatever the exact choice of limits, the 2-fold function should be constrained
        in a 180 degree range and the 1-fold function in a 360 degree range

    Returns
    -------
    fitParams : np.ndarray
        3D numpy array of fit parameters
            Ri
            Rj
            [A2, phi2, A1, phi1, B]

    fitCov: 4D numpy array with covariance matrix of fitted parameters
    """
    Ri_max, Rj_max, segments = data.shape
    fitParams = np.zeros((Ri_max, Rj_max, 5))
    fitCov = np.zeros((Ri_max, Rj_max, 5, 5))

    for Ri, Rj in tqdmnd(Ri_max, Rj_max):
        pop, pcov = curve_fit(
            func, np.arange(segments), data[Ri, Rj], p0=p0, bounds=bounds
        )
        fitParams[Ri, Rj] = pop
        fitCov[Ri, Rj] = pcov
    return fitParams, fitCov
