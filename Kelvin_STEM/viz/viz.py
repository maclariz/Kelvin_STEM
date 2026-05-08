import matplotlib.patheffects as path_effects
import numpy as np
from colorsys import hsv_to_rgb

"""
This module provides functions for visualisation and plotting.
"""


def add_scalebar(
    ax,
    scalebarparams={
        "calibration": 1,
        "units": "pixels",
        "horizontal": 10,
        "vertical": 10,
        "caliblength": 10,
        "voffset": 3,
        "fs": 18,
        "lw": 3,
    },
):
    """
    Adds a scalebar to an existing set of axes, usually used on images

    Parameters
    ----------
    ax: matplotlib axes object
        The axes you want to add the scalebar to.  Define using plt.add_subplot, plt.subplots or similar
    scalebarparams: dict
        A dictionary of parameters for the scalebar as below
        {
            'calibration': calibration in physical units per pixel,
            'calibration units': units of the calibration,
            'horizontal': position of left edge of scalebar in pixels or plot units,
            'vertical': vertical position of scalebar in pixels or plot units,
            'caliblength': length of the scalebar in physical units,
            'voffset': vertical offset of text from bar
            'fs': fontsize of text
            'lw': line width of bar
        }
        You can specify as many of those as you wish.  Not specifying a parameter leaves the default set
        (e.g. no need to set voffset, fs and lw in many cases)
    """

    scalebar = {
        "calibration": 1,
        "units": "pixels",
        "horizontal": 10,
        "vertical": 10,
        "caliblength": 10,
        "voffset": 3,
        "fs": 18,
        "lw": 3,
    }
    scalebar.update(scalebarparams)

    ax.plot(
        [
            scalebar["horizontal"],
            scalebar["horizontal"] + scalebar["caliblength"] / scalebar["calibration"],
        ],
        [scalebar["vertical"], scalebar["vertical"]],
        lw=scalebar["lw"],
        color="w",
        path_effects=[
            path_effects.Stroke(linewidth=scalebar["lw"] + 3, foreground="k"),
            path_effects.Normal(),
        ],
    )
    ax.text(
        scalebar["horizontal"] + scalebar["caliblength"] / scalebar["calibration"] / 2,
        scalebar["vertical"] - scalebar["voffset"],
        str(scalebar["caliblength"]) + scalebar["units"],
        horizontalalignment="center",
        color="w",
        fontsize=scalebar["fs"],
        fontweight="bold",
        path_effects=[
            path_effects.Stroke(linewidth=3, foreground="k"),
            path_effects.Normal(),
        ],
    )


def make_HSV_const_wheel_parameters(minangle=0, maxangle=360, innerradius=0.5):
    """
    gives arrays for plotting into a constant color wheel, however, this needs to be plotted
    into an inset axis with projection='polar' set at declaration
    (note different color wheels can be made where color varies with radiusw)

    Parameters
    ----------
    minangle: int, float
        The minimum angle (in degrees) to use in the wheel
    maxangle: int, float
        The maximum angle (in degrees) to use in the wheel
    innerradius: float
        The inner radius (in range 0-1)

    Returns
    -------
    P: np.ndarray
        array of phi values
    S: np.ndarray
        array of saturation values (radii for plot)
    c: np.ndarray
        array of c values (color tuples)
    """
    assert 0 <= innerradius < 1, "set min radius between 0 and 1"

    # Set up the angle ranges
    phi = np.linspace(np.radians(minangle), np.radians(maxangle), 300)
    hue = (phi - np.radians(minangle)) / (np.radians(maxangle) - np.radians(minangle))
    sat = np.linspace(innerradius, 1, 100)

    # Make meshgrids for plotting
    P, S = np.meshgrid(phi, sat)
    H = (P - np.radians(minangle)) / (np.radians(maxangle) - np.radians(minangle))
    V = np.ones_like(S)

    # Make the colours
    p, h, s, v = (
        P.flatten().tolist(),
        H.flatten().tolist(),
        (V * sat_overall).flatten().tolist(),
        V.flatten().tolist(),
    )
    c = [hsv_to_rgb(*x) for x in zip(h, s, v)]
    c = np.array(c)

    return P, S, c
