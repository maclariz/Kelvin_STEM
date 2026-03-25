import matplotlib.patheffects as path_effects


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
