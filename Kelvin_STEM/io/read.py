import numpy as np
import hyperspy.api as hs
import os

"""
This module provides a number of convenience functions for reading in datafiles from commonly used commercial data formats in our lab
"""


def read_mib_to_np(
    filepath,
    xdim,
    ydim,
    kx=256,
    ky=256,
    flybackpix=1,
    bit_depth=12,
    header=384,
    footer=0,
):
    """
    Reads a .mib file from Merlin on the ARM200F into a numpy data file

    Parameters
    ----------
    filepath: str
        A text string for the complete filepath of the file including the filemane
    xdim: int
        Vertical dimension of the scan in pixels
    ydim: int
        Horizontal dimension of the scan in pixels
    flybackpix: int
        Any pixels allowed for flyback data (defaults to 1)
    bit_depth: int
        Your chosen bit depth in the Merlin software, currently 1, 6 and 12 supported, 24 not supported yet
    header: int
        The header length in bytes on each detector frame (default 384)
    footer: int
        The footer length in bytes on each detector frame (default 0)

    Returns
    ----------
    data: np.ndarray
        A 4D array (xdim,ydim,kx,ky) of the data, possibly smaller than the expected xdim size
    """

    # set bit depth parameters
    if bit_depth == 12:
        data_type, b = np.uint16, 2
    elif bit_depth == 6:
        data_type, b = np.uint8, 1
    elif bit_depth == 1:
        data_type, b = np.uint8, 1
    else:
        raise Exception("Incorrect data type selected, please enter 12, 6 or 1")

    # map datafile to memory
    try:
        # Assuming the acquisition finished okay and the full datafile was recorded
        frametype = np.dtype(
            [
                ("head", np.uint8, header),
                ("data", data_type, (kx, ky)),
                ("foot", np.uint8, footer),
            ]
        )
        data = np.memmap(filepath, frametype, mode="r", shape=(xdim, ydim + flybackpix))
    except ValueError:
        # If the datafile didn't finish the full scan, we can still read what was there
        file_size = os.path.getsize(filepath)
        xdim = int(file_size / ((ydim + 1) * (kx * ky * b + header + footer)))
        frametype = np.dtype(
            [
                ("head", np.uint8, header),
                ("data", data_type, (kx, ky)),
                ("foot", np.uint8, footer),
            ]
        )
        data = np.memmap(filepath, frametype, mode="r", shape=(xdim, ydim + flybackpix))

    # Reshape to expected 4D shape
    data = data.reshape(xdim, ydim + flybackpix)
    # remove header
    if flybackpix > 0:
        return data["data"][:, :-flybackpix, :, :]
    else:
        return data["data"]


def read_mib_from_dmscan(
    dmpath, mibpath, kx=256, ky=256, flybackpix=1, bit_depth=12, header=384, footer=0
):
    """
    Reads a .mib file from Merlin on the ARM200F into a numpy data file

    Parameters
    ----------
    filepath: str
        A text string for the complete filepath of the file including the filemane
    xdim: int
        Vertical dimension of the scan in pixels
    ydim: int
        Horizontal dimension of the scan in pixels
    flybackpix: int
        Any pixels allowed for flyback data (defaults to 1)
    bit_depth: int
        Your chosen bit depth in the Merlin software, currently 1, 6 and 12 supported, 24 not supported yet
    header: int
        The header length in bytes on each detector frame (default 384)
    footer: int
        The footer length in bytes on each detector frame (default 0)

    Returns
    ----------
    data4d: np.ndarray
        A 4D array (xdim,ydim,kx,ky) of the data, possibly smaller than the expected xdim size
    """
    s = hs.load(dmpath)
    xdim, ydim = s.data.shape

    data4d = read_mib_to_np(
        mibpath, xdim, ydim, kx, ky, flybackpix, bit_depth, header, footer
    )
    return data4d
