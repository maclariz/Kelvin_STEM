import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit


"""
This module provides functions for working with nanobeam diffraction data in a pointsarray (i.e. vector) representation
and particularly in the Qr, Qphi polar representation.  This includes histogram (similar to diffractogram) plots and distortion
correction where it is easiest to parameterise the distortion in polar coordinates.
"""


def make_radial_histogram(
    pointsarray, 
    firstrow, 
    lastrow, 
    s1=0, 
    s2=None, 
    histbins = 256
):
    '''
    Makes a radial histogram for a selected region (only limited in vertical direction)
    from a points array.  The first and last rows should be defined and the range of radii
    in the output can be cut.
    
    Parameters
    ----------
    pointsarray:  np.ndarray
        Nx7 array of Qx, Qy, I, Rx, Ry, Qr, Qphi
    firstrow: int
        first row of pointsarray (axis 0)
    lastrow: int
        last row of pointsarray (axis 1)
    s1: int
        bin from which to start the output
    s2: int
        bin at which to stop the output
    histbins: int
        the number of bins to use in calculating the histogram in the first place
    '''
    if s2==None:
        s2 = histbins
    Iradialcalib = np.histogram(
        pointsarray[firstrow:lastrow,5],
        np.arange(histbins)-.5,
        weights=pointsarray[firstrow:lastrow,2]
    )
    Iradial = Iradialcalib[0][s1:s2]
    r = Iradialcalib[1][s1:s2]+.5
    return (r, Iradial)

def sinfunc(phi,a,phi1,r0):
    '''
    Makes a simple function of:
    a sin(phi+phi1) + r0
    
    Parameters
    ----------
    phi: float
        the main variable in degrees, defined in a suitable angular range (e.g. -180 - 180)
    a: float
        amplitude of sine oscillation
    phi0: float
        angular offset of the origin of the function, in degrees
    r0: float
        mean radius of the function
        
    Returns
    -------
    a sin(phi+phi1) + r0: float
    '''
    return a*np.sin(np.radians(phi+phi1))+r0

def plotring(pointsarray,figax,Qr1,Qr2,s=0.1,alpha=0.1,color='hotpink'):
    '''
    Plots the diffraction peaks in a dataset with a selection ring superimposed
    
    Parameters
    ----------
    pointsarray: np.ndarray
        Nx7 array of diffraction peaks as defined elsewhere.  The Qr and Qphi columns are needed.
    figax: tuple
        (fig,ax) defining the figure and axis to plot in
    Qr1: float, int
        The inner radius of the ring
    Qr2: float, int
        The outer radius of the ring
    s: float
        Optional set of the point size in the scatter plot.  Only adjust if too few spots and too faint.
    alpha: float (0-1)
        Transparency of the points in the scatter plot.  Only adjust if too few spots and too faint.
        
    Returns
    -------
    pointsarrayring: np.ndarray
        Nx7 array of diffraction peaks as within the defined radii.  
    '''
    fig,ax = figax
    ax.set_facecolor('silver')
    ann = mpatches.Annulus((0,0),Qr2,Qr2-Qr1,fc='white',ec='k',lw=0.75)
    ax.add_patch(ann)
    ax.scatter(pointsarray.T[1],pointsarray.T[0],s=s,alpha=alpha,color=color)

def setradialrange(pointsarray,Qr1,Qr2,s=0.1,alpha=0.1):
    '''
    Sets a radial range for selecting points on a ring in polycrystalline diffraction data
    for use in distortion correction with graphical output to allow the radii to be adjusted
    until the user is happy that the ring is completely selected with no significant overspill 
    to other diffraction spots
    
    Parameters
    ----------
    pointsarray: np.ndarray
        Nx7 array of diffraction peaks as defined elsewhere.  The Qr and Qphi columns are needed.
    Qr1: float, int
        The inner radius of the ring
    Qr2: float, int
        The outer radius of the ring
    s: float
        Optional set of the point size in the scatter plot.  Only adjust if too few spots and too faint.
    alpha: float (0-1)
        Transparency of the points in the scatter plot.  Only adjust if too few spots and too faint.
        
    Returns
    -------
    pointsarrayring: np.ndarray
        Nx7 array of diffraction peaks within the defined radii.  
    '''
    fig,ax = plt.subplots(figsize=(4,4))
    plotring(pointsarray,(fig,ax),Qr1,Qr2,color='hotpink')
    pointsarrayring = pointsarray[
        np.logical_and(
            pointsarray.T[5]<Qr2,
            pointsarray.T[5]>Qr1        
        )
    ]
    return pointsarrayring

def findcomadistortion(pointsarrayring, s=0.1, alpha=0.1):
    '''
    Corrects azimuthal distortion of a diffraction ring, where the effective camera length is longer
    in one direction than the diametrically opposed direction (probably from slight misalignment of
    beam into project lenses) and returns the correction coefficients only.  Does a simple visualisation
    of the fit and correction.
    
    Parameters
    ----------
    pointsarrayring: np.ndarray
        Nx7 array of diffraction peaks within a pair of defined radii.
    s: float
        Optional set of the point size in the scatter plot.  Only adjust if too few spots and too faint.
    alpha: float (0-1)
        Transparency of the points in the scatter plot.  Only adjust if too few spots and too faint.

    Returns
    -------
    pop: np.ndarray
        Array of the three fit parameters [a, phi0, r0]
    '''
    fig,axs = plt.subplots(1,2,figsize=(8,2))
    axs[0].set_title('Uncorrected and fit')
    axs[0].scatter(pointsarrayring.T[6],pointsarrayring.T[5], s=s, alpha=alpha)
    lims = (pointsarrayring.T[5].min(),pointsarrayring.T[5].max())
    #Fit
    pop, pcov = curve_fit(
        sinfunc,
        pointsarrayring.T[6],
        pointsarrayring.T[5],
        p0 = [pointsarrayring.T[5].std(),0,pointsarrayring.T[5].mean()],
        bounds = [
            [0,-180,0],
            [10,180,2000]
        ],

    )
    phiax = np.linspace(-180,180,100)
    axs[0].plot(phiax,sinfunc(phiax,*pop),color='grey')
    axs[0].set_ylim(*lims)
    axs[0].xaxis.set_major_locator(MultipleLocator(60))
    axs[0].set_xlim(-180,180)
    
    #Correct
    pointsarrayrcorr = (
        pointsarrayring.T[5]*
        (1-pop[0]/pop[2]*np.sin(np.radians(pointsarrayring.T[6]+pop[1])))
    )
    axs[1].scatter(pointsarrayring.T[6],pointsarrayrcorr, s=s, alpha=alpha)
    axs[1].set_title('Corrected')
    axs[1].set_ylim(*lims)
    axs[1].xaxis.set_major_locator(MultipleLocator(60))
    axs[1].set_xlim(-180,180)

    return pop

def undistortarray(rawarray,a,phi0,r0,Qr1,Qr2,plot=True):
    '''
    Corrects azimuthal distortion of a whole pointsarray ring, where the effective camera length is longer
    in one direction than the diametrically opposed direction (probably from slight misalignment of
    beam into project lenses) using the correction coefficients previously calculated.  If plot is set to
    True, then it displays a before and after.
    
    Parameters
    ----------
    rawarray: np.ndarray
        Nx7 array of diffraction peaks with some distortion needing correction.
    a: float
        amplitude of sine oscillation from fit parameters
    phi0: float
        angular offset of the origin of the function, in degrees, from fit parameters
    r0: float
        mean radius of the function from fit parameters
    Qr1: float, int
        The inner radius of the ring (only used if plot==True)
    Qr2: float, int
        The outer radius of the ring (only used if plot==True)
    plot: bool
        Whether to turn plotting on

    Returns
    -------
    correctedarray: np.ndarray
         Nx7 array of diffraction peaks after distortion correction.
   '''
    
    #correct
    # getting the arrays and running the calcs
    rraw = rawarray[:,5]
    phi = rawarray[:,6]
    corr = (1-a/r0*np.sin(np.radians(phi+phi0)))
    rcorr = rraw*corr
    xcorr = -np.sin(np.radians(phi))*rcorr
    ycorr = np.cos(np.radians(phi))*rcorr
    
    #restacking the array
    correctedarray = np.vstack(
        (
            xcorr,
            ycorr,
            rawarray.T[2:5],
            rcorr,
            phi
        )
    ).T
    
    #visualisation
    if plot:
        fig,axs = plt.subplots(1,2,figsize=(8,3.8))
        axs[0].set_title('Uncorrected')
        plotring(rawarray,(fig,axs[0]),Qr1,Qr2,s=0.01,alpha=0.1,color='hotpink')
        axs[1].set_title('Corrected')
        plotring(correctedarray,(fig,axs[1]),Qr1,Qr2,s=0.01,alpha=0.1,color='green')
            
    return correctedarray
