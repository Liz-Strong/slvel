import numpy as np
from scipy.special import factorial
import scipy as sp
import matplotlib.pyplot as plt
import scipy.io as sio
"""Calculate the intensities of electric fields.
"""
def calculate_e_field_intensity(l, p, w0, x, y, petaledbeam):
    """
    calculates intensity field of LG beam on a grid with size set by x and y
    
    Args:
        l (int): OAM azimuthal mode number
        p (int): OAM radial mode number
        w0 (float): beam waist
        x (int): grid size in x direction
        y (int): grid size in y direction
        petaledbeam (bool): calculates interference fringes of conjugate (+/-l) beams if true, calculates intensity field of single beam if false
        
    Returns:
        intensity_normalized (np array): intensity field calculated on x,y grid normalized by maximum intensity
        intensity_shape (tuple of ints): shape of intensity_normalized
        
    """
    x = np.arange(-np.floor(x/2.), np.floor(x/2.))
    y = np.arange(-np.floor(y/2.), np.floor(y/2.))
    
    x_, y_ = np.meshgrid(x, y)
    r = np.sqrt(x_**2+y_**2)
    phi = np.arctan2(y_, x_)
    gaussian = np.sqrt(2) / (np.sqrt(np.pi)*w0) * np.exp(-r**2/(w0**2))
    
    # LG beam for arbitrary l & p
    LG = sp.special.genlaguerre(p, abs(l))
    
    LGlp = np.sqrt(factorial(p)) / np.sqrt(factorial(p+np.absolute(l)))\
           * (np.sqrt(2)*r/w0)**(np.absolute(l)) * np.exp(1j*l*phi) * gaussian\
           * LG(2*r**2/w0**2)
    
    if petaledbeam:
        fieldout = (LGlp+np.conj(LGlp))/2
        intensity = np.abs(fieldout*np.conj(fieldout))      
    else:
        intensity = np.abs(LGlp*np.conj(LGlp))
        
    intensity_normalized = np.double(intensity/np.max(intensity))
    intensity_shape = intensity_normalized.shape
    return intensity_normalized, intensity_shape

def calculate_e_field_intensity_linear(w0, x, y, alpha, fringes=True):
    """
    calculates intensity field of Gaussian beam on a grid with size set by x and y
    
    Args:
        w0 (float): beam waist
        x (int): grid size in x direction
        y (int): grid size in y direction
        alpha (float): angle between beams
        fringes (bool): calculates interference fringes if True
    
    Returns:
        intensity_normalized (np array): intensity field calculated on x,y grid normalized by maximum intensity
        intensity_shape (tuple of ints): shape of intensity_normalized
    
    """
    x = np.arange(-np.floor(x/2.),np.floor(x/2.))
    y = np.arange(-np.floor(y/2.),np.floor(y/2.))

    x_, y_ = np.meshgrid(x,y)
    r = np.sqrt(x_**2+y_**2)
    phi = np.arctan2(y_, x_)
    gaussian = np.sqrt(2) / (np.sqrt(np.pi)*w0) * np.exp(-r**2/(w0**2)) * np.exp(-alpha*1j*x_)
    if fringes:
        fieldout = (gaussian+np.conj(gaussian))/2
        intensity = np.abs(fieldout*np.conj(fieldout))
    else:
        intensity = np.abs(gaussian*np.conj(gaussian))
        
    intensity_normalized = np.double(intensity/np.max(intensity))
    intensity_shape = intensity_normalized.shape
    return intensity_normalized, intensity_shape
