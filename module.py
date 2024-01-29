#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Aaron
@date: 2024/01/08
@github: https://github.com/AaronSchaepers/laser_stability_analysis
"""

import os
import pickle

import numpy as np

from numpy import pi

###############################################################################
# 1 Functions for data handling
###############################################################################

""" Save an object using the pickle module """
# The object will be stored inside a new "folder" in the given "directory".
def save_object(directory, folder, name, file):
    # Define place to store object
    where_to_store = os.path.join(directory, folder)
    
    # Check if this directory exists, if not, create it
    if not os.path.exists(where_to_store): 
        os.makedirs(where_to_store) 
        
    # Save the object
    with open(os.path.join(where_to_store, name+".pkl"), "wb") as f:
        pickle.dump(file, f, pickle.HIGHEST_PROTOCOL)
    return()


###############################################################################
# 2 Functions for the fitting procedure 
###############################################################################

""" Lorentzian as used in spectroscopy """
# https://en.wikipedia.org/wiki/Cauchy_distribution
# Here, lw = Half Width at Half Maximum!
def lorentzian(x, b, c, x0, lw):
   f = b + c / pi * lw / ((x-x0)**2 + lw**2)
   return(f)

""" Jacobian matrix of the Lorentzian as defined above """
def jacobian(x, b, c, x0, lw):
    # Define partial derivatives
    Ldiffb = np.ones(x.shape)
    Ldiffc = lw / pi / ((x-x0)**2 + lw**2)
    Ldiffx0 = c / pi * 2 * lw * (x-x0) / ((x-x0)**2 + lw**2)**2
    Ldifflw = c / pi * (1 / ((x-x0)**2 + lw**2) -  2 * lw**2 / ((x-x0)**2 + lw**2)**2 )
    jac = np.stack((Ldiffb, Ldiffc, Ldiffx0, Ldifflw)).T
    return(jac)
    
        
        
        
        
        
        