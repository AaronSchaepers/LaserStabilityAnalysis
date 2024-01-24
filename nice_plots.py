#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Aaron
@date: 2024/01/23
@github: https://github.com/AaronSchaepers/laser_stability_analysis
"""
import os
import pickle
import allantools

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Define plot layout
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["text.usetex"] = True

lw = 0.8

# Directory (0_analysis folder)
path = "/Users/aaron/Desktop/Institut/Data/20240109-0001/0_Analysis"

# =============================================================================
# # Import datadict issued from the analysis script
# with open(os.path.join(path, "datadict.pkl"), "rb") as file:
#     datadict = pickle.load(file)
# =============================================================================
    
# Extract relevant variables from the datadict    
fitresults = datadict["fitresults"]
fitresults_std = datadict["fitresults_std"]
time_array = datadict["time_array"]
dt = datadict["dt"]


###############################################################################
# Frequency
###############################################################################
scaling = 1.5
scaling_inset = 0.15
plt.figure(figsize=(scaling*3.375, scaling*2.086)) # Ensures golden ratio
plt.plot(time_array, fitresults[2,:]*1e-6, linewidth=lw) # Convert frequency to MHz
plt.xlabel("Time (s)")
plt.ylabel("Beat frequency (MHz)")
######## Zoom-in ########
ax2 = plt.axes([0.27, 0.57, scaling_inset*3.375, scaling_inset*2.086]) # left, bottom, width, height (with respect to full figure, maximum = 1)
ax2.plot(time_array, fitresults[2,:]*1e-6, linewidth=lw) # Convert frequency to MHz
ax2.set_xlim((0, 400))
ax2.set_ylim((22.94, 23.05))
ax2.set_xlabel("") #labelpad for positioning closer to the axis)
########################
#plt.tight_layout()
plt.show()
plt.close()


###############################################################################
# Power spectrum of frequency
###############################################################################
# Calculate power spectrum (FFT) of beat frequency
n_samples = len(fitresults[2,:])
# Fourier frequency resolution = 1 / sampling time
df = 1/(n_samples*dt) 
# Length of the FFT array according to np.fft.rfft docs
n_powerspectrum = n_samples//2+1
# Calculate Fourier frequency vector
freqs_powerspectrum = np.arange(0, n_powerspectrum) * df
# Get Hanning window
hanning = np.hanning(n_samples)
# Calculate power spectrum as defined e.g. in
# https://pure.mpg.de/rest/items/item_152164/component/file_152163/content,
# weighting the data with the hanning window
powerspectrum = 2 * np.real(np.fft.rfft(fitresults[2,:]*hanning))**2 /sum(hanning)**2 
# Plot the power spectum
scaling = 1.5
plt.figure(figsize=(scaling*3.375, scaling*2.086)) # Ensures golden ratio
plt.plot(freqs_powerspectrum, powerspectrum, linewidth=lw)
plt.yscale("log")
plt.xlabel("Fourier frequency (Hz) (s)")
plt.ylabel("Power spectrum (arb. u.)")
plt.tight_layout()
plt.show()
plt.close()

###############################################################################
# Overlapping Allan deviation
###############################################################################
# Calculate Allan deviation
fractional_frequency = fitresults[2,:] / np.mean(fitresults[2,:])
rate = 1/dt
taus, allandev, allandev_std, ns = allantools.oadev(fractional_frequency, rate, data_type="freq")

# Plot Allan deviation
scaling = 1.5
plt.figure(figsize=(scaling*3.375, scaling*2.086)) # Ensures golden ratio
plt.errorbar(taus, allandev, allandev_std, linewidth=lw, fmt=".")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$\tau$")
plt.ylabel("Overlapping Allan deviation")
plt.tight_layout()
plt.show()
plt.close()

###############################################################################
# Linewidth
###############################################################################
scaling = 1.5
plt.figure(figsize=(scaling*3.375, scaling*2.086)) # Ensures golden ratio
plt.plot(time_array, fitresults[3,:]*1e-3, linewidth=lw) # Convert linewidth to kHz
plt.xlabel("Time (s)")
plt.ylabel("Beat linewidth (kHz)")
plt.tight_layout()
plt.show()
plt.close()




