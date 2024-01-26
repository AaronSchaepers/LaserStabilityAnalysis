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
import matplotlib.pyplot as plt

# Define plot layout
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "cmr10"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["grid.alpha"] = 0
plt.rcParams["grid.color"] = "grey"
plt.rcParams["axes.linewidth"] = 1
#plt.rcParams["axes.grid"] = True

# Directory (0_analysis folder)
path = "/Volumes/~go68jit/TUM-PC/Dokumente/Daten/Laser stabilization/Koheras to comb locking/20240122-0002/waveforms/0_Analysis"

# Linewidth of the plot line
lw = 0.8

# Save the plots?
save = False

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
plt.plot(time_array/3600, fitresults[2,:]*1e-6, linewidth=lw) # Convert frequency to MHz
plt.fill_between(time_array/3600, (fitresults[2,:]-fitresults_std[2,:])*1e-6,\
                                  (fitresults[2,:]+fitresults_std[2,:])*1e-6,\
                                  alpha=0.3) # One sigma band
plt.ylim((22.33, 22.44))
plt.xlabel("Time (h)")
plt.ylabel("Beat frequency (MHz)")
######## Inset ########
ax2 = plt.axes([0.28, 0.55, scaling_inset*3.375, scaling_inset*2.086]) # left, bottom, width, height (with respect to full figure, maximum = 1)
ax2.plot(time_array/3600, fitresults[2,:]*1e-6, linewidth=lw) # Convert frequency to MHz
ax2.set_xlim((0.5, 0.534))
ax2.set_ylim((22, 22.6))
ax2.set_xlabel("")
########################
plt.tight_layout()
if save == True:
    plt.savefig(os.path.join(path, "Frequency.png"), format="png", dpi=900)
    plt.savefig(os.path.join(path, "Frequency.svg"), format="svg", dpi=300)
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
plt.xlabel("Fourier frequency (Hz)")
plt.ylabel("Power spectrum (arb. u.)")
plt.tight_layout()
plt.tight_layout()
if save == True:
    plt.savefig(os.path.join(path, "Power_spectrum.png"), format="png", dpi=900)
    plt.savefig(os.path.join(path, "Power_spectrum.svg"), format="svg", dpi=300)
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
plt.plot(taus, allandev, linewidth=lw)
plt.fill_between(taus, allandev-allandev_std, allandev+allandev_std, alpha=0.3) # One sigma band
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$\tau$ (s)")
plt.ylabel("Overlapping Allan deviation")
plt.tight_layout()
if save == True:
    plt.savefig(os.path.join(path, "Allan_deviation.png"), format="png", dpi=900)
    plt.savefig(os.path.join(path, "Allan_deviation.svg"), format="svg", dpi=300)
plt.show()
plt.close()

###############################################################################
# Linewidth
###############################################################################
scaling = 1.5
plt.figure(figsize=(scaling*3.375, scaling*2.086)) # Ensures golden ratio
plt.plot(time_array/3600, fitresults[3,:]*1e-3, linewidth=lw) # Convert linewidth to kHz
plt.fill_between(time_array/3600, (fitresults[3,:]-fitresults_std[3,:])*1e-3,\
                                  (fitresults[3,:]+fitresults_std[3,:])*1e-3,\
                                  alpha=0.3) # One sigma band
plt.ylim((5, 55))
plt.xlabel("Time (h)")
plt.ylabel("Beat linewidth (kHz)")
plt.tight_layout()
if save == True:
    plt.savefig(os.path.join(path, "Linewidth.png"), format="png", dpi=900)
    plt.savefig(os.path.join(path, "Linewidth.svg"), format="svg", dpi=300)
plt.show()
plt.close()




