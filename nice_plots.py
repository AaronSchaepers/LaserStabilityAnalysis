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
path = r"\\nas.ads.mwn.de\go68jit\TUM-PC\Dokumente\Daten\Laser stabilization\Koheras comb lock\20240109-0001\waveforms\0_Analysis"

# Linewidth of the plot line
lw = 0.8

# Save the plots?
save = True

# Import datadict issued from the analysis script
with open(os.path.join(path, "datadict.pkl").replace(os.sep, "/"), "rb") as file:
    datadict = pickle.load(file)

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
# Create figure with size that ensures golden ratio
plt.figure(figsize=(scaling*3.375, scaling*2.086))
plt.xlabel("Time (h)")
plt.ylabel("Beat frequency (MHz)")
# If requried, define plot range
#plt.ylim((22.33, 22.44))
#plt.xlim((0,4.1))
# Plot and convert frequency to MHz
plt.plot(time_array/3600, fitresults[2,:]*1e-6, linewidth=lw)
 # Show fit uncertainty as shaded band around the plot
plt.fill_between(time_array/3600, (fitresults[2,:]-fitresults_std[2,:])*1e-6,\
                                  (fitresults[2,:]+fitresults_std[2,:])*1e-6,\
                                  alpha=0.3)

######## Inset ########
# =============================================================================
# ax2 = plt.axes([0.28, 0.55, scaling_inset*3.375, scaling_inset*2.086]) # left, bottom, width, height (with respect to full figure, maximum = 1)
# ax2.plot(time_array/3600, fitresults[2,:]*1e-6, linewidth=lw) # Convert frequency to MHz
# ax2.set_xlim((0.5, 0.532))
# ax2.set_ylim((22, 22.55))
# ax2.set_xlabel("")
# =============================================================================
######## Inset ########

plt.tight_layout()
if save == True:
    plt.savefig(os.path.join(path, "Frequency.png"), format="png", dpi=900, bbox_inches='tight')
    plt.savefig(os.path.join(path, "Frequency.svg"), format="svg", dpi=300, bbox_inches='tight')
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

# Create figure with size that ensures golden ratio
scaling = 1.5
plt.figure(figsize=(scaling*3.375, scaling*2.086))
plt.xlabel("Fourier frequency (Hz)")
plt.ylabel("Power spectrum (arb. u.)")
# Plot powerspectrum
plt.plot(freqs_powerspectrum, powerspectrum, linewidth=lw)
plt.yscale("log")
plt.tight_layout()
if save == True:
    plt.savefig(os.path.join(path, "Power_spectrum.png"), format="png", dpi=900, bbox_inches='tight')
    plt.savefig(os.path.join(path, "Power_spectrum.svg"), format="svg", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

###############################################################################
# Overlapping Allan deviation
###############################################################################
# Calculate Allan deviation
fractional_frequency = fitresults[2,:] / np.mean(fitresults[2,:])
rate = 1/dt
taus, allandev, allandev_std, ns = allantools.oadev(fractional_frequency, rate, data_type="freq")

scaling = 1.5
# Create figure with size that ensures golden ratio
plt.figure(figsize=(scaling*3.375, scaling*2.086))
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$\tau$ (s)")
plt.ylabel("Overlapping Allan deviation")
# If required, define plot range
#plt.ylim((9e-6, 3e-4))
# Plot Allan deviation
plt.plot(taus, allandev, linewidth=lw)
 # Show fit uncertainty as shaded band around the plot
plt.fill_between(taus, allandev-allandev_std, allandev+allandev_std, alpha=0.3) # One sigma band

plt.tight_layout()
if save == True:
    plt.savefig(os.path.join(path, "Allan_deviation.png"), format="png", dpi=900, bbox_inches='tight')
    plt.savefig(os.path.join(path, "Allan_deviation.svg"), format="svg", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

###############################################################################
# Linewidth
###############################################################################
scaling = 1.5
# Create figure with size that ensures golden ratio
plt.figure(figsize=(scaling*3.375, scaling*2.086))
plt.xlabel("Time (h)")
plt.ylabel("FWHM (kHz)")
# If requried, define plot range
#plt.ylim((10, 110))
# Plot linewidth, converted from Hz to kHz by factor 1e-3
plt.plot(time_array/3600, fitresults[3,:]*1e-3, linewidth=lw)
 # Show fit uncertainty as shaded band around the plot
plt.fill_between(time_array/3600, (fitresults[3,:]-fitresults_std[3,:])*1e-3,\
                                  (fitresults[3,:]+fitresults_std[3,:])*1e-3,\
                                  alpha=0.3) # One sigma band

plt.tight_layout()
if save == True:
    plt.savefig(os.path.join(path, "Linewidth.png"), format="png", dpi=900, bbox_inches='tight')
    plt.savefig(os.path.join(path, "Linewidth.svg"), format="svg", dpi=300, bbox_inches='tight')
plt.show()
plt.close()





