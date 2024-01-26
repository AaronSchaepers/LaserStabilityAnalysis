#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 09:15:29 2024

@author: aaron
"""
import os
import numpy as np

from line_profiler import LineProfiler

directory = "/Users/aaron/Desktop/Institut/Data/Prelim"

def profiled_function(directory):
    # Create a list of the waveform files
    files = sorted(os.listdir(directory))
    
    # Delete all elements from the list that are no .txt files
    files = list(filter(lambda x: ".txt" in x, files))
    
    # Read first spectrum
    file_path = os.path.join(directory, files[0])
    first = []
    # Open file
    with open(file_path, 'r') as file:
        # Skip the first three lines
        for _ in range(3):
           next(file)
        # Read remaining lines
        for line in file:
            # Split the line by tab
            entries = line.strip().split('\t')
            # Append entries to first list
            first.append([float(x) for x in entries])
        # Convert first list to numpy array
        first = np.array(first)
    
    # Take frequency vector from first spectrum
    freqs = first[:,0]*1e6 # in Hz
    
    # Create array that contains all the spectra
    spectra = np.zeros([len(freqs), len(files)])
    number_spectra = len(files)
 
    # Iterate over all .txt files in the directory and open them
    for i in range(number_spectra):
        helper_spectrum = []
        file_path = os.path.join(directory, files[i])
        with open(file_path, 'r') as file:
            # Skip the first three lines
            for _ in range(3):
               next(file)
            # Read remaining lines
            for line in file:
                # Split the line by tab
                entries = line.strip().split('\t')
                # Extract the second entry (frequency) and append to the current spectrum
                helper_spectrum.append(float(entries[1]))
        # Add current spectrum to spectra array
        spectra[:,i] = np.array(helper_spectrum)
        
      
lp = LineProfiler()
lp_wrapper = lp(profiled_function)
lp_wrapper(directory)
lp.print_stats()