#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:07:14 2023

@author: Aaron
@date: 2024/01/08
@github: https://github.com/AaronSchaepers/laser_stability_analysis

Required packages: pyqtgraph, allantools
    
Class GUI:
Inherits from the QtWidgets.QMainWindow class

Methods:
    - 
Attributes:
    - One for each interactive element of the GUI
    - self.datadict: A dictionary containing all relevant quantities from the 
                     original data and its processing
        - directory: Directory of the .txt files containing the spectra
        - number_spectra (int): Number of spectra
        - dt (float): The fixed time between spectrum accquisitions
        - freqs (1D array): The frequency vector in Hz
        - spectra (2D array): y data of the spectra,
        - self.initvals (list): User defined initial values for the Lorentzian fit
        - time_array (1D array): Array of the wave form accquisition times
        - fitrange_idx (list): Lower and upper bound of fitrange
        - initvals (list): Initial values in the order (offset, intensity, position, linewidth)
        - fitresults (2D array): Lorentzian best fit parameters for all spectra
        - fitresults_std (2D array): Standard deviation of Lorentzian best fit parameters for all spectra
    - self.plotted_spectrum: Currently plotted spectrum in the frame_spectrum
    - self.plotted_peak: Currently plotted Lorentzian in the frame_spectrum
    - 
"""

import sys
import os
import pickle
import allantools

import numpy as np
import pyqtgraph as pg
import module as mod

from scipy.optimize import curve_fit
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QLineEdit, QPushButton, QMainWindow, QProgressBar
from PyQt5.QtCore import QCoreApplication # To update progress bars in real time



###############################################################################
""" The main window and interactive part """
###############################################################################
# Create a base class that will load the .ui file in the constructor
# It inherits the QtWidgets.QMainWindow class because I created a new 
# "Main Window" when selecting the form type when first creating the .ui file 
# in PyQt Designer
class GUI(QMainWindow):
    def __init__(self):
        
        # Call the inherited classes __init__ method
        super(GUI, self).__init__() 
        
        # Load the .ui file
        uic.loadUi('gui.ui', self) 
        
        # Connect the plot frames to attributes of this class
        self.frame_spectrum = self.findChild(pg.PlotWidget, 'frame_spectrum')
        self.frame_frequency = self.findChild(pg.PlotWidget, 'frame_frequency')
        self.frame_linewidth = self.findChild(pg.PlotWidget, 'frame_linewidth')
        self.frame_allandev = self.findChild(pg.PlotWidget, 'frame_allandev')
        self.frame_linhist = self.findChild(pg.PlotWidget, 'frame_linhist')
        
        # Set up the mouse click events for frequency and linewdith plot
        self.frame_frequency.scene().sigMouseClicked.connect(self.signal_frequency)
        self.frame_linewidth.scene().sigMouseClicked.connect(self.signal_linewidth)
        
        # Define axis labels and white background
        self.frame_spectrum.setLabel("left", "Amplitude", "dBu", textColor="k")
        self.frame_spectrum.setLabel("bottom", "Frequency", "Hz", textColor="k")
        self.frame_spectrum.setBackground("w")
        
        self.frame_frequency.setLabel("left", "Beat frequency", "Hz", textColor="k")
        self.frame_frequency.setLabel("bottom", "Time", "s", textColor="k")
        self.frame_frequency.setBackground("w")
        
        self.frame_linewidth.setLabel("left", "Beat linewidth", "Hz", textColor="k")
        self.frame_linewidth.setLabel("bottom", "Time", "s", textColor="k")
        self.frame_linewidth.setBackground("w")
        
        self.frame_allandev.setLabel("left", "Overlapping Allan deviation", textColor="k")
        self.frame_allandev.setLabel("bottom", "Tau", "s", textColor="k")
        self.frame_allandev.setBackground("w")
        
        self.frame_linhist.setLabel("left", "N", textColor="k")
        self.frame_linhist.setLabel("bottom", "Beat linewidth", "Hz", textColor="k")
        self.frame_linhist.setBackground("w")
        
        # Set axis pen color to black
        self.frame_spectrum.getAxis('bottom').setPen(pg.mkPen(color="k"))  # X-axis
        self.frame_spectrum.getAxis('left').setPen(pg.mkPen(color="k"))    # Y-axis
        
        self.frame_frequency.getAxis('bottom').setPen(pg.mkPen(color="k"))  # X-axis
        self.frame_frequency.getAxis('left').setPen(pg.mkPen(color="k"))    # Y-axis
        
        self.frame_linewidth.getAxis('bottom').setPen(pg.mkPen(color="k"))  # X-axis
        self.frame_linewidth.getAxis('left').setPen(pg.mkPen(color="k"))    # Y-axis
        
        self.frame_allandev.getAxis('bottom').setPen(pg.mkPen(color="k"))  # X-axis
        self.frame_allandev.getAxis('left').setPen(pg.mkPen(color="k"))    # Y-axis
        
        self.frame_linhist.getAxis('bottom').setPen(pg.mkPen(color="k"))  # X-axis
        self.frame_linhist.getAxis('left').setPen(pg.mkPen(color="k"))    # Y-axis
        
        # Set axis tick and label color to black
        self.frame_spectrum.getAxis('bottom').setTextPen(color="k")
        self.frame_spectrum.getAxis('left').setTextPen(color="k")
        
        self.frame_frequency.getAxis('bottom').setTextPen(color="k")
        self.frame_frequency.getAxis('left').setTextPen(color="k")
        
        self.frame_linewidth.getAxis('bottom').setTextPen(color="k")
        self.frame_linewidth.getAxis('left').setTextPen(color="k")
        
        self.frame_allandev.getAxis('bottom').setTextPen(color="k")
        self.frame_allandev.getAxis('left').setTextPen(color="k")
        
        self.frame_linhist.getAxis('bottom').setTextPen(color="k")
        self.frame_linhist.getAxis('left').setTextPen(color="k")
        
        # Connect the GUI input elements to attributes of this class
        self.input_name = self.findChild(QLineEdit, 'in_name')
        self.input_dt = self.findChild(QLineEdit, 'in_dt')
        self.input_init_off = self.findChild(QLineEdit, 'in_init_off')
        self.input_init_int = self.findChild(QLineEdit, 'in_init_int')
        self.input_init_pos = self.findChild(QLineEdit, 'in_init_pos')
        self.input_init_lw = self.findChild(QLineEdit, 'in_init_lw')
        self.input_fitrange_l = self.findChild(QLineEdit, 'in_fitrange_l')
        self.input_fitrange_u = self.findChild(QLineEdit, 'in_fitrange_u')
        
        # Connect the progress bars to attributes of this class
        self.pB_import = self.findChild(QProgressBar, "pB_import")
        self.pB_fitting = self.findChild(QProgressBar, "pB_fitting")
        self.pB_import.setValue(0)
        self.pB_fitting.setValue(0)
        
        # Connect the "Choose directory..." button to a method of this class
        self.choose_directory_button = self.findChild(QPushButton, 'b_choose_directory')
        self.choose_directory_button.clicked.connect(self.choose_directory_clicked) 
        
        # Connect the "Check initial values" button to a method of this class
        self.check_initvals_button = self.findChild(QPushButton, 'b_check_initvals')
        self.check_initvals_button.clicked.connect(self.check_initvals_clicked)
        
        # Connect the "Start fitting" button to a method of this class
        self.start_fitting_button = self.findChild(QPushButton, 'b_start_fitting')
        self.start_fitting_button.clicked.connect(self.fit_all_spectra)
        
        # Connect the "Plot results" button to a method of this class
        self.plot_results_button = self.findChild(QPushButton, 'b_plot_results')
        self.plot_results_button.clicked.connect(self.plot_results_clicked)
        
        # Show the GUI
        self.show()
        
        
    # Called when "Choose dierctory" is clicked
    #   - Plots the first spectrum
    #   - Imports all other spectra
    #   - Creates time steps array
    #   - Creates datadict
    def choose_directory_clicked(self):        
        # Open a dialog where the user can choose the directory
        file_dialog = QFileDialog()
        path = file_dialog.getExistingDirectory(window, 'Select data directory')
        
        # Create attribute of self storing the directory
        if path:
            directory = path
            
        # Create a list of the waveform files and store them in self.spectra
        files = sorted(os.listdir(directory))
        
        # Delete all elements from the list that are no .txt files
        files = list(filter(lambda x: ".txt" in x, files))
        
        # Read first spectrum and create self.spectra and self.freqs
        first = np.genfromtxt(os.path.join(directory, files[0]), delimiter="\t", skip_header=2)
        freqs = first[:,0]*1e6 # in Hz
        spectra = np.zeros([len(freqs), len(files)])
        number_spectra = len(files)
        
        # Set the min and max of the import progress bar
        self.pB_import.setMinimum(0)
        self.pB_import.setMaximum(number_spectra)
        
        # Iterate over all spectra and save them to self.spectra
        for i in range(len(files)):
            self.pB_import.setValue(i+1)
            QCoreApplication.processEvents()  # Process GUI events to update the progress bar
            
            # Load and process the spectra
            path = os.path.join(directory, files[i])
            helper = np.genfromtxt(path, delimiter="\t", skip_header=2)
            spectra[:,i] = helper[:,1]
            
        # Plot the first spectrum
        self.plotted_spectrum = self.frame_spectrum.plot(freqs, spectra[:,0], pen=pg.mkPen(color='k'))
         
        # Create time steps array (not needed in this method but what's done is done)
        dt = float(window.input_dt.text())
        time_array = np.arange(0, number_spectra)*dt
        
        # Create the attribute self.datatict and save all relevant variables
        # from this measurement in it
        self.datadict = {}
        self.datadict["freqs"] = freqs
        self.datadict["spectra"] = spectra
        self.datadict["number_spectra"] = number_spectra 
        self.datadict["dt"] = dt
        self.datadict["time_array"] = time_array
        self.datadict["directory"] = directory
        
        
    # Called when "Check initial values" is clicked
    #   - Read out the user input of initial values and fit range
    #   - Plot a Lorentzian based on the initial values in frame_spectrum
    def check_initvals_clicked(self):
        # Get relevant quantities from datadict
        freqs = self.datadict["freqs"]
        
        # Get fit range
        fitrange_l_freq = float(window.input_fitrange_l.text())*1e6 # to Hz
        fitrange_u_freq = float(window.input_fitrange_u.text())*1e6 # to Hz
        fitrange_l_idx = (np.abs(freqs - fitrange_l_freq)).argmin()
        fitrange_u_idx = (np.abs(freqs - fitrange_u_freq)).argmin()
        fitrange_idx = [fitrange_l_idx, fitrange_u_idx]

        # Get initial values
        initvals = [float(window.input_init_off.text()),
                    float(window.input_init_int.text())*1e6, # from MHz*dBu to Hz*dBu
                    float(window.input_init_pos.text())*1e6, # from MHz to Hz
                    float(window.input_init_lw.text())*1e3]  # from kHz to Hz
        
        # Remove previously plotted peak if it exists
        if hasattr(self, 'plotted_peak'):
            self.frame_spectrum.removeItem(self.plotted_peak)
            
        # Plot the new Lorentzian and store the item
        self.plotted_peak = self.frame_spectrum.plot(freqs[fitrange_idx[0]:fitrange_idx[1]],\
                                                     mod.lorentzian(freqs[fitrange_idx[0]:fitrange_idx[1]], *initvals),\
                                                     pen=pg.mkPen(color='r'))
        
        # Save relevant new variables in the datadict
        self.datadict["fitrange_idx"] = fitrange_idx
        self.datadict["initvals"] = initvals
    
        
    # Called when "Start fitting" is clicked
    #   - Iterate over all spectra and perform a Lorentzian fit to each within
    #     the chosen fitrange
    #   - Store the best fit parameters and standard deviations in two separate
    #     arrays
    #   - Create a folder called "0_Analysis" in the data directory and store 
    #     the datadict in it
    def fit_all_spectra(self):
        # Get relevant quantities from datadict
        freqs = self.datadict["freqs"]
        spectra = self.datadict["spectra"]
        fitrange_idx = self.datadict["fitrange_idx"]
        number_spectra = self.datadict["number_spectra"]
        initvals = self.datadict["initvals"]
        directory = self.datadict["directory"]
        
        # Array for storing fitresults and uncertainties
        fitresults = np.zeros([4, number_spectra])
        fitresults_std = np.zeros([4, number_spectra])
        
        # Define boundaries to speed up the fitting routine
        #                   off      int     pos    lw
        lbounds = np.array((-np.inf, 0,      0,      0))
        ubounds = np.array((np.inf,  np.inf, np.inf, np.inf))
        
        # Set the min and max of the import progress bars
        self.pB_fitting.setMinimum(0)
        self.pB_fitting.setMaximum(number_spectra)
        
        # Iterate over all spectra and do the fitting
        p0 = initvals
        for i in range(number_spectra):
            # Fit
            try:
                # Use highest point in spectrum as inital value for pos
                idx_max = np.argmax(spectra[fitrange_idx[0]:fitrange_idx[1], i])
                p0[2] = freqs[fitrange_idx[0]:fitrange_idx[1]][idx_max]
                # Fit
                popt, pcov = curve_fit(mod.lorentzian,\
                                       freqs[fitrange_idx[0]:fitrange_idx[1]],\
                                       spectra[fitrange_idx[0]:fitrange_idx[1], i],\
                                       p0, bounds=(lbounds, ubounds), jac=mod.jacobian)
                pstd = np.sqrt(np.diag(pcov))
                # If the fit worked, use best fit parameters as initial values 
                # for next spectrum
                p0 = initvals
            except:
                popt = np.zeros([4])
                pstd = np.zeros([4])
                # If the fit didn't work, use original initial values
                p0 = initvals
            # Save rfit results
            fitresults[:,i] = popt
            fitresults_std[:,i] = pstd
            
            # Update progress bar
            window.pB_fitting.setValue(i+1)
            QCoreApplication.processEvents()  # Process GUI events to update the progress bar
        
        # Save relevant quantities in the datadict
        self.datadict["fitresults"] = fitresults        
        self.datadict["fitresults_std"] = fitresults_std
        
        # Save the datadict
        mod.save_object(directory, "0_Analysis", "datadict", self.datadict)
        print("datadict saved")
    
        
    # Called when "Load and plot results" is clicked
    #   - Open a file dialog in which the user can choose a previously saved 
    #     datadict that is to be analyzed
    #   - From the chosen datadict, plot the peak frequency and linewidth as a
    #     function of time
    #   - Calculate the fractional frequency and from this the overlapping Allan
    #     deviation which is also plotted
    def plot_results_clicked(self):
        # Initiate file dialog
        file_dialog = QFileDialog()
        
        # Open a dialog where the user can choose the previously saved data 
        # instance to be plotted
        file_path, _ = file_dialog.getOpenFileName(window, 'Import File')
        
        # Import chosen datadict
        with open(file_path, "rb") as file:
            self.datadict = pickle.load(file)
        
        # Get relevant quantities from datadict
        fitresults = self.datadict["fitresults"]
        fitresults_std = self.datadict["fitresults_std"]
        time_array = self.datadict["time_array"]
        dt = self.datadict["dt"]
        number_spectra = self.datadict["number_spectra"]
        
        # Clear the frames in case there are previous plots
        self.frame_frequency.clear()
        self.frame_linewidth.clear()
        self.frame_allandev.clear()
        self.frame_linhist.clear()
            
        # Create shaded 1 sigma bands
        # Frequency
        line1_frequency = pg.PlotCurveItem(x=time_array, y=fitresults[2,:]-fitresults_std[2,:], pen=pg.mkPen(color='k'))
        line2_frequency = pg.PlotCurveItem(x=time_array, y=fitresults[2,:]+fitresults_std[2,:], pen=pg.mkPen(color='k'))
        shade_area_frequency = pg.FillBetweenItem(curve1=line1_frequency, curve2=line2_frequency, brush=pg.mkBrush(color=(200, 200, 200, 70)))
        # Linewidth
        line1_linewidth = pg.PlotCurveItem(x=time_array, y=fitresults[3,:]-fitresults_std[3,:], pen=pg.mkPen(color='k'))
        line2_linewidth = pg.PlotCurveItem(x=time_array, y=fitresults[3,:]+fitresults_std[3,:], pen=pg.mkPen(color='k'))
        shade_area_linewidth = pg.FillBetweenItem(curve1=line1_linewidth, curve2=line2_linewidth, brush=pg.mkBrush(color=(200, 200, 200, 70)))
        
        # Add plots and shaded sigma bands to plots
        self.frame_frequency.plot(x=time_array, y=fitresults[2,:], pen=pg.mkPen(color='k'))
        self.frame_linewidth.plot(x=time_array, y=fitresults[3,:], pen=pg.mkPen(color='k'))
        self.frame_frequency.addItem(shade_area_frequency)
        self.frame_linewidth.addItem(shade_area_linewidth)
        
        # Calculate Allan deviation
        fractional_frequency = fitresults[2,:] / np.mean(fitresults[2,:])
        rate = 1/dt
        taus, allandev, allandev_std, ns = allantools.oadev(fractional_frequency, rate, data_type="freq")
        
        # Create scatter item for Allan deviation
        scatter_allandev = pg.ScatterPlotItem(x=taus, y=allandev, pen=pg.mkPen(color='k'))
        
        # Create shaded 1 sigma band for Allan deviation
        line1_allandev = pg.PlotCurveItem(x=taus, y=allandev+allandev_std, pen=pg.mkPen(color='k'))
        line2_allandev = pg.PlotCurveItem(x=taus, y=allandev-allandev_std, pen=pg.mkPen(color='k'))
        shade_area_allandev = pg.FillBetweenItem(curve1=line1_allandev, curve2=line2_allandev, brush=pg.mkBrush(color=(200, 200, 200, 70)))
        
        # Add scatter items and shaded sigma bands to plots
        self.plotted_allandev = window.frame_allandev.addItem(scatter_allandev)
        self.plotted_allandev_std = window.frame_allandev.addItem(shade_area_allandev)
        
        # Create linewidth histogram
        hist, edges = np.histogram(fitresults[3,:], bins="fd") # Number of bins estimated by Freedman Diaconis Estimator
        self.frame_linhist.plot(x=edges, y=hist, stepMode="center", pen=pg.mkPen(color='k'))
        
    
    # Called at mouse click in the frequency and linewidth plot, respectively
    #   - Determine the position of the mouse click in data coordinates  
    #   - Call the plot_clicked_point method
    def signal_frequency(self, event):
        # Get click position
        click_pos = self.frame_frequency.plotItem.vb.mapSceneToView(event.scenePos())
        # Call plot_clicked_point method
        self.plot_clicked_point(click_pos)
    def signal_linewidth(self, event):
        # Get click position
        click_pos = self.frame_linewidth.plotItem.vb.mapSceneToView(event.scenePos())
        # Call plot_clicked_point method
        self.plot_clicked_point(click_pos)
    
    
    # Called by the methods signal_frequency or signal_linewidth
    #   - Get the x index of the data point nearest to the click
    #   - Get the spectrum corresponding to the clicked data point and plot it
    #     along with the Lorentzian fit
    def plot_clicked_point(self, click_pos):
        # Get relevant quantities from datadict
        time_array = self.datadict["time_array"]
        spectra = self.datadict["spectra"]
        fitrange_idx = self.datadict["fitrange_idx"]
        freqs = self.datadict["freqs"]
        fitresults = self.datadict["fitresults"]
        
        # Get x index of data point closest to click
        click_idx = (np.abs(time_array - click_pos.x())).argmin()        

        # Clear previous spectrum plot if it exists
        if hasattr(self, 'plotted_spectrum'):
            self.frame_spectrum.removeItem(self.plotted_spectrum)
    
        # Plot the new spectrum and store the item
        self.plotted_spectrum = self.frame_spectrum.plot(freqs[fitrange_idx[0]:fitrange_idx[1]],\
                                                         spectra[fitrange_idx[0]:fitrange_idx[1],click_idx],\
                                                         pen=pg.mkPen(color='k'))
        
        # Clear previous Lorentzian peak from plot if it exists
        if hasattr(self, 'plotted_peak'):
            self.frame_spectrum.removeItem(self.plotted_peak)
            
        # Plot the fit to the new spectrum
        self.plotted_peak = self.frame_spectrum.plot(freqs[fitrange_idx[0]:fitrange_idx[1]],\
                                                     mod.lorentzian(freqs[fitrange_idx[0]:fitrange_idx[1]], *fitresults[:,click_idx]),\
                                                     pen=pg.mkPen(color='r'))
  

###############################################################################
""" Run the code """
###############################################################################
if __name__ == '__main__':
    
    # Create an instance of QtWidgets.QApplication
    app = QtWidgets.QApplication(sys.argv) 
    
    # Create an instance of the GUI
    window = GUI()
    
    # Start the application
    sys.exit(app.exec_())
