#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:53:46 2024

@author: aaron
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Define plot layout
plt.rcParams['text.latex.preamble'] = r'\usepackage{/usr/local/texlive/2023/bin/universal-darwin/latex}'
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["text.usetex"] = True


plt.plot(np.arange(1,100,1), np.arange(1,100,1))
plt.xlabel("Time (s)")
plt.ylabel("Beat frequency (MHz)")
plt.show()
