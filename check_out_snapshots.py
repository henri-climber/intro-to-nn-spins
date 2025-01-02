#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 16:04:36 2021

@author: annabellebohrdt
"""

# !/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 15:26:38 2017
Convolutional Neural Network
@author: annabelle
"""

import matplotlib.pyplot as plt
import numpy as np
# from batch_creator import *
from loadSnapshots import *
from matplotlib.colors import ListedColormap

# hot = hot HF + holes sprinkled in
# cases=['hf','exp','sprinkl','AS','rand','pi','hot']

# experiment, sprinkling, string theory, pi flux
cases = ['exp', 'sprinkl', 'AS', 'pi']
#cases = ['AS', 'pi']
# cases=['AS','rand']
#cases = ['AS']

dopings = [2.0, 3.0, 4.5, 6.0, 7.0, 9.0, 10.0, 12.5, 14.0, 17.0, 20.0, 25.0, 32.0]
# dopings=[2.0,3.0,4.5,6.0,7.0,9.0,10.0,12.5,14.0,20.0,25.0]
# dopings=[6.0]

doping = 6.0

## set limit to the maximum number of snapshots available in EACH class (number of snapshots might vary from class to class)
lim = 5867  # exp data: 5867 #qmcFI: 6800 #AS: 5867 #qmc: 5000/6800# #exp 9%: 2059/2017#hot: 2848/2891# AS: 5326 #pi_Ts:5200
sample, labels, sms_1 = loading(cases, doping, lim, visualize=False)

# Samples is a list of numpy arrays, each array is a 10x10 image but in the form of 100 elements (1D array)
# please make an image out of it
# Define custom colormap
cmap = ListedColormap(['red', 'white', 'blue'])  # Assign colors for -1, 0, 1 respectively

# Display images with custom colormap
for i in range(10):
    plt.imshow(sample[i].reshape(10, 10), cmap=cmap, interpolation='nearest')
    plt.colorbar()  # Optional: to show the color scale
    plt.show()




print(str(len(sample)) + ' snapshots loaded!')

