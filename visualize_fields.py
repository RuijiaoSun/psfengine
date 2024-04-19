# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:52:14 2024

@author: david
"""

import matplotlib.pyplot as plt
import numpy as np

def DisplayField(E):
    

    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(np.real(E[:, :, 0]))
    plt.title("X polarization [Real]")
    plt.colorbar();
    plt.subplot(2, 3, 2)
    plt.imshow(np.real(E[:, :, 1]))
    plt.title("Y polarization [Real]")
    plt.colorbar()
    plt.subplot(2, 3, 3)
    plt.imshow(np.real(E[:, :, 2]))
    plt.title("Z polarization [Real]")
    plt.colorbar()
    
    plt.subplot(2, 3, 4)
    plt.imshow(np.imag(E[:, :, 0]))
    plt.title("X polarization [Imag]")
    plt.colorbar();
    plt.subplot(2, 3, 5)
    plt.imshow(np.imag(E[:, :, 1]))
    plt.title("Y polarization [Imag]")
    plt.colorbar()
    plt.subplot(2, 3, 6)
    plt.imshow(np.imag(E[:, :, 2]))
    plt.title("Z polarization [Imag]")
    plt.colorbar()
    
DisplayField(BackAperture)
DisplayField(FrontAperture)