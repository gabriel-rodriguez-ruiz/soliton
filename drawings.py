#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 16:42:01 2023

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt

def Phi(epsilon, y):
    r"""
    Phase difference due to the magnetic flux.

    Parameters
    ----------
    epsilon : float
        Flux around pi.
        
    y : float
        Coordinate along the junction.

    Returns
    -------
    float
        The phase difference.

    """
    return np.pi + np.sign(y)*epsilon

def mass(phi, y):
    """
    Mass term.    

    Parameters
    ----------
    phi : float
        Phase difference.
    y : float
        Coordinate along the junction.

    Returns
    -------
    None.

    """

#%% Plotting of flux

epsilon = 0.04*np.pi
y = np.linspace(-1, 1)

fig, ax = plt.subplots()
ax.plot(y, [Phi(epsilon, y_value) for y_value in y])
ax.set_xlabel("y")
ax.set_ylabel(r"$\phi(y)$")
ax.set_yticks([np.pi-epsilon, np.pi, np.pi+epsilon],
              [r"$\pi-\epsilon$", r"$\pi$", r"$\pi+\epsilon$"])
ax.set_xticks([0],["0"])
fig.tight_layout()
