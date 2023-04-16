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

def mass(phi, t_J):
    """
    Mass term.    

    Parameters
    ----------
    phi : float
        Phase difference for a given y.
    t_J : float
        Hopping parameter.
    Returns
    -------
    The mass term.

    """
    return t_J*np.cos(phi/2)

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

#%% Plotting of the mass
t_J = 1

fig, ax = plt.subplots()
ax.plot(y, [mass(Phi(epsilon, y_value), t_J) for y_value in y])
ax.set_xlabel("y")
ax.set_ylabel(r"$m(y)$")
ax.set_yticks([mass(np.pi-epsilon, t_J), 0, mass(np.pi+epsilon, t_J)],
              [r"$t_J\sin(\epsilon/2)$", r"0", r"$-t_J\sin(\epsilon/2)$"])
ax.set_xticks([0],["0"])
fig.tight_layout()