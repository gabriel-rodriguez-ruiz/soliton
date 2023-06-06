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

def localized_solution(y):
    """
    Solution of the effective Hamiltonian localized at y=0.
    I assume Delta=1.

    Parameters
    ----------
    y : float
        Coordinate along the junction.

    Returns
    -------
    None.
    The solution at y.
    """
    xi = mass(Phi(epsilon, y), t_J)
    return np.exp(xi*y)

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

#%% Plotting of the effective solution
y = np.linspace(-100, 100, 10000)
fig, ax = plt.subplots()
ax.plot(y, [localized_solution(y_value) for y_value in y])
ax.set_xlabel("y")
ax.set_ylabel(r"$\Psi(y)$")
ax.text(-50, 0.75, r"$e^{\frac{t_J}{\Delta}\sin(\epsilon/2)y}$")
ax.text(50, 0.75, r"$e^{-\frac{t_J}{\Delta}\sin(\epsilon/2)y}$")
ax.arrow(-50, 0.75, -10-(-50), localized_solution(-10)-0.75)
ax.arrow(50, 0.75, 10-(50), localized_solution(10)-0.75)
ax.set_yticks([])
ax.set_xticks([0],["0"])
fig.tight_layout()

#%% Plotting of mass with tanh

epsilon = 0.1
t_J = 1
mu = 1
y = np.linspace(0, 10)
fig, ax = plt.subplots()
ax.plot(y, [-t_J*np.sin( (epsilon*np.tanh(mu/2*(y_value-5)))/2 ) for y_value in y])

#%%

def mass(x, x_1, x_2):
    return (np.sinh(x-x_1)*np.sinh(x-x_2)-1)/(np.cosh(x-x_1)*np.cosh(x-x_2))

x = np.linspace(0, 100, 1000)
plt.figure()
plt.plot(x, [mass(x, 25, 75) for x in x])