#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 16:42:01 2023

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from phase_functions import phase_single_soliton_arctan, phase_single_soliton_S, phase_soliton_antisoliton_arctan

L_x = 200
L_y = 200       #L_y should be odd for single soliton
t = 1
Delta = 1
mu = -2  #-2
t_J = t/2   #t
L = 100      #L_y//2
k = 12 #number of eigenvalues
lambda_J = 10
# phi_profile = phase_single_soliton_arctan
phi_external = 0
y = np.arange(1, L_y+1)
y_0 = (L_y-L)//2
y_1 = (L_y+L)//2
y_s = (L_y+1)//2

# Phi = phase_single_soliton_arctan(phi_external, y, y_s, lambda_J)
Phi = phase_soliton_antisoliton_arctan(phi_external, y, y_0, y_1, lambda_J)
#%% Phase soliton
plt.rc("font", family="serif")  # set font family
plt.rc("xtick", labelsize=18)  # reduced tick label size
plt.rc("ytick", labelsize=18)
plt.rc('font', size=18) #controls default text size
plt.rc('axes', titlesize=18) #fontsize of the title
plt.rc('axes', labelsize=18) #fontsize of the x and y labels
plt.rc("text", usetex=True) # for better LaTex (slower)
plt.rcParams['xtick.top'] = True    #ticks on top
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['ytick.right'] = True    #ticks on left
plt.rcParams['ytick.labelright'] = False
plt.rc('legend', fontsize=18) #fontsize of the legend

fig, ax = plt.subplots()
ax.plot(y, Phi/(2*np.pi))
# ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\phi/2\pi$")
ax.set_xticklabels(ax.get_xticks(), rotation = 90)
ax.set_xticks([y_s], [r"$y_0$"])
plt.tight_layout()

#%% Mass soliton

fig, ax = plt.subplots()
# ax.plot(y, -np.tanh((y-y_s)/lambda_J))
ax.plot(y, np.cos(Phi/2))
# ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$m(\phi)/m_0$")
ax.set_xticks([y_s], [r"$y_0$"])
plt.tight_layout()

#%% Mass and Phase

fig, ax = plt.subplots()
ax.plot(y, Phi/(2*np.pi), "b", label=r"$\phi/(2\pi)$")
ax2 = ax.twinx()
ax2.plot(y, -np.tanh((y-y_s)/lambda_J), "r--", label=r"$m(\phi)/m_0$")
# ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\phi/2\pi$")
ax2.set_ylabel(r"$m(\phi)/m_0$", rotation=270)
ax.set_xticks([y_s], [r"$x_0$"])
ax.legend(loc="center left")
ax2.legend(loc="center right")

plt.tight_layout()

#%% Phase TRITOPS-S

y = np.linspace(-100, 100, 1000)
y_0 = 0
lambda_J = 10
phi_external = 0
phi_0 = 0.14*2*np.pi


Phi = phase_single_soliton_S(phi_external=phi_external, y=y, y_0=y_0, lambda_J=lambda_J, phi_0=phi_0)

fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(y, Phi[0]/(2*np.pi), "b", label=r"$\phi_1/(2\pi)$")
ax.plot(y, Phi[1]/(2*np.pi), "r", label=r"$\phi_2/(2\pi)$")
ax.set_ylabel(r"$\phi/(2\pi)$")
ax2.set_ylabel(r"$m(\phi)/m_0$", rotation=270)
ax2.plot(y, np.sin(Phi[0]), "b--", label=r"$m(\phi_1)/m_0$")
ax2.plot(y, np.sin(Phi[1]), "r--", label=r"$m(\phi_2)/m_0$")
ax.set_xticks([0], [r"$x_0$"])
ax.legend(loc="center left")
ax2.legend()
ax2.legend(loc="center right")

plt.tight_layout()

#%% Localized states soliton
x = np.linspace(-1, 1, 1000)
kappa = 5
fig, ax = plt.subplots()
ax.plot(x, np.exp(-2*kappa*np.abs(x)))
ax.set_xticks([0], [r"$0$"])
ax.set_yticks([1], [r"$|C|^2$"])
ax.set_xlabel("x")
# ax.plot(1/(2*kappa), np.exp(-2*kappa*np.abs(1/(2*kappa))), "o")
# ax.plot(-1/(2*kappa), np.exp(-2*kappa*np.abs(-1/(2*kappa))), "o")
plt.arrow(-1/(2*kappa), np.exp(-2*kappa*np.abs(-1/(2*kappa))), 1/kappa, 0, color="k", width= 0.01, length_includes_head=True)
plt.arrow(1/(2*kappa), np.exp(-2*kappa*np.abs(-1/(2*kappa))), -1/kappa, 0, color="k", width= 0.01, length_includes_head=True)
plt.text(-0.07, np.exp(-2*kappa*np.abs(-1/(2*kappa)))-0.08, r"$1/\kappa$", fontsize=18)
plt.tight_layout()

#%% Localized states soliton-antisoliton
from scipy.optimize import root
def trascendental_equation(k, m_0, Delta, L):
    """
    Wavevector of the localized state.

    Parameters
    ----------
    m_0 : float
        Mass.
    Delta : floar
        Gap.
    L : float
        Length.

    Returns
    -------
    A function whose roots represents the trascendental equation.
        (m_0/Delta)**2 - k**2 - (m_0/Delta)**2 * np.exp(-2*k*L)=0
    """
    return (m_0/Delta)**2 - k**2 - (m_0/Delta)**2 * np.exp(-2*k*L)

def Kappa(m_0, Delta, L):
    """
    Wavevector of the localized state.

    Parameters
    ----------
    (y, kappa, m_0, Delta, L)m_0 : float
        Mass.
    Delta : floar
        Gap.
    L : float
        Length.

    Returns
    -------
    The wavevector k solving:
        (m_0/Delta)**2 - k**2 = (m_0/Delta)**2 * np.exp(-2*k*L)
    """
    return root(trascendental_equation, 1, args=(m_0, Delta, L)).x

L = 3
x = np.linspace(-5, L+5, 1000)
kappa = float(Kappa(1, 1, L))

def density(x):
    if x<=0:
        return kappa**2*np.exp(2*kappa*(x-L))
    elif 0<x<L:
        return (-(kappa+1)*np.exp(kappa*(x-L)) + np.exp(-kappa*(x+L)))**2
    else:
        return (-(kappa+1)*np.exp(kappa*L) + np.exp(-kappa*L))**2 * np.exp(-2*kappa*x)

fig, ax = plt.subplots()
ax.plot(x, [density(x_0) + density(-x_0 + L) for x_0 in x])
ax.set_xticks([0, L], [r"$0$", "L"])
ax.set_yticks([density(L)], [""])
ax.set_xlabel("x")
plt.tight_layout()

#%% Todo junto

fig, ax = plt.subplots(2)
ax[0].plot(x, np.exp(-2*kappa*np.abs(x)))
ax[0].set_xticks([0], [r"$0$"])
ax[0].set_yticks([1], [r"$|C|^2$"])
ax[0].set_xlabel("x")
ax[0].arrow(-1/(2*kappa), np.exp(-2*kappa*np.abs(-1/(2*kappa))), 1/kappa, 0, color="k", width= 0.01, length_includes_head=True)
ax[0].arrow(1/(2*kappa), np.exp(-2*kappa*np.abs(-1/(2*kappa))), -1/kappa, 0, color="k", width= 0.01, length_includes_head=True)
ax[0].text(-0.3, np.exp(-2*kappa*np.abs(-1/(2*kappa)))-0.2, r"$1/\kappa$", fontsize=12)

ax[1].plot(x, [density(x_0) + density(-x_0 + L) for x_0 in x])
ax[1].set_xticks([0, L], [r"$0$", "L"])
ax[1].set_yticks([density(L)], [""])
ax[1].set_xlabel("x")
plt.tight_layout()