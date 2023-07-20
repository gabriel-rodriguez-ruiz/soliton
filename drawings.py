#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 16:42:01 2023

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from phase_functions import phase_single_soliton_arctan, phase_single_soliton_S

L_x = 200
L_y = 200       #L_y should be odd for single soliton
t = 1
Delta = 1
mu = -2  #-2
t_J = t/2   #t
L = 30      #L_y//2
k = 12 #number of eigenvalues
lambda_J = 10
# phi_profile = phase_single_soliton_arctan
phi_external = 0
y = np.arange(1, L_y+1)
y_0 = (L_y-L)//2
y_1 = (L_y+L)//2
y_s = (L_y+1)//2

Phi = phase_single_soliton_arctan(phi_external, y, y_s, lambda_J)

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
ax.set_xticks([y_s], [r"$x_0$"])
plt.tight_layout()

#%% Mass soliton

fig, ax = plt.subplots()
ax.plot(y, -np.tanh((y-y_s)/lambda_J))
# ax.plot(y, np.cos(Phi/2))
# ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$m(\phi)/m_0$")
ax.set_xticks([y_s], [r"$x_0$"])
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
phi_0 = 0.25*2*np.pi


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
ax.legend()
ax2.legend()
ax2.legend(loc="center right")

plt.tight_layout()
