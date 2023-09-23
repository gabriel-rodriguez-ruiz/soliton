# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 08:08:28 2023

@author: gabri
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rc("font", family="serif")  # set font family
plt.rc("xtick", labelsize="large")  # reduced tick label size
plt.rc("ytick", labelsize="large")
plt.rc('font', size=18) #controls default text size
plt.rc('axes', titlesize=18) #fontsize of the title
plt.rc('axes', labelsize=18) #fontsize of the x and y labels
plt.rc("text", usetex=True) # for better LaTex (slower)
plt.rcParams['xtick.top'] = True    #ticks on top
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['ytick.right'] = True    #ticks on left
plt.rcParams['ytick.labelright'] = False
plt.rc('legend', fontsize=18) #fontsize of the legend


def Energy(k, phi):
    return np.sqrt(k + np.cos(phi/2)**2)

fig, ax = plt.subplots()
k = np.linspace(0, 1, 11)
phi = np.linspace(0, 2*np.pi, 1000)
for k in k:
    ax.plot(phi/np.pi, Energy(k,phi), "k")
    # ax.plot(phi/np.pi, -Energy(k,phi))
    ax.plot(phi/np.pi, -Energy(0,phi), "c")
    ax.plot(phi/np.pi, -Energy(0.1,phi), "m")
    ax.plot(phi/np.pi, -Energy(0.2,phi), "m")
    ax.plot(phi/np.pi, -Energy(0.3,phi), "m")
    ax.plot(phi/np.pi, -Energy(0.4,phi), "m")
    ax.plot(phi/np.pi, -Energy(0.5,phi), "m")
    ax.plot(phi/np.pi, -Energy(0.6,phi), "r")
    ax.plot(phi/np.pi, -Energy(0.7,phi), "r")
    ax.plot(phi/np.pi, -Energy(0.8,phi), "r")
    ax.plot(phi/np.pi, -Energy(0.9,phi), "r")
    ax.plot(phi/np.pi, -Energy(1,phi), "r")

plt.tight_layout()
    
ax.set_ylabel(r"$\epsilon_k(\phi)/t_J$")
ax.set_xlabel(r"$\phi/\pi$")