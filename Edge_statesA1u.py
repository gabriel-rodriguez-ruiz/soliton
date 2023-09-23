# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 14:12:52 2023

@author: gabri
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
from hamiltonians import Hamiltonian_A1u_semi_infinite

t = 1
Delta = t/2  #1
mu = -2*t     #mu = -3  entre -4t y 4t hay estados de borde
k = np.linspace(0, np.pi, 150)
L_x = 200  #200

params = dict(t=t, mu=mu, Delta=Delta,
              L_x=L_x)

def spectrum(system, k_values, **params):
    """Returns an array whose rows are the eigenvalues of the system for
    a definite k. System should be a function that returns an array.
    """
    eigenvalues = []
    for k in k_values:
        params["k"] = k
        H = system(**params)
        energies = np.linalg.eigvalsh(H)
        energies = list(energies)
        eigenvalues.append(energies)
    eigenvalues = np.array(eigenvalues)
    return eigenvalues

spectrum_A1u = spectrum(Hamiltonian_A1u_semi_infinite, k, **params)

#%% Plotting of spectrum
# plt.close()

plt.rc("font", family="serif")  # set font family
plt.rc("xtick", labelsize="small")  # reduced tick label size
plt.rc("ytick", labelsize="small")
plt.rc("text", usetex=True) # for better LaTex (slower)
plt.rcParams['xtick.top'] = True    #ticks on top
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['ytick.right'] = True    #ticks on left
plt.rcParams['ytick.labelright'] = False


fig, ax = plt.subplots(figsize=(4, 3))
# fig.set_figwidth(246/72)    # in inches, \columnwith=246pt and 1pt=1/72 inch
ax.plot(
    k, spectrum_A1u, linewidth=0.1, color="m"
)  # each column in spectrum is a separate dataset
ax.plot(
    k[:45], spectrum_A1u[:45, 398:402], linewidth=1, color="c"
)  # each column in spectrum is a separate dataset

ax.set_ylim((-7, 7))
ax.set_xlim((0, np.pi))
ax.set_xticks(np.arange(0, 1.2, step=0.2) * np.pi)
ax.set_xticklabels(
    ["0"] + list(np.array(np.round(np.arange(0.2, 1, step=0.2), 1), dtype=str)) + ["1"])
ax.set_xticks(np.arange(0, 1.1, step=0.1) * np.pi, minor=True)
ax.set_yticks(np.arange(-6, 7, step=2))
ax.set_yticks(np.arange(-6, 7, step=1), minor=True)
ax.set_xlabel(r"$k_y/\pi$")
ax.set_ylabel(r"$E(k_y)$")

plt.tight_layout()