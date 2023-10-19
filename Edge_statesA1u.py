# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 14:12:52 2023

@author: gabri
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
from hamiltonians import Hamiltonian_A1u_semi_infinite, Hamiltonian_A1us_k, Hamiltonian_A1us_S_junction_k
from functions import spectrum

t = 1
Delta = t/10  #1
mu = -2*t     #mu = -3  entre -4t y 4t hay estados de borde
k = np.linspace(0, np.pi, 100)
L_x = 200  #200

params = dict(t=t, mu=mu, Delta=Delta,
              L_x=L_x)

Delta_0 = t/10
t_J = t/2
# spectrum_A1u = spectrum(Hamiltonian_A1u_semi_infinite, k, **params)
spectrum_A1u = spectrum(Hamiltonian_A1us_k, k, t=t, mu=mu, L=L_x, Delta_A1u=Delta, Delta_S=Delta_0)
# spectrum_A1u = spectrum(Hamiltonian_A1us_S_junction_k, k, t=t, mu=mu, L_A1u=L_x//2, L_S=L_x//2, Delta_A1u=Delta, Delta_S=Delta_0, phi=0, t_J=t_J)

# H = Hamiltonian_A1u_semi_infinite(k=0.1, **params)
# eigenvalues, eigenvectors = np.linalg.eigh(H)


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
    k[:45], spectrum_A1u[:45, 2*L_x-2:2*L_x+2], linewidth=1, color="c"
)  # each column in spectrum is a separate dataset

ax.set_xlim((0, np.pi))
ax.set_xticks(np.arange(0, 1.2, step=0.2) * np.pi)
ax.set_xticklabels(
    ["0"] + list(np.array(np.round(np.arange(0.2, 1, step=0.2), 1), dtype=str)) + ["1"])
ax.set_xticks(np.arange(0, 1.1, step=0.1) * np.pi, minor=True)
ax.set_yticks(np.arange(-6, 7, step=2))
ax.set_yticks(np.arange(-6, 7, step=1), minor=True)
ax.set_xlabel(r"$k_y/\pi$")
ax.set_ylabel(r"$E(k_y)$")
ax.set_ylim((-3, 3))

plt.tight_layout()

#%% Spectrum vs mu

k = [0]
mu = np.linspace(-4, 0, 100)
spectrum_mu = []
for mu_value in mu:
    spectrum_mu.append(spectrum(Hamiltonian_A1us_k, k, t=t, mu=mu_value, L=L_x, Delta_A1u=Delta, Delta_S=Delta_0))

fig, ax = plt.subplots()
ax.plot(mu, [spectrum_mu[i][0,:] for i in range(len(mu))])
ax.set_xlabel(r"$\mu/t$")
ax.set_ylabel(r"$E$")