#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 11:51:06 2023

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from hamiltonians import Hamiltonian_A1u_junction_k,\
    Hamiltonian_A1u_S_junction_k, Hamiltonian_S_S_junction_k, \
    Hamiltonian_A1us_S_junction_k, Hamiltonian_ZKM_S_junction_k
from functions import phi_spectrum

# Pauli matrices
sigma_0 = np.eye(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
tau_0 = np.eye(2)
tau_x = np.array([[0, 1], [1, 0]])
tau_y = np.array([[0, -1j], [1j, 0]])
tau_z = np.array([[1, 0], [0, -1]])

#%%

t = 1
t_J = t/2    #t/10
Delta_A1u = t/10     #t/5
# Delta_1 = -0.2*t
# Delta_0 = 0.4*t
# Lambda = t/2
Delta_S = t/40      #t/10

mu = -2*t
# phi_values = np.linspace(0, 2*np.pi, 240)
phi_values = np.linspace(0, 2*np.pi, 10)
# k_values = np.linspace(0, np.pi, 100)
k_values = np.linspace(0, np.pi, 10)

# A1u-S junction
L_A1u = 300      #50
L_S = 100       # 1
L = L_A1u + L_S
params = dict(t=t, mu=mu, Delta_A1u=Delta_A1u,
              L_A1u=L_A1u, L_S=L_S, t_J=t_J,
              Delta_S=Delta_S)

# ZKM-S junction
# L_ZKM = 20
# L_S = 20 
# L = L_ZKM + L_S
# params = dict(t=t, mu=mu,
#               L_ZKM=L_ZKM, L_S=L_S, t_J=t_J,
#               Delta_0=Delta_0, Delta_1=Delta_1,
#               Lambda=Lambda)

# S-S junction
# Delta_S_1 = t/2
# Delta_S_2 = Delta_S_1
# L_S_1 = 10
# L_S_2 = L_S_1
# params = dict(t=t, mu=mu,
#                 L_S_1=L_S_1, L_S_2=L_S_2,
#                 Delta_S_1=Delta_S_1, Delta_S_2=Delta_S_2,
#                 t_J=t_J)

# A1u-A1u junction
# params = dict(t=t, mu=mu, Delta=Delta_A1u,
#                 L=L_A1u, t_J=t_J)

# E_phi = phi_spectrum(Hamiltonian_ZKM_S_junction_k, k_values, phi_values, **params)
# E_phi = phi_spectrum(Hamiltonian_A1u_S_junction_k, k_values, phi_values, **params)
E_phi = phi_spectrum(Hamiltonian_A1us_S_junction_k, k_values, phi_values, **params)
# E_phi = phi_spectrum(Hamiltonian_A1u_junction_k, k_values, phi_values, **params)
# E_phi = phi_spectrum(Hamiltonian_S_S_junction_k, k_values, phi_values, **params)

print('\007')  # Ending bell

#%% Plotting for a given k

fig, ax = plt.subplots()
j = 0   #index of k-value
for i in range(np.shape(E_phi)[2]):
    plt.plot(phi_values, E_phi[j, :, i], ".k", markersize=1)

plt.title(f"k={k_values[j]}")
plt.xlabel(r"$\phi$")
plt.ylabel(r"$E_k$")

#%% Total energy

E_positive = E_phi[:, :, np.shape(E_phi)[2]//2:]
total_energy_k = np.sum(E_positive, axis=2)
total_energy = np.sum(total_energy_k, axis=0) 
phi_eq = phi_values[np.where(min(-total_energy)==-total_energy)]

#%% Josephson current

Josephson_current = np.diff(-total_energy)
Josephson_current_k = np.diff(-total_energy_k)

J_0 = np.max(Josephson_current) 
fig, ax = plt.subplots()
ax.plot(phi_values[:-1]/(2*np.pi), Josephson_current/J_0)
ax.set_xlabel(r"$\phi/(2\pi)$")
ax.set_ylabel(r"$J(\phi)/J_0$")
ax.set_title("Josephson current")

fig, ax = plt.subplots()
ax.set_xlabel(r"$\phi/(2\pi)$")
ax.set_ylabel(r"$J_k(\phi)$")
ax.set_title("Josephson current for given k")

for i, k in enumerate(k_values):
    ax.plot(phi_values[:-1]/(2*np.pi), Josephson_current_k[i,:])

#%% Plotting of total energy

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

def energy(phi_values, E_0, E_J):
    # return E_0*(4*np.cos(phi_eq[0])*(1-np.cos(phi_values))-2*np.sin(phi_values)**2) 
    return E_J*(1-np.cos(phi_values)) - 2*E_0*np.sin(phi_values)**2
popt, pcov = scipy.optimize.curve_fit(energy, xdata = phi_values, ydata = -total_energy+total_energy[0])
E_0 = popt[0]
E_J = popt[1]
fig, ax = plt.subplots()
ax.plot(phi_values/(2*np.pi), -total_energy+total_energy[0], label="Numerical")
ax.plot(phi_values/(2*np.pi), energy(phi_values, E_0, E_J))
ax.set_xlabel(r"$\phi/(2\pi)$")
ax.set_ylabel(r"$E(\phi)$")
ax.set_title(r"$\phi_{0}=$"+f"{(2*np.pi-phi_eq[0])/(2*np.pi):.2}"+r"$\times 2\pi$")

# plt.legend()
plt.tight_layout()