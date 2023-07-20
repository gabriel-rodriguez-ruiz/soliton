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
    Hamiltonian_A1u_S_junction_k, Hamiltonian_A1u_S_1D_junction_k

# Pauli matrices
sigma_0 = np.eye(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
tau_0 = np.eye(2)
tau_x = np.array([[0, 1], [1, 0]])
tau_y = np.array([[0, -1j], [1j, 0]])
tau_z = np.array([[1, 0], [0, -1]])

def phi_spectrum(Junction, k_values, phi_values, **params):
    """Returns an array whose rows are the eigenvalues of the junction (with function Junction) for
    a definite phi_value given a fixed k_value.
    """
    eigenvalues = []
    for k_value in k_values:
        eigenvalues_k = []
        params["k"] = k_value
        for phi in phi_values:
            params["phi"] = phi
            H = Junction(**params)
            energies = np.linalg.eigvalsh(H)
            energies = list(energies)
            eigenvalues_k.append(energies)
        eigenvalues.append(eigenvalues_k)
    eigenvalues = np.array(eigenvalues)
    return eigenvalues

#%%

t = 1
t_J = t/10
Delta = t
mu = -2*t
phi_values = np.linspace(0, 2*np.pi, 240)
#phi = np.linspace(0, 2*np.pi, 750)
#k = np.linspace(0, np.pi, 75)
k_values = np.linspace(0, 2*np.pi, 200)
# k_values = np.linspace(0, 0.01*np.pi, 200)

#k = np.array([0, 0.01, 0.02])*np.pi
#k = np.linspace(-3, -, 5)

L = 10

params = dict(t=t, mu=mu, Delta=Delta,
              L=L, t_J=t_J)

# E_phi = phi_spectrum(Hamiltonian_A1u_S_junction_k, k_values, phi_values, **params)
E_phi = phi_spectrum(Hamiltonian_A1u_S_1D_junction_k, k_values, phi_values, **params)

print('\007')  # Ending bell

#%% Plotting for a given k

j = 5   #index of k-value
for i in range(4*L):
    plt.plot(phi_values, E_phi[j, :, i], ".k", markersize=1)

plt.title(f"k={k_values[j]}")
plt.xlabel(r"$\phi$")
plt.ylabel(r"$E_k$")

#%% Total energy

E_positive = E_phi[:, :, 2*L:]
total_energy_k = np.sum(E_positive, axis=2)
total_energy = np.sum(total_energy_k, axis=0) 
phi_eq = phi_values[np.where(min(-total_energy)==-total_energy)]

#%% Josephson current

Josephson_current = np.diff(-total_energy)
J_0 = np.max(Josephson_current) 
fig, ax = plt.subplots()
ax.plot(phi_values[:-1]/(2*np.pi), Josephson_current/J_0)
ax.set_xlabel(r"$\phi/(2\pi)$")
ax.set_ylabel(r"$J(\phi)/J_0$")
ax.set_title("Josephson current")

#%% Plotting of total energy
def energy(phi_values, E_0):
    return E_0*(4*np.cos(phi_eq[0])*(1-np.cos(phi_values))-2*np.sin(phi_values)**2) 

E_0 = scipy.optimize.curve_fit(energy, xdata = phi_values, ydata = -total_energy+total_energy[0])[0]
fig, ax = plt.subplots()
ax.plot(phi_values/(2*np.pi), -total_energy+total_energy[0], label="Numerical")
ax.plot(phi_values/(2*np.pi), energy(phi_values, E_0), label=f"Analytical E_0={E_0[0]:.2}" )

ax.set_xlabel(r"$\phi/(2\pi)$")
ax.set_ylabel(r"$E(\phi)$")
ax.set_title(r"$\phi_{eq}=$"+f"{phi_eq[0]/(2*np.pi):.2}"+r"$\times 2\pi$")

plt.legend()