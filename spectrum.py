#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:01:39 2023

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from hamiltonians import Hamiltonian_soliton_A1u, Hamiltonian_soliton_A1u_sparse, Hamiltonian_A1u_single_step_sparse, Hamiltonian_A1u_sparse, Zeeman
from functions import get_components
import scipy

L_x = 120
L_y = 120
t = 1
Delta = 1
mu = -1  #-2
Phi = np.pi  #superconducting phase
t_J = 1    #t/2
L = L_y//2
k = 8   #number of eigenvalues
Delta_Z = 0
theta = np.pi/2
phi = 0

#H = Hamiltonian_soliton_A1u_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi, L=L)
H = Hamiltonian_A1u_single_step_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi)
#H = (Hamiltonian_A1u_sparse(t, mu, L_x, L_y, Delta) + Zeeman(theta=theta, Delta_Z=Delta_Z, L_x=L_x, L_y=L_y, phi=phi))

eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H, k=k, sigma=0) 

#%% Plotting vs. Phi

Phi_values = np.linspace(0, np.pi, 30)
eigenvalues = []

for Phi_value in Phi_values:
    H = Hamiltonian_soliton_A1u_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi_value, L=L)
    eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H, k=k, sigma=0) 
    eigenvalues_sparse.sort()
    eigenvalues.append(eigenvalues_sparse)
    
fig, ax = plt.subplots()
ax.plot(Phi_values, [eigenvalues[i][0::2] for i in range(len(Phi_values))], "o", alpha=0.1)
ax.plot(Phi_values, [eigenvalues[i][0::1] for i in range(len(Phi_values))], "*", alpha=0.1)
plt.yscale('log')
ax.set_xlabel(r"$\Phi$")
ax.set_ylabel(r"$E$")
#%% Plotting vs. t_J

t_J_values = np.linspace(0, 1, 10)
eigenvalues = []
index = np.arange(k)   #which zero mode (less than k)
probability_density = []
zero_state = []
localized_state_up = [] # it is the site ((L_x+L)/2, L_y/2)
localized_state_down = []

for t_J_value in t_J_values:
    H = Hamiltonian_soliton_A1u_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J_value, Phi=Phi, L=L)
    eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H, k=k, sigma=0) 
    eigenvalues_sparse.sort()
    eigenvalues.append(eigenvalues_sparse)
    destruction_up, destruction_down, creation_down, creation_up = get_components(eigenvectors_sparse[:,0], L_x, L_y)
    probability_density.append((np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)/(np.linalg.norm(np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)))
    zero_state.append(np.stack((destruction_up, destruction_down, creation_down, creation_up), axis=2)) #positive energy eigenvector splitted in components

#%% Plotting vs. tJ
    
fig, ax = plt.subplots()
ax.plot(t_J_values, [eigenvalues[i][0::2] for i in range(len(t_J_values))], "o", alpha=0.5, markersize=10)
ax.plot(t_J_values, [eigenvalues[i][0::1] for i in range(len(t_J_values))], "*", alpha=1, markersize=5)
ax.set_xlabel(r"$t_J$")
ax.set_ylabel(r"$E$")
plt.yscale('log')

fig, ax = plt.subplots()
for k in [1,3,5,7,9]:
    ax.plot(probability_density[k][:, L_x//2], label=r"$t_J=$"+f"{t_J_values[k].round(2)}")
ax.set_xlabel("y")
ax.set_ylabel("Probability density")
ax.text(5,25, rf'$index={index}; \Phi={np.round(Phi, 2)}$')
ax.set_title("Probability density at the junction")
ax.legend()
#%% Plotting vs. Ly

L_y_values = np.linspace(50, 200, 11, dtype=int)
eigenvalues = []

for L_y_value in L_y_values:
    L_x = int(L_y_value)
    L = int(L_y_value//2)
    H = Hamiltonian_soliton_A1u_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y_value, Delta=Delta, t_J=t_J, Phi=Phi, L=L)
    eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H, k=k, sigma=0) 
    eigenvalues_sparse.sort()
    eigenvalues.append(eigenvalues_sparse)
    
fig, ax = plt.subplots()
ax.plot(L_y_values, [eigenvalues[i][0::2] for i in range(len(L_y_values))], "o", alpha=0.5, markersize=10)
ax.plot(L_y_values, [eigenvalues[i][0::1] for i in range(len(L_y_values))], "*", alpha=0.5, markersize=5)
ax.set_xlabel(r"$L_y$")
plt.yscale('log')
ax.set_ylabel("E")
