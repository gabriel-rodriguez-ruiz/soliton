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
k = 12   #number of eigenvalues
Delta_Z = 0
theta = np.pi/2
phi = 0

H = Hamiltonian_soliton_A1u_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi, L=L)
#H = Hamiltonian_A1u_single_step_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi)
#H = (Hamiltonian_A1u_sparse(t, mu, L_x, L_y, Delta) + Zeeman(theta=theta, Delta_Z=Delta_Z, L_x=L_x, L_y=L_y, phi=phi))

eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H, k=k, sigma=0) 

#%% Plotting vs. Phi

Phi_values = np.linspace(0.63, 0.65, 30)
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

for t_J_value in t_J_values:
    H = Hamiltonian_soliton_A1u_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J_value, Phi=Phi, L=L)
    eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H, k=k, sigma=0) 
    eigenvalues_sparse.sort()
    eigenvalues.append(eigenvalues_sparse)
    
fig, ax = plt.subplots()
ax.plot(t_J_values, [eigenvalues[i][0::2] for i in range(len(t_J_values))], "o", alpha=0.1)
ax.plot(t_J_values, [eigenvalues[i][0::1] for i in range(len(t_J_values))], "*", alpha=0.1)
ax.set_xlabel(r"$t_J$")
plt.yscale('log')

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
ax.plot(L_y_values, [eigenvalues[i][0::2] for i in range(len(L_y_values))], "o", alpha=0.1)
ax.plot(L_y_values, [eigenvalues[i][0::1] for i in range(len(L_y_values))], "*", alpha=0.1)
ax.set_xlabel(r"$L_y$")
plt.yscale('log')
