#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 12:05:09 2023

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from hamiltonians import Hamiltonian_A1u_S, Zeeman
from functions import get_components
import scipy
from functions import mean_spin, mean_spin_xy, get_components

L_x = 20
L_y = 400
t = 1
Delta = 1
mu = -2
Phi = np.pi/2   #superconducting phase
t_J = 1    #t/2
k = 4
theta = np.pi/2
phi = np.pi/2
Delta_Z = 0
index = 1
L_y_values = np.linspace(100, 2000, 20, dtype=int)
spin_values = []

for i,L_y in enumerate(L_y_values):
    H = Hamiltonian_A1u_S(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi) + Zeeman(theta=theta, phi=phi, Delta_Z=Delta_Z, L_x=L_x, L_y=L_y)
    eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H, k=k, sigma=0) 
    zero_modes = eigenvectors_sparse      #4 (2) modes with zero energy (with Zeeman)
    creation_up, creation_down, destruction_down, destruction_up = get_components(zero_modes[:,index], L_x, L_y)
    zero_state = np.stack((destruction_up, destruction_down, creation_down, creation_up), axis=2) #positive energy eigenvector splitted in components
    # Spin mean value
    spin_values.append(mean_spin_xy(zero_state))
    
fig, ax = plt.subplots()
ax.plot(L_y_values, [np.sum(spin_values[i][:,:,1]) for i in range(len(L_y_values))])
#ax.set_title("Total spin in y-direction", fontsize=18)
ax.set_ylabel(r"$S_y$", fontsize=18)
ax.set_xlabel(r"$L_y$", fontsize=18)
plt.tight_layout()
