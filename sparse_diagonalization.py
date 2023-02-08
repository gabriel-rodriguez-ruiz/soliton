#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:44:35 2023

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from hamiltonians import Hamiltonian_soliton_A1u, Hamiltonian_soliton_A1u_sparse
from functions import probability_density, get_components
import scipy

L_x = 50
L_y = 200
t = 1
Delta = 1
mu = -1  #-2
Phi = np.pi   #superconducting phase
t_J = 1    #t/2
index = 1   #which zero mode
L = 100
H = Hamiltonian_soliton_A1u_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi, L=L)
eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H, k=4, sigma=0) 

#%% Spin determination
from functions import mean_spin_xy, get_components

zero_modes = eigenvectors_sparse
creation_up, creation_down, destruction_down, destruction_up = get_components(zero_modes[:,index], L_x, L_y)
zero_state = np.stack((destruction_up, destruction_down, creation_down, creation_up), axis=2) #positive energy eigenvector splitted in components
# corner_state = zero_plus_state[L_x-2, L_y//2, :].reshape(4,1)  #positive energy point state localized at the junction
# corner_state_normalized = corner_state/np.linalg.norm(corner_state[:2]) #normalization with only particle part
zero_state_normalized = zero_state/np.linalg.norm(zero_state)
# Spin mean value
# spin_mean_value = mean_spin(corner_state_normalized)

spin = mean_spin_xy(zero_state_normalized)

# fig, ax = plt.subplots()
# image = ax.imshow(spin[:,:,2].T, cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
# plt.colorbar(image)
#image.set_clim(np.min(spin[:,:,1].T), np.max(spin[:,:,1].T))

# Meshgrid
x, y = np.meshgrid(np.linspace(0, L_x-1, L_x), 
                    #np.linspace(L_y-1, 0, L_y))
                    np.linspace(0, L_y-1, L_y))


  
# Directional vectors
u = spin[:, :, 0]   #x component
v = spin[:, :, 1]   #y component

# Plotting Vector Field with QUIVER
fig, ax = plt.subplots()
ax.quiver(x, y, u, v, color='r', angles='uv')
ax.set_title('Spin Field in the plane')

#%% Spin in z
fig, ax = plt.subplots()
image = ax.imshow(spin[:,:,2], cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
plt.colorbar(image)
ax.set_title('Spin Field in the z direction')

fig, ax = plt.subplots()
ax.plot(spin[:, L_x//2,2])
total_spin = np.sum(spin[:, L_x//2, 2])
plt.text(0,0.5, f"Total spin={total_spin}")
ax.set_title('Spin Field in the z direction')