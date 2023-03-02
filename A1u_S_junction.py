#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 12:14:56 2023

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from hamiltonians import Hamiltonian_A1u_S, Zeeman, Hamiltonian_A1u_S_double_soliton
from functions import get_components
import scipy

L_x = 120
L_y = 120
t = 1
Delta = 1
mu = -2
Phi = np.pi/2   #superconducting phase
t_J = 1    #t/2
k = 4
theta = np.pi/2
phi = np.pi/2
Delta_Z = 0
L = L_y//2

#H = Hamiltonian_A1u_S(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi) + Zeeman(theta=theta, phi=phi, Delta_Z=Delta_Z, L_x=L_x, L_y=L_y)
H = Hamiltonian_A1u_S_double_soliton(t, mu, L_x, L_y, Delta, t_J, Phi, L)
eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H, k=k, sigma=0) 

#%% Probability density
index = np.arange(k)   #which zero mode (less than k)
probability_density = []
zero_state = []
localized_state_up = [] # it is the site ((L_x+L)/2, L_y/2)
localized_state_down = []
for i in index:
    destruction_up, destruction_down, creation_down, creation_up = get_components(eigenvectors_sparse[:,i], L_x, L_y)
    probability_density.append(np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)
    zero_state.append(np.stack((destruction_up, destruction_down, creation_down, creation_up), axis=2)) #positive energy eigenvector splitted in components
    localized_state_up.append(zero_state[i][(L_y-L)//2, L_x-2,:])
    localized_state_down.append(zero_state[i][(L_y+L)//2, L_x-2,:])

#%%
index = 0
fig, ax = plt.subplots(num="Probability density", clear=False)
image = ax.imshow(probability_density[index], cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
plt.colorbar(image)
#ax.set_title(f"{params}")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Probability density (TRITOPs-S)")
#ax.text(5,25, rf'$t_J={t_J}; \Phi={np.round(Phi, 2)}$; index={index}')
#plt.plot(probability_density[10,:,0])
plt.tight_layout()
#%% Energies
ax2 = fig.add_axes([0.3, 0.3, 0.25, 0.25])
ax2.scatter(np.linspace(0, len(eigenvalues_sparse), len(eigenvalues_sparse)), eigenvalues_sparse)
# ax2.set_xlim([2*(L_x*L_y-5), 2*(L_x*L_y+5)])
# ax2.set_ylim([-0.05, 0.05])

#%% Spin determination
from functions import mean_spin, mean_spin_xy, get_components

zero_modes = eigenvectors_sparse      #4 (2) modes with zero energy (with Zeeman)
creation_up, creation_down, destruction_down, destruction_up = get_components(zero_modes[:,index], L_x, L_y)
zero_state = np.stack((destruction_up, destruction_down, creation_down, creation_up), axis=2) #positive energy eigenvector splitted in components

spin = mean_spin_xy(zero_state)
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
ax.quiver(x, y, u, v, color='r')
ax.set_title('Spin Field in the plane')

#%% Spin in y
fig, ax = plt.subplots()
image = ax.imshow(spin[:,:,1], cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
plt.colorbar(image)

fig, ax = plt.subplots()
ax.plot(spin[:, L_x-2, 1])
