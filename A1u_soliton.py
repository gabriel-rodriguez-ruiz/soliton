#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 12:04:46 2023

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from hamiltonians import Hamiltonian_soliton_A1u
from functions import probability_density, get_components

L_x = 20
L_y = 80
t = 1
Delta = 1
mu = -1  #-2
Phi = np.pi   #superconducting phase
t_J = 1    #t/2
index = 0   #which zero mode
L = 40
H = Hamiltonian_soliton_A1u(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi, L=L)
probability_density_2D, eigenvalues, eigenvectors = probability_density(H, L_x, L_y, index=index)

#%%
fig, ax = plt.subplots()
image = ax.imshow(probability_density_2D, cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
plt.colorbar(image)
#ax.set_title(f"{params}")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.text(5,25, rf'$index={index}; \Phi={np.round(Phi, 2)}$')
#plt.plot(probability_density[10,:,0])
plt.tight_layout()

#%% Energies
ax2 = fig.add_axes([0.15, 0.15, 0.25, 0.25])
ax2.scatter(np.arange(0, 4*L_x*L_y, 1), eigenvalues)
ax2.set_xlim([2*(L_x*L_y-5), 2*(L_x*L_y+5)])
ax2.set_ylim([-0.05, 0.05])

#%% Spin determination
from functions import mean_spin_xy, get_components

zero_modes = eigenvectors[:, 2*(L_x*L_y-1):2*(L_x*L_y+1)]      #4 (2) modes with zero energy (with Zeeman)
creation_up, creation_down, destruction_down, destruction_up = get_components(zero_modes[:,index], L_x, L_y)
zero_plus_state = np.stack((destruction_up, destruction_down, creation_down, creation_up), axis=2) #positive energy eigenvector splitted in components
# corner_state = zero_plus_state[L_x-2, L_y//2, :].reshape(4,1)  #positive energy point state localized at the junction
# corner_state_normalized = corner_state/np.linalg.norm(corner_state[:2]) #normalization with only particle part
zero_plus_state_normalized = zero_plus_state/np.linalg.norm(zero_plus_state[:,:,:2], axis=2, keepdims=True)

# Spin mean value
# spin_mean_value = mean_spin(corner_state_normalized)

#spin = mean_spin_xy(zero_plus_state_normalized)
spin = mean_spin_xy(zero_plus_state)

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
ax.quiver(x, y, u, v, color='r', angles='uv', scale_units='xy', scale=1)
ax.set_title('Spin Field in the plane')

#%% Spin in z
fig, ax = plt.subplots()
image = ax.imshow(spin[:,:,2], cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
plt.colorbar(image)

fig, ax = plt.subplots()
ax.plot(spin[:,10,2])
plt.text(0,0, f"Total spin={np.sum(spin[:,10,2])}")
#%% Phi spectrum
from functions import phi_spectrum

Phi_values = np.linspace(0, 2*np.pi, 10)
phi_energy = phi_spectrum(Hamiltonian_soliton_A1u, Phi_values, t, mu, L_x, L_y, Delta, t_J, L)

fig, ax = plt.subplots()
ax.plot(Phi_values, phi_energy)
ax.set_xlabel(r"$\phi$")
ax.set_ylabel("E")

