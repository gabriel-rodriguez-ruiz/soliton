#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:44:35 2023

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from hamiltonians import Hamiltonian_soliton_A1u, Hamiltonian_soliton_A1u_sparse, Hamiltonian_A1u_single_step_sparse
from functions import probability_density, get_components
import scipy

L_x = 50
L_y = 200
t = 1
Delta = 1
mu = -1  #-2
Phi = np.pi   #superconducting phase
t_J = 1    #t/2
index = 0   #which zero mode (less than k)
L = 100
k = 4   #number of eigenvalues
H = Hamiltonian_soliton_A1u_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi, L=L)
#H = Hamiltonian_A1u_single_step_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi)

eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H, k=k, sigma=0) 
a, b, c, d = get_components(eigenvectors_sparse[:,index], L_x, L_y)
probability_density = np.abs(a)**2 + np.abs(b)**2 + np.abs(c)**2 + np.abs(d)**2

#%% Probability density

fig, ax = plt.subplots()
image = ax.imshow(probability_density, cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
plt.colorbar(image)
#ax.set_title(f"{params}")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.text(5,25, rf'$index={index}; \Phi={np.round(Phi, 2)}$')
#plt.plot(probability_density[10,:,0])
plt.tight_layout()
ax.set_title("Probability density")

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
#ax.quiver(x, y, u, v, color='r', angles='uv')
#ax.set_title('Spin Field in the plane')

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

#%% Phi spectrum
"""
from functions import phi_spectrum_sparse_single_step

Phi_values = np.linspace(0, 2*np.pi, 10)
phi_energy = phi_spectrum_sparse_single_step(Hamiltonian_A1u_single_step_sparse, Phi_values, t, mu, L_x, L_y, Delta, t_J)

fig, ax = plt.subplots()
ax.plot(Phi_values, phi_energy)
ax.set_xlabel(r"$\phi$")
ax.set_ylabel("E")

"""