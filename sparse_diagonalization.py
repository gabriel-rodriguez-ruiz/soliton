#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:44:35 2023

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from hamiltonians import Hamiltonian_soliton_A1u, Hamiltonian_soliton_A1u_sparse, Hamiltonian_A1u_single_step_sparse, Hamiltonian_A1u_sparse, Zeeman, Hamiltonian_A1u_S
from functions import get_components
import scipy

L_x = 200
L_y = 200
t = 1
Delta = 1
mu = -2  #-2
Phi = 0.04*np.pi  #height of the phase soliton around flux pi
t_J = 1   #t/2
L = L_y//2
k = 8   #number of eigenvalues
Delta_Z = 0
theta = np.pi/2
phi = 0

#H = Hamiltonian_A1u_S(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi)
#H = Hamiltonian_soliton_A1u_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi, L=L)
params = {"t": t, "mu": mu, "L_x": L_x, "L_y": L_y, "Delta": Delta, "t_J": t_J, "Phi": Phi, "L": L}
H = Hamiltonian_A1u_single_step_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi)
#H = (Hamiltonian_A1u_sparse(t, mu, L_x, L_y, Delta) + Zeeman(theta=theta, Delta_Z=Delta_Z, L_x=L_x, L_y=L_y, phi=phi))

eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H, k=k, sigma=0) 

#%% Probability density
index = np.arange(k)   #which zero mode (less than k)
probability_density = []
zero_state = []
localized_state_up = [] # it is the site ((L_x+L)/2, L_y/2)
localized_state_down = []
localized_state_center_upper_left = []
localized_state_center_upper_right = []
localized_state_center_bottom_left = []
localized_state_center_bottom_right = []
localized_state_bottom_edge_middle = []
localized_state_left_edge_middle = []

for i in index:
    destruction_up, destruction_down, creation_down, creation_up = get_components(eigenvectors_sparse[:,i], L_x, L_y)
    probability_density.append((np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)/(np.linalg.norm(np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)))
    zero_state.append(np.stack((destruction_up, destruction_down, creation_down, creation_up), axis=2)) #positive energy eigenvector splitted in components
    localized_state_down.append(zero_state[i][(L_y-L)//2, L_x//2,:])
    localized_state_up.append(zero_state[i][(L_y+L)//2, L_x//2,:])
    localized_state_center_upper_left.append(zero_state[i][L_y//2, L_x//2-1,:])
    localized_state_center_upper_right.append(zero_state[i][L_y//2, L_x//2,:])
    localized_state_center_bottom_left.append(zero_state[i][L_y//2-1, L_x//2-1,:])
    localized_state_center_bottom_right.append(zero_state[i][L_y//2-1, L_x//2,:])
    localized_state_bottom_edge_middle.append(zero_state[i][0, L_x//2,:])
    localized_state_left_edge_middle.append(zero_state[i][L_y//2, 0,:])

#%% Plotting of probability density

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


index = 0
fig, ax = plt.subplots()
image = ax.imshow(probability_density[index], cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
plt.colorbar(image)
#ax.set_title(f"{params}")
ax.set_xlabel("x")
ax.set_ylabel("y")
#ax.text(5,25, rf'$index={index}; \Phi={np.round(Phi, 2)}$')
ax.text(5,25, rf'$\Phi={np.round(Phi, 2)}$')
#plt.plot(probability_density[10,:,0])
plt.tight_layout()
ax.set_title("Probability density")
plt.tight_layout()

fig, ax = plt.subplots()
ax.plot(probability_density[index][:, L_x//2])
ax.set_xlabel("y")
ax.set_ylabel("Probability density")
ax.text(5,25, rf'$index={index}; \Phi={np.round(Phi, 2)}$')
ax.set_title("Probability density at the junction")
    #%% Spin determination
from functions import mean_spin_xy, get_components

# zero_modes = eigenvectors_sparse
# creation_up, creation_down, destruction_down, destruction_up = get_components(zero_modes[:,index], L_x, L_y)
# zero_state = np.stack((destruction_up, destruction_down, creation_down, creation_up), axis=2) #positive energy eigenvector splitted in components
#corner_state = zero_state[L, L_y//2, :].reshape(4,1)  #positive energy point state localized at the junction
# corner_state_normalized = corner_state/np.linalg.norm(corner_state[:2]) #normalization with only particle part
# spin_mean_value = mean_spin(corner_state_normalized)

spin = []
for i in range(k):
    spin.append(mean_spin_xy(zero_state[i]))

#%%
# fig, ax = plt.subplots()
# image = ax.imshow(spin[:,:,2].T, cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
# plt.colorbar(image)
#image.set_clim(np.min(spin[:,:,1].T), np.max(spin[:,:,1].T))

# Meshgrid
x, y = np.meshgrid(np.linspace(0, L_x-1, L_x), 
                    #np.linspace(L_y-1, 0, L_y))
                    np.linspace(0, L_y-1, L_y))


  
# Directional vectors
u = spin[index][:, :, 0]   #x component
v = spin[index][:, :, 1]   #y component

# Plotting Vector Field with QUIVER
fig,ax = plt.subplots()
ax.quiver(x, y, u, v, color='r', angles='uv')
ax.set_title('Spin Field in the plane')

#%% Spin in z
fig, ax = plt.subplots()
image = ax.imshow(spin[index][:,:,2], cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
plt.colorbar(image)
ax.set_title('Spin Field in the z direction')
plt.text(0,0, f"index={index}")

# fig, ax = plt.subplots()
# ax.plot(spin[:, L_x//2,2])
# total_spin = np.sum(spin[:, L_x//2, 2])
# plt.text(0,0.25, f"Total spin={total_spin}, index={index}")
# plt.text(0,-0.25, f"Total spin={total_spin}, index={index}")
# ax.set_title('Spin Field in the z direction')
# plt.text(0,0, f"index={index}")

#%% Energy spectrum

fig, ax = plt.subplots()
plt.semilogy(np.abs(eigenvalues_sparse), "o")
ax.set_xlabel("Label of eingevalue")
ax.set_ylabel("Absolute value of energy")
plt.tight_layout()

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

#%% Spinors to txt
from functions import mean_spin

with open("spinors.txt", "w+") as f:
  data = f.read()
  f.write(f"{params}\n")
  f.write(f"energies={eigenvalues_sparse}\n\n")
  for i in range(4):
      f.write(f"{i}th-localized state at the center\n\n")
      for j in range(4):
          f.write(f"{str(localized_state_center_upper_left[i].round(4)[j]):30}"+"%    "+
                  f"{str(localized_state_center_upper_right[i].round(4)[j])}"+"\n")
      f.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
      for j in range(4):
          f.write(f"{str(localized_state_center_bottom_left[i].round(4)[j]):30}"+"%    "+
                  f"{str(localized_state_center_bottom_right[i].round(4)[j])}"+"\n")
      f.write("\n\n")
    # for i in range(4):
  #     f.write(f"{i}th-localized state at the top\n")
  #     for j in range(len(localized_state_up[i])):
  #         f.write(str(localized_state_up[i].round(4)[j])+"\n")
   
  