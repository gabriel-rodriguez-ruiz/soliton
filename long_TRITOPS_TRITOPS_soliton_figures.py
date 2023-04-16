# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 13:28:51 2023

@author: gabri
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
Phi = 0  #superconducting phase
t_J = 1    #t/2
L = L_y//2
k = 4   #number of eigenvalues
Delta_Z = 0
theta = np.pi/2
phi = 0

###########Choose Hamiltonian
#H = Hamiltonian_A1u_S(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi)
H = Hamiltonian_soliton_A1u_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi, L=L)
params = {"t": t, "mu": mu, "L_x": L_x, "L_y": L_y, "Delta": Delta, "t_J": t_J, "Phi": Phi, "L": L}
#H = Hamiltonian_A1u_single_step_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi)
#H = (Hamiltonian_A1u_sparse(t, mu, L_x, L_y, Delta) + Zeeman(theta=theta, Delta_Z=Delta_Z, L_x=L_x, L_y=L_y, phi=phi))

eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H, k=k, sigma=0) 

#%% Probability density
index = np.arange(k)   #which zero mode (less than k)
probability_density = []
zero_state = []
localized_state_up = [] # it is the site ((L_x+L)/2, L_y/2)
localized_state_down = []
for i in index:
    destruction_up, destruction_down, creation_down, creation_up = get_components(eigenvectors_sparse[:,i], L_x, L_y)
    probability_density.append((np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)/(np.linalg.norm(np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)))
    zero_state.append(np.stack((destruction_up, destruction_down, creation_down, creation_up), axis=2)) #positive energy eigenvector splitted in components
    localized_state_up.append(zero_state[i][(L_y-L)//2, L_x//2,:])
    localized_state_down.append(zero_state[i][(L_y+L)//2, L_x//2,:])

#%% Plotting of probability density

plt.rc("font", family="serif")  # set font family
plt.rc("xtick", labelsize="small")  # reduced tick label size
plt.rc("ytick", labelsize="small")
plt.rc("text", usetex=True) # for better LaTex (slower)
plt.rcParams['xtick.top'] = True    #ticks on top
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['ytick.right'] = True    #ticks on left
plt.rcParams['ytick.labelright'] = False

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


#%% Spin in z
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8,16))
for i in range(k):
    image = axes[i,0].imshow(probability_density[i], cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
    image2 = axes[i,1].imshow(spin[i][:,:,2], cmap="Blues", origin="lower", vmin=np.min(spin), vmax=np.max(spin)) #I have made the transpose and changed the origin to have xy axes as usually
    axes[i,0].set_xlim(45,155)
    axes[i,0].set_ylim(45,155)
    axes[i,1].set_xlim(45,155)
    axes[i,1].set_ylim(45,155)

fig.tight_layout()
fig.subplots_adjust(left=0.2)
cbar_ax = fig.add_axes([0.1, 0.15, 0.05, 0.6])
fig.colorbar(image, cax=cbar_ax)

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.6])
fig.colorbar(image2, cax=cbar_ax)

axes[0,0].set_title("Probability density")
axes[0,1].set_title("Spin in z")


