#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 09:41:17 2023

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from hamiltonians import Hamiltonian_A1u_junction_sparse, Hamiltonian_A1u_S, Hamiltonian_A1u_junction_sparse_periodic, Hamiltonian_A1u_junction_sparse_periodic_in_x
from functions import get_components
from phase_functions import phase_soliton_antisoliton_arctan, phase_single_soliton, phase_single_soliton_arctan, phase_soliton_soliton_arctan, phase_soliton_antisoliton
import scipy

L_x = 200
L_y = 201
t = 1
Delta = 1
mu = -2  #-2
t_J = t   #t
L = 30      #L_y//2
k = 12 #number of eigenvalues
lambda_J = 5
# phi_profile = phase_single_soliton_arctan
phi_external = 0
y = np.arange(1, L_y+1)
y_0 = (L_y-L)//2
y_1 = (L_y+L)//2
y_s = (L_y+1)//2

# Phi = phi_profile(phi_external, y, L_y//2, lambda_J)
Phi = phase_single_soliton(phi_external, y, y_s)
# Phi = phase_soliton_antisoliton(phi_external, y, y_0, y_1)
# Phi = phase_soliton_antisoliton_arctan(phi_external, y, (L_y-L)//2, (L_y+L)//2, lambda_J)

params = {"t": t, "mu": mu, "L_x": L_x, "L_y": L_y, "Delta": Delta, "t_J": t_J, "L": L}
H = Hamiltonian_A1u_junction_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi)
# H = Hamiltonian_A1u_junction_sparse_periodic(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi)
# H = Hamiltonian_A1u_junction_sparse_periodic_in_x(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi)

eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H, k=k, sigma=0) 

#%% Probability density
index = np.arange(k)   #which zero mode (less than k)
particle_density = []
zero_state = []
localized_state_upper_left = [] # it is the site (L_x/2-1, (L_y+L)/2)
localized_state_upper_right = [] # it is the site (L_x/2, (L_y+L)/2)
localized_state_bottom_left = [] # it is the site (L_x/2-1, (L_y-L)/2)
localized_state_bottom_right = [] # it is the site (L_x/2, (L_y-L)/2)
localized_state_left = []
localized_state_right = []

for i in index:
    destruction_up, destruction_down, creation_down, creation_up = get_components(eigenvectors_sparse[:,i], L_x, L_y)
    particle_density.append(np.abs(destruction_up)**2 + np.abs(destruction_down)**2 )
    zero_state.append(np.stack((destruction_up, destruction_down, creation_down, creation_up), axis=2)) #positive energy eigenvector splitted in components
    localized_state_upper_left.append(zero_state[i][(L_y+L)//2, L_x//2-1,:])
    localized_state_upper_right.append(zero_state[i][(L_y+L)//2, L_x//2,:])
    localized_state_bottom_left.append(zero_state[i][(L_y-L)//2, L_x//2-1,:])
    localized_state_bottom_right.append(zero_state[i][(L_y-L)//2, L_x//2,:])
    localized_state_left.append(zero_state[i][:, L_x//2-1,:])
    localized_state_right.append(zero_state[i][:, L_x//2,:])

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
image = ax.imshow(particle_density[index], cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
plt.colorbar(image)
#ax.set_title(f"{params}")
ax.set_xlabel("x")
ax.set_ylabel("y")
#ax.text(5,25, rf'$index={index}; \Phi={np.round(Phi, 2)}$')
#plt.plot(probability_density[10,:,0])
ax.set_title("Probability density")
plt.tight_layout()
particle_density_right = particle_density[index][:, L_x//2]/np.sum(particle_density[index][:, L_x//2])
particle_density_left = particle_density[index][:, L_x//2-1]/np.sum(particle_density[index][:, L_x//2-1])

xi = t_J/Delta  #m_0/v
y_analytical = np.linspace(1, L_y+1, 10000)
dy = np.diff(y_analytical)[0]
particle_density_analytical = 1/2*xi*np.exp(-xi*np.abs(y_analytical-y_s))

fig, ax = plt.subplots()
ax.plot(y, particle_density_right, "o", label="Numerics")
# ax.plot(y, particle_density_left, "o")

ax.plot(y_analytical, particle_density_analytical, label="Theory")

#ax.plot(np.arange(1, L_y+1), probability_density[index][:, L_x//2-1])
ax.set_xlabel(r"$\ell$")
ax.set_ylabel("Particle density")
ax.text(5,25, rf'$index={index}$')
ax.set_xticks([1,50,100,150,200])
# ax.set_title("Probability density at the junction")
plt.tight_layout()
ax.legend()
#%% Energy spectrum

fig, ax = plt.subplots()
plt.plot(eigenvalues_sparse[:4], "o")
ax.set_xlabel("Label of eingevalue")
ax.set_ylabel("Energy")
plt.tight_layout()

#%%

fig, ax = plt.subplots(1, 2)
image_left = ax[0].imshow(particle_density[index][:, :L_x//2], cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
image_right = ax[1].imshow(particle_density[index][:, L_x//2:], cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually

# plt.colorbar(image)
#ax.set_title(f"{params}")
ax[0].set_xlabel(r"$\ell$")
ax[1].set_xlabel(r"$\ell$")

ax[0].set_ylabel("y")
ax[0].set_xticks([0, L_x//2-1], [f"{1}", f"{L_x//2-1}"])
ax[1].set_xticks([0, L_x//2], [f"{L_x//2}", f"{L_x}"])
ax[1].set_yticks([])
# set the spacing between subplots
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,  
                    top=0.9,
                    wspace=0,
                    hspace=0)
