# -*- coding: utf-8 -*-
"""
Created on Sun May 28 18:28:17 2023

@author: gabri
"""

import numpy as np
import matplotlib.pyplot as plt
from hamiltonians import Hamiltonian_A1u_sparse, Hamiltonian_S_sparse_periodic, Hamiltonian_S_periodic
from functions import get_components
import scipy
L_x = 2
L_y = 4
t = 1
Delta = t/2
mu = -2*t
k = 24

# H = Hamiltonian_A1u_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta)
H = Hamiltonian_S_sparse_periodic(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta_0=Delta)
H_array = Hamiltonian_S_periodic(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta_0=Delta)

eigenvalues, eigenvectors = np.linalg.eigh(H_array) 
eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H, k=k, sigma=0) 
params = {"t": t, "mu": mu, "L_x": L_x, "L_y": L_y, "Delta": Delta}

#%% Probability density
index = np.arange(k)   #which zero mode (less than k)
probability_density = []
zero_state = []
localized_state_upper_left = [] # it is the site (L_x/2-1, (L_y+L)/2)
localized_state_upper_right = [] # it is the site (L_x/2, (L_y+L)/2)
localized_state_bottom_left = [] # it is the site (L_x/2-1, (L_y-L)/2)
localized_state_bottom_right = [] # it is the site (L_x/2, (L_y-L)/2)
localized_state_left = []
localized_state_right = []

for i in index:
    destruction_up, destruction_down, creation_down, creation_up = get_components(eigenvectors_sparse[:,i], L_x, L_y)
    probability_density.append((np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)/(np.linalg.norm(np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)))
    zero_state.append(np.stack((destruction_up, destruction_down, creation_down, creation_up), axis=2)) #positive energy eigenvector splitted in components

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
#plt.plot(probability_density[10,:,0])
ax.set_title("Probability density")
plt.tight_layout()

fig, ax = plt.subplots()
ax.plot(np.arange(1, L_y+1), probability_density[index][:, L_x//2])
#ax.plot(np.arange(1, L_y+1), probability_density[index][:, L_x//2-1])
ax.set_xlabel("y")
ax.set_ylabel("Probability density")
ax.set_title("Probability density at the junction")
