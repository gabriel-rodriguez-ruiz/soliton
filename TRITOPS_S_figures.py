#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 12:31:18 2023

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from hamiltonians import Hamiltonian_soliton_A1u, Hamiltonian_soliton_A1u_sparse, Hamiltonian_A1u_single_step_sparse, Hamiltonian_A1u_sparse, Zeeman, Hamiltonian_A1u_S
from functions import get_components
import scipy

L_x = 120
L_y = 120
t = 1
Delta = 1
mu = -2  #-2
Phi = np.pi/2  #superconducting phase
t_J = 1    #t/2
L = L_y//2
k = 8   #number of eigenvalues
Delta_Z = 0
theta = np.pi/2
phi = 0

H = Hamiltonian_A1u_S(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi)
#H = Hamiltonian_soliton_A1u_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi, L=L)
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
#ax.text(5,25, rf'$\phi={np.round(Phi, 2)}$')
ax.set_title("Probability density (TRITOPS-S)")
ax.set_ylim((0,L_y))

def single_soliton(y, L_y):
    return np.heaviside(L_y//2-y_values, 1)-np.heaviside(y_values-L_y//2, 1)

left, bottom, width, height = [0.6, 0.24, 0.1, 0.5]
ax2 = fig.add_axes([left, bottom, width, height])
y_values = np.arange(L_y+1)
ax2.plot(single_soliton(y_values, L_y), y_values, "r")
ax2.set_ylabel("y")
ax2.set_xlabel(r"$sgn(\phi)$")
ax2.set_title(r"$f_{1s}$")
ax2.set_ylim((0,L_y))
