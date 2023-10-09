#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:44:35 2023

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from hamiltonians import Hamiltonian_A1u_junction_sparse, Hamiltonian_A1u_S, Hamiltonian_A1u_junction_sparse_periodic, Hamiltonian_A1u_S_periodic, Hamiltonian_A1us_S_sparse, Hamiltonian_A1us_junction_sparse, Hamiltonian_A1us_junction_sparse_periodic, Hamiltonian_A1us_junction_sparse_periodic_in_x_and_y, Hamiltonian_A1us_S_sparse_periodic, Hamiltonian_A1us_S_sparse_extended
from functions import get_components
from phase_functions import phase_soliton_antisoliton_arctan, phase_single_soliton, phase_single_soliton_arctan, phase_soliton_soliton_arctan, phase_soliton_antisoliton, phase_soliton_antisoliton_arctan_A1u_S_around_zero, \
    phase_antisoliton_soliton, phase_soliton_antisoliton_arctan_A1u_S_around_pi, phase_soliton_antisoliton_S_around_zero, phase_soliton_antisoliton_S_around_pi
import scipy

L_x = 100
L_y = 200       #L_y should be odd for single soliton
t = 1
Delta = t/3    # TRITOPS-S      t/5
# Delta = t/2   #TRITOPS-TRITOPS
Delta_0 = t/10
mu = -2*t  #-2
t_J = t/10   #t/10
L = 100      #L_y//2
n = 12 #number of eigenvalues
# lambda_J = 10
phi_external = 0
phi_eq = 0.08*2*np.pi    #0.14*2*np.pi
y = np.arange(1, L_y+1)
y_0 = (L_y-L)//2
y_1 = (L_y+L)//2
y_s = (L_y+1)//2
# y_s = L
# Phi = phi_profile(phi_external, y, L_y//2, lambda_J)
# Phi = phase_single_soliton(phi_external, y, y_s)
# Phi = phase_single_soliton_arctan(phi_external, y, y_s, lambda_J)
# Phi = phi_eq * np.ones_like(y)
# Phi = phase_soliton_antisoliton(phi_external, y, y_0, y_1)
# Phi = phase_soliton_antisoliton_arctan(phi_external, y, y_0, y_1, lambda_J)
# Phi = phase_soliton_antisoliton_arctan_A1u_S_around_zero(phi_external, y, y_0, y_1, lambda_J)
# Phi = phase_antisoliton_soliton(phi_external, y, y_0, y_1)
# Phi = phase_soliton_antisoliton_S_around_pi(phi_external, phi_eq, y, y_0, y_1)
Phi = phase_soliton_antisoliton_S_around_zero(phi_external, phi_eq, y, y_0, y_1)

params = {"t": t, "mu": mu, "L_x": L_x, "L_y": L_y, "Delta": Delta, "t_J": t_J, "L": L}
# H = Hamiltonian_A1u_junction_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi)
# H = Hamiltonian_A1u_junction_sparse_periodic(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi)
# H = Hamiltonian_A1u_S(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi)
# H = Hamiltonian_A1u_S_periodic(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi)
# H = Hamiltonian_A1u_junction_sparse_periodic_in_x(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi)
# H = Hamiltonian_A1u_junction(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi)
# H =  Hamiltonian_A1us_S_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, Delta_0=Delta_0, t_J=t_J, Phi=Phi)
# H = Hamiltonian_A1us_junction_sparse_periodic_in_x_and_y(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, Delta_0=Delta_0, t_J=t_J, Phi=Phi)
# H = Hamiltonian_A1us_junction_sparse_periodic(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, Delta_0=Delta_0, t_J=t_J, Phi=Phi)
# H =  Hamiltonian_A1us_S_sparse_periodic(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, Delta_0=Delta_0, t_J=t_J, Phi=Phi)
H =  Hamiltonian_A1us_S_sparse_extended(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, Delta_0=Delta_0, t_J=t_J, Phi=Phi)

eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H, k=n, sigma=0) 

#%% Probability density
index = np.arange(n)   #which zero mode (less than k)
probability_density = []
zero_state = []
localized_state_upper_left = [] # it is the site (L_x/2-1, (L_y+L)/2)
localized_state_upper_right = [] # it is the site (L_x/2, (L_y+L)/2)
localized_state_bottom_left = [] # it is the site (L_x/2-1, (L_y-L)/2)
localized_state_bottom_right = [] # it is the site (L_x/2, (L_y-L)/2)
localized_state_left = []
localized_state_right = []
probability_density_particle = []

for i in index:
    destruction_up, destruction_down, creation_down, creation_up = get_components(eigenvectors_sparse[:,i], L_x, L_y)
    probability_density.append((np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)/(np.linalg.norm(np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)))
    probability_density_particle.append((np.abs(destruction_up)**2 + np.abs(destruction_down)**2))
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
image = ax.imshow(probability_density[index], cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
plt.colorbar(image)
#ax.set_title(f"{params}")
ax.set_xlabel("x")
ax.set_ylabel("y")
#ax.text(5,25, rf'$index={index}; \Phi={np.round(Phi, 2)}$')
#plt.plot(probability_density[10,:,0])
ax.set_title("Probability density")
plt.tight_layout()
probability_density_right = probability_density[index][:, L_x//2]/np.linalg.norm(probability_density[index][:, L_x//2])  #The y-axis is inverted
# probability_density_right = probability_density[index][:, L_x-1]/np.linalg.norm(probability_density[index][:, L_x-1])  #The y-axis is inverted

fig, ax = plt.subplots()
ax.plot(y, probability_density_right, "o")
#ax.plot(np.arange(1, L_y+1), probability_density[index][:, L_x//2-1])
ax.set_xlabel(r"$\ell$")
ax.set_ylabel("Probability density at the junction")
ax.text(5,25, rf'$index={index}$')
ax.set_xticks([1,50,100,150,200])
# ax.set_title("Probability density at the junction")
# plt.tight_layout()

fig, ax = plt.subplots()
plt.title("Probability density particle")
image = ax.imshow(probability_density_particle[index], cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
plt.colorbar(image)
#ax.set_title(f"{params}")
ax.set_xlabel("x")
ax.set_ylabel("y")

fig, ax = plt.subplots()
plt.title("Soliton profile")
ax.plot(y, Phi)
ax.set_xlabel("x")
ax.set_ylabel("y")

#%% Spin determination
from functions import mean_spin_xy, get_components

# zero_modes = eigenvectors_sparse
# creation_up, creation_down, destruction_down, destruction_up = get_components(zero_modes[:,index], L_x, L_y)
# zero_state = np.stack((destruction_up, destruction_down, creation_down, creation_up), axis=2) #positive energy eigenvector splitted in components
#corner_state = zero_state[L, L_y//2, :].reshape(4,1)  #positive energy point state localized at the junction
# corner_state_normalized = corner_state/np.linalg.norm(corner_state[:2]) #normalization with only particle part
# spin_mean_value = mean_spin(corner_state_normalized)

spin = []
for i in range(n):
    spin.append(mean_spin_xy(zero_state[i]))

#%%
# fig, ax = plt.subplots()
# image = ax.imshow(spin[:,:,2].T, cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
# plt.colorbar(image)
#image.set_clim(np.min(spin[:,:,1].T), np.max(spin[:,:,1].T))

# Meshgrid
x_mesh, y_mesh = np.meshgrid(np.linspace(0, L_x-1, L_x), 
                    #np.linspace(L_y-1, 0, L_y))
                    np.linspace(0, L_y-1, L_y))


  
# Directional vectors
u = spin[index][:, :, 0]   #x component
v = spin[index][:, :, 1]   #y component

# Plotting Vector Field with QUIVER
fig,ax = plt.subplots()
ax.quiver(x_mesh, y_mesh, u, v, color='r', angles='uv')
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
plt.plot(eigenvalues_sparse, "o")
ax.set_xlabel("Label of eingevalue")
ax.set_ylabel("Energy")
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

index = range(0,4)

with open("spinors.txt", "w+") as f:
    data = f.read()
    f.write(f"{params}\n")
    f.write(f"energies={eigenvalues_sparse}\n\n")
    for i in index:
        f.write(f"{i}th-localized state at the upper soliton\n\n")
        for j in range(4):
            f.write(f"{str(localized_state_upper_left[i].round(4)[j]):30}"+"%    "+
                    f"{str(localized_state_upper_right[i].round(4)[j])}"+"\n")
        f.write("\n")
        f.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
    f.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
    f.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
    for i in index:
        f.write(f"{i}th-localized state at the bottom soliton\n\n")
        for j in range(4):
            f.write(f"{str(localized_state_bottom_left[i].round(4)[j]):30}"+"%    "+
                    f"{str(localized_state_bottom_right[i].round(4)[j])}"+"\n")
        f.write("\n")
        f.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
  