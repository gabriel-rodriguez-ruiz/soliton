# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 07:57:39 2023

@author: gabri
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
Phi = np.pi  #superconducting phase
t_J = 1    #t/2
L = L_y//2
k = 8   #number of eigenvalues
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

index = 0
fig, ax = plt.subplots()
image = ax.imshow(probability_density[index], cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
plt.colorbar(image)
#ax.set_title(f"{params}")
ax.set_xlabel("x")
ax.set_ylabel("y")
#ax.text(5,25, rf'$index={index}; \Phi={np.round(Phi, 2)}$')
#ax.text(5,25, rf'$\Phi={np.round(Phi, 2)}$')
#plt.plot(probability_density[10,:,0])
plt.tight_layout()
ax.set_title("Probability density (TRITOPS-TRITOPS)")
def single_soliton(y, L_y):
    return np.heaviside(L_y//2-y_values, 1)-np.heaviside(y_values-L_y//2, 1)

def double_soliton(y, L_y, L):
    return 1+2*(np.heaviside(y_values-((L_y+L)/2), 1)-np.heaviside(y_values-((L_y-L)//2), 1))


# left, bottom, width, height = [0.65, 0.28, 0.1, 0.5]
left, bottom, width, height = [0.65, 0.28, 0.1, 0.5]
ax2 = fig.add_axes([left, bottom, width, height])
y_values = np.arange(L_y+1)
# ax2.plot(single_soliton(y_values, L_y), y_values, "r")
ax2.plot(double_soliton(y_values, L_y, L), y_values, "r")
ax2.set_ylabel("y")
ax2.set_xlabel(r"$sgn(\phi)$")
ax2.set_title(r"$f_{2s}$")
ax2.set_ylim((20,100))

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
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5,8))
i = 0
for ax in axes.flat:
    image = ax.imshow(spin[i][:,:,2], cmap="Blues", origin="lower", vmin=np.min(spin), vmax=np.max(spin)) #I have made the transpose and changed the origin to have xy axes as usually
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    i += 1
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(image, cax=cbar_ax)
axes[0].set_title('Spin Field in the z direction')
#plt.text(0,0, f"index={index}")

# fig, ax = plt.subplots()
# ax.plot(spin[:, L_x//2,2])
# total_spin = np.sum(spin[:, L_x//2, 2])
# plt.text(0,0.25, f"Total spin={total_spin}, index={index}")
# plt.text(0,-0.25, f"Total spin={total_spin}, index={index}")
# ax.set_title('Spin Field in the z direction')
# plt.text(0,0, f"index={index}")

