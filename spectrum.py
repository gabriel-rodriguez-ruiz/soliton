#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:01:39 2023

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from hamiltonians import Hamiltonian_soliton_A1u, Hamiltonian_soliton_A1u_sparse, Hamiltonian_A1u_single_step_sparse, Hamiltonian_A1u_sparse, Zeeman
from functions import get_components
import scipy
from phase_functions import phase_single_soliton, phase_double_soliton

L_x = 200
L_y = 200
t = 1
Delta = 1
mu = -2  #-2
t_J = t    #t/2
L = L_y//2
k = 8   #number of eigenvalues

phi_profile = phase_single_soliton
phi_external = 0.5*np.pi
# H = Hamiltonian_soliton_A1u_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, phi_external=phi_external, phi_profile=phi_profile, L=L)
H = Hamiltonian_A1u_single_step_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, phi_external=phi_external, phi_profile=phi_profile)

eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H, k=k, sigma=0) 

#%% Plotting vs. Phi
Phi_values = np.linspace(0, 2*np.pi, 30)
eigenvalues = []

for Phi_value in Phi_values:
    # H = Hamiltonian_soliton_A1u_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi_value, L=L)
    H = Hamiltonian_A1u_single_step_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, phi_external=Phi_value, phi_profile=phi_profile)
    eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H, k=k, sigma=0) 
    eigenvalues_sparse.sort()
    eigenvalues.append(eigenvalues_sparse)
    
fig, ax = plt.subplots()
# ax.plot(Phi_values, [eigenvalues[i][0::2] for i in range(len(Phi_values))], "o", alpha=0.1)
# ax.plot(Phi_values, [eigenvalues[i][0::1] for i in range(len(Phi_values))], "*", alpha=0.1)
ax.plot(Phi_values, [np.abs(eigenvalues[i][0]) for i in range(len(Phi_values))], "or", alpha=0.5, markersize=5)
ax.plot(Phi_values, [np.abs(eigenvalues[i][1]) for i in range(len(Phi_values))], "ob", alpha=0.5, markersize=5)
ax.plot(Phi_values, [np.abs(eigenvalues[i][2]) for i in range(len(Phi_values))], "ok", alpha=0.5, markersize=5)
ax.plot(Phi_values, [np.abs(eigenvalues[i][3]) for i in range(len(Phi_values))], "oy", alpha=0.5, markersize=5)
plt.yscale('log')
ax.set_xlabel(r"$\Phi$")
ax.set_ylabel(r"$E$")
#%% Plotting vs. t_J
from hamiltonians import Hamiltonian_A1u_junction_sparse

from phase_functions import phase_single_soliton

phi_external = 0
y = np.arange(1, L_y+1)
y_s = (L_y+1)//2
Phi = phase_single_soliton(phi_external, y, y_s)

t_J_values = np.linspace(0, 1, 10)
eigenvalues = []
index = np.arange(k)   #which zero mode (less than k)
probability_density = []
zero_state = []
localized_state_up = [] # it is the site ((L_x+L)/2, L_y/2)
localized_state_down = []

for t_J_value in t_J_values:
    H = Hamiltonian_A1u_junction_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J_value, Phi=Phi)
    eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H, k=k, sigma=0) 
    eigenvalues_sparse.sort()
    eigenvalues.append(eigenvalues_sparse)
    destruction_up, destruction_down, creation_down, creation_up = get_components(eigenvectors_sparse[:,0], L_x, L_y)
    probability_density.append((np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)/(np.linalg.norm(np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)))
    zero_state.append(np.stack((destruction_up, destruction_down, creation_down, creation_up), axis=2)) #positive energy eigenvector splitted in components

#%% Plotting vs. tJ
    
fig, ax = plt.subplots()
ax.plot(t_J_values, [eigenvalues[i][0::2] for i in range(len(t_J_values))], "o", alpha=0.5, markersize=10)
ax.plot(t_J_values, [eigenvalues[i][0::1] for i in range(len(t_J_values))], "*", alpha=1, markersize=5)
ax.set_xlabel(r"$t_J$")
ax.set_ylabel(r"$E$")
plt.yscale('log')

fig, ax = plt.subplots()
for k in [1,3,5,7,9]:
    ax.plot(probability_density[k][:, L_x//2], label=r"$t_J=$"+f"{t_J_values[k].round(2)}")
ax.set_xlabel("y")
ax.set_ylabel("Probability density")
ax.text(5,25, rf'$index={index}; \Phi={np.round(Phi, 2)}$')
ax.set_title("Probability density at the junction")
ax.legend()
#%% Plotting vs. Ly

# L_y_values = np.linspace(50, 200, 11, dtype=int)
L_y_values = np.linspace(200, 400, 11, dtype=int)

eigenvalues = []

for L_y_value in L_y_values:
    L_x = int(L_y_value)
    L = int(L_y_value//2)
    #H = Hamiltonian_soliton_A1u_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y_value, Delta=Delta, t_J=t_J, Phi=Phi, L=L)
    H = Hamiltonian_A1u_single_step_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y_value, Delta=Delta, t_J=t_J, Phi=Phi)
    eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H, k=k, sigma=0) 
    eigenvalues_sparse.sort()
    eigenvalues.append(eigenvalues_sparse)
    
fig, ax = plt.subplots()
#ax.plot(L_y_values, [eigenvalues[i][0::2] for i in range(len(L_y_values))], "o", alpha=0.5, markersize=10)
#ax.plot(L_y_values, [eigenvalues[i][0::1] for i in range(len(L_y_values))], "*", alpha=0.5, markersize=5)
ax.plot(L_y_values, [np.abs(eigenvalues[i][0]) for i in range(len(L_y_values))], "or", alpha=0.5, markersize=5)
ax.plot(L_y_values, [np.abs(eigenvalues[i][1]) for i in range(len(L_y_values))], "ob", alpha=0.5, markersize=5)
ax.plot(L_y_values, [np.abs(eigenvalues[i][2]) for i in range(len(L_y_values))], "ok", alpha=0.5, markersize=5)
ax.plot(L_y_values, [np.abs(eigenvalues[i][3]) for i in range(len(L_y_values))], "oy", alpha=0.5, markersize=5)

ax.set_xlabel(r"$L_y$")
plt.yscale('log')
ax.set_ylabel("E")

#%% Plotting E vs. L
import numpy as np
from phase_functions import phase_soliton_antisoliton, phase_single_soliton
from hamiltonians import Hamiltonian_A1u, Hamiltonian_A1us_junction_sparse, Hamiltonian_A1us_junction_sparse_periodic, Hamiltonian_A1us_junction_sparse_periodic_in_x_and_y

L_x = 200
L_y = 200
t = 1
Delta = t/2
Delta_0 = t/10
mu = -2*t  #-2
phi_external = 0.
t_J = t/10    #t/2
k = 12
y = np.arange(1, L_y+1)
y_s = (L_y+1)//2
L_values = np.linspace(10, 100, 10, dtype=int)

eigenvalues = []

for L_value in L_values:
    y_0 = (L_y-L_value)//2
    y_1 = (L_y+L_value)//2
    y_s = L_value
    Phi = phase_soliton_antisoliton(phi_external, y, y_0, y_1)
    # Phi = phase_single_soliton(phi_external, y, y_s)
    # H = Hamiltonian_A1us_junction_sparse_periodic(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, Delta_0=Delta_0, t_J=t_J, Phi=Phi)
    # H = Hamiltonian_A1us_junction_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, Delta_0=Delta_0, t_J=t_J, Phi=Phi)
    H = Hamiltonian_A1us_junction_sparse_periodic_in_x_and_y(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, Delta_0=Delta_0, t_J=t_J, Phi=Phi)
    eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H, k=k, sigma=0) 
    eigenvalues_sparse.sort()
    eigenvalues.append(eigenvalues_sparse)

index = np.arange(k)
E_numerical = []
for j in index:
    E_numerical.append(np.array([eigenvalues[i][j] for i in range(len(L_values))]))
fig, ax = plt.subplots()
#ax.plot(L_y_values, [eigenvalues[i][0::2] for i in range(len(L_y_values))], "o", alpha=0.5, markersize=10)
#ax.plot(L_y_values, [eigenvalues[i][0::1] for i in range(len(L_y_values))], "*", alpha=0.5, markersize=5)
#ax.plot(L_values, [eigenvalues[i][0] for i in range(len(L_values))], "or", alpha=0.5, markersize=5)
# ax.plot(L_values, [eigenvalues[i][1] for i in range(len(L_values))], "ob", alpha=0.5, markersize=5)
# ax.plot(L_values, E_numerical[6], "o", label="Numerical")
# ax.plot(L_values, E_numerical[8], "o", label="Numerical")
# ax.plot(L_values, E_numerical[10], "o", label="Numerical")
for i in range(len(index)):
    ax.plot(L_values, E_numerical[i], "*", label="Numerical")

ax.set_xlabel(r"$L$")
plt.yscale('log')
ax.set_ylabel("E")

from analytical_solution import Kappa

m_0 = t_J/2

def positive_energy(L, m_0):
    kappa_value = Kappa(m_0=m_0, Delta=Delta, L=L_value)
    return m_0*np.exp(-kappa_value*L)

E = []
for L_value in L_values:
    Energy = positive_energy(L=L_value, m_0=m_0)
    E.append(Energy[0])

E_analytical = np.array([E[i] for i in range(len(L_values))])
ax.plot(L_values, E_analytical, "ok", label="Analytical")
plt.yscale('log')

#%% Least square fitting

# m_numerical, b_numerical = np.polyfit(L_values[3:], np.log(E_numerical[6][3:]), 1)
m_numerical, b_numerical = np.polyfit(L_values[1:-3], np.log(E_numerical[8][1:-3]), 1)
m_analytical, b_analytical = np.polyfit(L_values, np.log(E_analytical), 1)

ax.plot(L_values[1:-3], np.exp(m_numerical*L_values + b_numerical)[1:-3], label=f"{m_numerical:.3}L{b_numerical:.3}")
ax.plot(L_values, np.exp(m_analytical*L_values + b_analytical), label=f"{m_analytical:.3}L{b_analytical:.3}")
ax.legend()
plt.title(r"$\phi_{ext}=$"+f"{phi_external:.2}, Delta={Delta}")
plt.tight_layout()
