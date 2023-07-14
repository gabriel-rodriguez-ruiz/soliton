#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 11:51:06 2023

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from hamiltonians import Hamiltonian_A1u_junction_k, Hamiltonian_A1u_S_junction_sparse_k

# Pauli matrices
sigma_0 = np.eye(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
tau_0 = np.eye(2)
tau_x = np.array([[0, 1], [1, 0]])
tau_y = np.array([[0, -1j], [1j, 0]])
tau_z = np.array([[1, 0], [0, -1]])



def Hamiltonian(t, k, mu, L, Delta_0, Delta_1, lambda_R, theta):
    r"""Returns the H_k matrix for ZKM model with:

    .. math::
        H_{ZKM} = \frac{1}{2}\sum_k H_k
        
        H_k = \sum_{n=1}^L 
            \vec{c}^\dagger_n\left[ 
            \xi_k\tau_z\sigma_0+\Delta_k\tau_x\sigma_0
            -2\lambda\sin(k)\tau_z(cos(\theta)\sigma_x + sin(\theta)\sigma_z)\right]\vec{c}_n+
            \sum_{n=1}^{L-1}             
            \left[
            \vec{c}^\dagger_n(-t\tau_z\sigma_0-i\lambda\tau_z\sigma_z + \Delta_1\tau_x\sigma_0 )\vec{c}_{n+1}
            + H.c.
            \right]
            
       \vec{c} = (c_{k,\uparrow}, c_{k,\downarrow},c^\dagger_{-k,\downarrow},-c^\dagger_{-k,\uparrow})^T
       
       \xi_k = -\mu - 2t\cos(k)
       
       \Delta_k = \Delta_0+2\Delta_1\cos(k)
    """
    chi_k = -mu - 2*t * np.cos(k)
    Delta_k = Delta_0 + 2*Delta_1*np.cos(k)
    onsite = chi_k * np.kron(tau_z, sigma_0) + \
            Delta_k * np.kron(tau_x, sigma_0) - \
            2*lambda_R*np.sin(k) * (np.cos(theta)*np.kron(tau_z, sigma_x) + np.sin(theta)*np.kron(tau_z, sigma_z))
    hopping = -t*np.kron(tau_z, sigma_0) - 1j*lambda_R * np.kron(tau_z, sigma_z) + Delta_1*np.kron(tau_x, sigma_0)
    matrix_diagonal = np.kron(np.eye(L), onsite)     #diagonal part of matrix
    matrix_outside_diagonal = np.block([ [np.zeros((4*(L-1),4)),np.kron(np.eye(L-1), hopping)],
                                         [np.zeros((4,4*L))] ])     #upper diagonal part
    matrix = (matrix_diagonal + matrix_outside_diagonal + matrix_outside_diagonal.conj().T)
    return matrix

def Hamiltonian_A1u(t, k, mu, L, Delta):
    r"""Returns the H_k matrix for A1u model with:

    .. math::
        H_{A1u} = \frac{1}{2}\sum_k H_k
        
        H_k = \sum_n^L \vec{c}^\dagger_n\left[ 
            \xi_k\tau_z\sigma_0 +
            \Delta sin(k_y)\tau_x\sigma_y \right] +
            \sum_n^{L-1}\vec{c}^\dagger_n(-t\tau_z\sigma_0 + \frac{\Delta}{2i}\tau_x\sigma_x)\vec{c}_{n+1}
            + H.c.
            
       \vec{c} = (c_{k,\uparrow}, c_{k,\downarrow},c^\dagger_{-k,\downarrow},-c^\dagger_{-k,\uparrow})^T
    """
    chi_k = -mu - 2*t * np.cos(k)
    onsite = chi_k * np.kron(tau_z, sigma_0) + \
            Delta *np.sin(k)* np.kron(tau_x, sigma_y)
    hopping = -t*np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_x)
    matrix_diagonal = np.kron(np.eye(L), onsite)     #diagonal part of matrix
    matrix_outside_diagonal = np.block([ [np.zeros((4*(L-1),4)),np.kron(np.eye(L-1), hopping)],
                                         [np.zeros((4,4*L))] ])     #upper diagonal part
    matrix = (matrix_diagonal + matrix_outside_diagonal + matrix_outside_diagonal.conj().T)
    return matrix

def Junction_A1u_s(t, k, mu, L, Delta_0, t_J, phi):
    r"""Returns the array for the Hamiltonian of Josephson junction
     (TRITOPS-S) tilted in an angle theta for A1u simmetry.
    
    .. math::
        H = H_k^{S1} + H_k^{S2} + H_{J,k}
        
        H_{J,k} = t_J\vec{c}_{S1,k,L}^{\dagger}\left( 
            \frac{\tau^z+\tau^0}{2} e^{i\phi/2} + \frac{\tau^z-\tau^0}{2} e^{-i\phi/2}
            \right)\vec{c}_{S2,k,1} + H.c.
    """
    H_S1 = Hamiltonian_A1u(t, k, mu, L, Delta_0)
    H_S2 = Hamiltonian(t, k, mu, L, Delta_0, 0, 0, 0)
    block_diagonal_matrix = np.block([[H_S1, np.zeros((4*L,4*L))],
                             [np.zeros((4*L,4*L)), H_S2]]) 
    tau_phi = (np.kron((tau_z + np.eye(2))/2, np.eye(2))*np.exp(1j*phi/2)
                + np.kron((tau_z - np.eye(2))/2, np.eye(2))*np.exp(-1j*phi/2))
    block_diagonal_matrix[4*(L-1):4*L, 4*L:4*(L+1)] = t_J*tau_phi
    block_diagonal_matrix[4*L:4*(L+1), 4*(L-1):4*L] = t_J*tau_phi.conj().T
    return block_diagonal_matrix


def phi_spectrum(Junction, k_values, phi_values, **params):
    """Returns an array whose rows are the eigenvalues of the junction (with function Junction) for
    a definite phi_value given a fixed k_value.
    """
    eigenvalues = []
    for k_value in k_values:
        eigenvalues_k = []
        params["k"] = k_value
        for phi in phi_values:
            params["phi"] = phi
            H = Junction(**params)
            energies = np.linalg.eigvalsh(H)
            energies = list(energies)
            eigenvalues_k.append(energies)
        eigenvalues.append(eigenvalues_k)
    eigenvalues = np.array(eigenvalues)
    return eigenvalues

#%%

t = 1
t_J = 1
Delta = t/2
mu = -2
phi_values = np.linspace(0, 2*np.pi, 240)
#phi = np.linspace(0, 2*np.pi, 750)
#k = np.linspace(0, np.pi, 75)
k_values = np.linspace(0, 0.001*np.pi, 2)
#k = np.array([0, 0.01, 0.02])*np.pi
#k = np.linspace(-3, -, 5)

L = 50

params = dict(t=t, mu=mu, Delta_0=Delta,
              L=L, t_J=t_J)

E_phi = phi_spectrum(Junction_A1u_s, k_values, phi_values, **params)
print('\007')  # Ending bell

#%% Plotting for a given k
E_positive = E_phi[:, :, 4*L:]  #positive energy

for i in range(4*L):
    plt.plot(phi_values, E_positive[0, :, i])

plt.title(f"k={k_values[0]}")
plt.xlabel(r"$\phi$")
plt.ylabel(r"$E_k$")

#%% Plotting of total energy

total_energy_k = np.sum(E_positive, axis=2)
total_energy = np.sum(total_energy_k, axis=0) 
fig, ax = plt.subplots()
ax.plot(phi_values, total_energy)
