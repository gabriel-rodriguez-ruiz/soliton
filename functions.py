#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 12:05:43 2023

@author: gabriel
"""
import numpy as np
import scipy

# Pauli matrices
sigma_0 = np.eye(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
tau_0 = np.eye(2)
tau_x = np.array([[0, 1], [1, 0]])
tau_y = np.array([[0, -1j], [1j, 0]])
tau_z = np.array([[1, 0], [0, -1]])

# Spin operators
P = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  #particle projector
# S_x = np.kron(tau_z, sigma_x)
# S_y = np.kron(tau_z, sigma_y)
# S_z = np.kron(tau_0, sigma_z)

S_x = P.T@sigma_x@P         #I do not write the factor 1/2 because I should only normalize the particle part.
S_y = P.T@sigma_y@P
S_z = P.T@sigma_z@P

def spectrum(system, k_values, **params):
    """Returns an array whose rows are the eigenvalues of the system for
    a definite k. System should be a function that returns an array.
    """
    eigenvalues = []
    for k in k_values:
        params["k"] = k
        H = system(**params)
        energies = np.linalg.eigvalsh(H)
        energies = list(energies)
        eigenvalues.append(energies)
    eigenvalues = np.array(eigenvalues)
    return eigenvalues

def mean_spin(state):
    """Returns a 1D-array of length 3 with the spin mean value of the state.
    State should be a vector of shape (4,1).
    """
    #state = state/np.linalg.norm(state)
    spin_mean_value = np.concatenate([state.T.conj()@S_x@state,
                                      state.T.conj()@S_y@state,
                                      state.T.conj()@S_z@state])
    #return np.real(spin_mean_value/np.linalg.norm(spin_mean_value))
    return np.real(spin_mean_value)


def mean_spin_xy(state):
    N_x, N_y, N_z = np.shape(state)     #N_z is always 4
    spin_mean_value = np.zeros((N_x, N_y, 3))
    for i in range(N_x):
        for j in range(N_y):
            for k in range(3):
                spin_mean_value[i, j, k] = mean_spin(np.reshape(state[i,j,:], (4,1)))[k][0]
    return spin_mean_value

def get_components(state, L_x, L_y):
    """
    Get the components of the state: creation_up,
    creation_down, destruction_down, destruction_up for a given
    column state. Returns an array of shape (L_y, L_x)
    """
    destruction_up = state[0::4].reshape((L_x, L_y))
    destruction_down = state[1::4].reshape((L_x, L_y))
    creation_down = state[2::4].reshape((L_x, L_y))
    creation_up = state[3::4].reshape((L_x, L_y))
    return (np.flip(destruction_up.T, axis=0),
            np.flip(destruction_down.T, axis=0),
            np.flip(creation_down.T, axis=0),
            np.flip(creation_up.T, axis=0))

def probability_density(Hamiltonian, L_x, L_y, index):
    """
    Returns the probability density of a 2D system given a matrix Hamiltonian and the index of the zero mode (0<=index<=3).
    The matrix element order are analogous to the real space grid.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(Hamiltonian)
    #zero_modes = eigenvectors[:, 2*(L_x*L_y-1):2*(L_x*L_y+1)]      #4 (2) modes with zero energy (with Zeeman)
    a, b, c, d = get_components(eigenvectors[:,2*(L_x*L_y-1)+index], L_x, L_y)
    probability_density = np.abs(a)**2 + np.abs(b)**2 + np.abs(c)**2 + np.abs(d)**2
    return probability_density, eigenvalues, eigenvectors

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

def phi_spectrum_sparse_single_step(Hamiltonian, Phi, t, mu, L_x, L_y, Delta, t_J):
    """
    Returns the phi spectrum for the six lowest energies.
    """
    energies = []
    for Phi_value in Phi:
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(Hamiltonian(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi_value), k=4, sigma=0) 
        energies.append(eigenvalues)
    return energies
