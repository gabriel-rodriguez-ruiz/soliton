# -*- coding: utf-8 -*-
"""
Created on Sun May 28 18:44:44 2023

@author: gabri
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
from hamiltonians import Hamiltonian_A1u_semi_infinite_sparse, charge_conjugation_operator_sparse_1D
from scipy.linalg import orth

L_x = 200
t = 1
Delta = 1
mu = -2    # topological phase if 0<mu<4
k_value = 0.1*np.pi
k = 4

mu_0 = np.abs(mu)
g_0 = np.sqrt(2*mu_0/(1-np.exp(-2*mu_0*L_x)))
g_left = lambda x : np.exp(-mu_0/Delta*np.abs(x))
x = np.arange(L_x)
v_0 = np.kron(np.array([g_left(x) for x in x]), 1/2*np.array([1, 1, -1j, -1j], dtype=complex))

H = Hamiltonian_A1u_semi_infinite_sparse(k=k_value, t=t, mu=mu, L_x=L_x, Delta=Delta)
eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H, k=k, sigma=0, v0=v_0) 
idx = eigenvalues_sparse.argsort()
eigenvalues_sparse_sorted = eigenvalues_sparse[idx]
eigenvectors_sparse_sorted = eigenvectors_sparse[:, idx]
eigenvalues = eigenvalues_sparse_sorted
eigenvectors = eigenvectors_sparse_sorted

eigenvectors_negative = scipy.linalg.orth(eigenvectors[:,:2]) #for negative eigenvalues
C = charge_conjugation_operator_sparse_1D(L_x)
A = C.dot(eigenvectors_negative[:,0].conj()) #positive eigenvector
is_charge_conjugation_A = np.allclose(H @ A, -eigenvalues[0]*A)
B = C.dot(eigenvectors_negative[:,1].conj()) #positive eigenvector
is_charge_conjugation_B = np.allclose(H @ B, -eigenvalues[1]*B)
# eigenvectors = np.append(eigenvectors_negative, A.reshape((len(A), 1)), axis=1)
# eigenvectors = np.append(eigenvectors, B.reshape((len(B), 1)), axis=1)

#%%
fig, ax = plt.subplots()
ax.plot(np.abs(eigenvectors[:,1]))