#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 11:51:06 2023

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from hamiltonians import Hamiltonian_A1u_junction_sparse_k, Hamiltonian_A1u_S_junction_sparse_k

k = 0.01*np.pi

t = 1
Delta = 1
mu = -2  #-2
t_J = t   #   t
L = 200      #L_y//2
# n = 100 #number of eigenvalues
n = 100 #number of eigenvalues
index = np.arange(n)

Phi = np.linspace(0, 2*np.pi, 100)
eigenvalues = np.zeros((len(Phi), n))

for i, phi in enumerate(Phi):
    # H = Hamiltonian_A1u_junction_sparse_k(t=t, k=k, mu=mu, L=L, Delta=Delta, phi=phi, t_J=t_J)
    H = Hamiltonian_A1u_S_junction_sparse_k(t=t, k=k, mu=mu, L=L, Delta=Delta, phi=phi, t_J=t_J)
    # eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H, k=n, sigma=0) 
    eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H, k=n, sigma=0.03)     
    eigenvalues[i] = eigenvalues_sparse

fig, ax = plt.subplots()
ax.set_xlabel(r"$\Phi$")
ax.set_ylabel(r"$E_k$")
plt.title(r"$k=$"+f"{k:.2}")
for i in index:
    ax.plot(Phi, eigenvalues[:, i], "ob", markersize=1)