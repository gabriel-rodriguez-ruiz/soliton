# -*- coding: utf-8 -*-
"""
Created on Thu May 25 15:50:34 2023

@author: gabri
"""
import numpy as np
import matplotlib.pyplot as plt
from hamiltonians import Hamiltonian_soliton_A1u_sparse, charge_conjugation_operator_sparse
from functions import get_components
import scipy

L_x = 200
L_y = 200
t = 1
Delta = 1
mu = -2  #-2
Phi = 0.1*np.pi  #height of the phase soliton around flux pi
t_J = 1   #t
L = 30      #L_y//2
k = 4   #number of eigenvalues
Delta_Z = 0
theta = np.pi/2
phi = 0

H = Hamiltonian_soliton_A1u_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi, L=L)
eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H, k=k, sigma=0) 
idx = eigenvalues_sparse.argsort()
eigenvalues_sparse = eigenvalues_sparse[idx]
eigenvectors_sparse = eigenvectors_sparse[:, idx]

from scipy.linalg import orth
eigenvectors_negative = scipy.linalg.orth(eigenvectors_sparse[:,:2]) #for negative eigenvalues

C = charge_conjugation_operator_sparse(L_x, L_y)
A = C.dot(eigenvectors_negative[:,0].conj()) #positive eigenvector
is_charge_conjugation_A = np.allclose(H @ A, -eigenvalues_sparse[0]*A)
B = C.dot(eigenvectors_negative[:,1].conj()) #positive eigenvector
is_charge_conjugation_B = np.allclose(H @ B, -eigenvalues_sparse[1]*B)
eigenvectors = np.append(eigenvectors_negative, A.reshape((len(A), 1)), axis=1)
eigenvectors = np.append(eigenvectors, B.reshape((len(B), 1)), axis=1)

