# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:05:24 2023

@author: gabri
"""
import numpy as np

class Hamiltonian:
    """A class for Bogoliubov-de-Gennes hamiltonians"""
    def __init__(self, L_x, L_y):
        self.L_x = L_x
        self.L_y = L_y
        self.matrix = self.get_matrix(L_x, L_y)
    def get_matrix(self, L_x, L_y):
        return np.zeros((4*(L_x)*L_y, 4*(L_x)*L_y), dtype=complex)
    def charge_conjugation(self):
        """
        Check if charge conjugation is present.

        Parameters
        ----------
        H : ndarray
            H_BdG Hamiltonian.

        Returns
        -------
        True or false depending if the symmetry is present or not.

        """
        L_x = self.L_x
        L_y = self.L_y
        def index(i, j, alpha, L_x, L_y):
            return alpha + 4*( L_y*(i-1) + j - 1)
        tau_y = np.array([[0, -1j], [1j, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        M = np.zeros((4*L_x*L_y, 4*L_x*L_y), dtype=complex)
        C = np.kron(tau_y, sigma_y)     #charge conjugation operator
        for i in range(1, L_x+1):
          for j in range(1, L_y+1):
            for alpha in range(4):
              for beta in range(4):
                M[index(i, j, alpha, L_x, L_y), index(i, j, beta, L_x, L_y)] = C[alpha, beta]   
        return np.all(np.linalg.inv(M) @ self.matrix @ M == -self.matrix.conj())