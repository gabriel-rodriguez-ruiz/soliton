#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 17:08:13 2023

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

def index(i, j, alpha, L_x, L_y):
  r"""Return the index of basis vector given the site (i,j)
  and spin index alpha in {0,1,2,3} for i in {1, ..., L_x} and
  j in {1, ..., L_y}. The site (1,1) corresponds to the lower left real space position.
  
  .. math ::
     (c_{11}, c_{12}, ..., c_{1L_y}, c_{21}, ..., c_{L_xL_y})^T
     
     \text{real space}
     
     (c_{1L_y} &... c_{L_xL_y})
                      
     (c_{11} &... c_{L_x1})
  """
  if (i>L_x or j>L_y):
      raise Exception("Site index should not be greater than samplesize.")
  return alpha + 4*( L_y*(i-1) + j - 1)
    
def phase_single_soliton(phi_external, y, y_0):
    r"""Step function for the phase single soliton.
    y should be an ndarray
        .. math ::
            \phi(y) = 2\pi\theta(y-y_0)
            
            \theta(0) =1/2
    """
    return phi_external + 2*np.pi*np.heaviside(y-y_0, 1/2)

def phase_double_soliton(phi_external, y, y_0, y_1):
    r"""Profile function for the phase double soliton.
    y should be an ndarray
        .. math ::
            \phi(y) = 2\pi\left(\theta(y-y_0) - \theta(y-L) \right)
            
            \theta(0) =1/2
    """
    return phi_external + 2*np.pi*( np.heaviside(y-y_0, 1/2) - np.heaviside(y-y_1, 1/2) )
