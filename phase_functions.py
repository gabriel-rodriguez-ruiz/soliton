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
            
            \theta(0) = 1/2
    """
    return phi_external + 2*np.pi*np.heaviside(y-y_0, 1/2)

def phase_soliton_antisoliton(phi_external, y, y_0, y_1):
    r"""Profile function for the phase soliton-antisoliton.
    y should be an ndarray
        .. math ::
            \phi(y) = \phi_{ext} + 2\pi\left(\theta(y-y_0) - \theta(y-y_1) \right)
            
            \theta(0) = 1/2
    """
    return phi_external + 2*np.pi*( np.heaviside(y-y_0, 1/2) - np.heaviside(y-y_1, 1/2) )

def phase_antisoliton_soliton(phi_external, y, y_0, y_1):
    r"""Profile function for the phase soliton-antisoliton.
    y should be an ndarray
        .. math ::
            \phi(y) = \phi_{ext} + 2\pi - 2\pi\left(\theta(y-y_0) - \theta(y-y_1) \right)
            
            \theta(0) = 1/2
    """
    return phi_external + 2*np.pi - 2*np.pi*( np.heaviside(y-y_0, 1/2) - np.heaviside(y-y_1, 1/2) )


def phase_single_soliton_arctan(phi_external, y, y_0, lambda_J):
    r"""Step function for the phase single soliton.
    y should be an ndarray
        .. math ::
            \phi(y) = 2\pi\theta(y-y_0)
            
            \theta(0) =1/2
    """
    return phi_external + 4*np.arctan(np.exp((y-y_0)/lambda_J))

def phase_soliton_antisoliton_arctan(phi_external, y, y_0, y_1, lambda_J):
    r"""Profile function for the phase soliton-antisoliton.
    y should be an ndarray
        .. math ::
            \phi(y) = \phi_{ext}+ 4\arctan(e^{(y-y_0)/\lambda}) - 4\arctan(e^{(y-y_1)/\lambda})
            
    """
    return phi_external + 4*np.arctan(np.exp((y-y_0)/lambda_J)) - 4*np.arctan(np.exp((y-y_1)/lambda_J))

def phase_soliton_soliton_arctan(phi_external, y, y_0, y_1, lambda_J):
    r"""Profile function for the phase soliton-antisoliton.
    y should be an ndarray
        .. math ::
            \phi(y) = \phi_{ext}+ 4\arctan(e^{(y-y_0)/\lambda}) + 4\arctan(e^{(y-y_1)/\lambda})
            
    """
    return phi_external + 4*np.arctan(np.exp((y-y_0)/lambda_J)) + 4*np.arctan(np.exp((y-y_1)/lambda_J))

def phase_soliton_antisoliton_arctan_A1u_S_around_zero(phi_external, y, y_0, y_1, lambda_J):
    r"""Profile function for the phase soliton-antisoliton.
    y should be an ndarray
        .. math ::
            \phi(y) = 2\pi\left(\theta(y-y_0) - \theta(y-L) \right)
            
            \theta(0) =1/2
    """
    return phi_external + 2*np.arctan(np.exp(2*(y-y_0)/lambda_J)) - 3*np.pi/2 + 2*np.arctan(np.exp(-2*(y-y_1)/lambda_J))

def phase_soliton_antisoliton_arctan_A1u_S_around_pi(phi_external, y, y_0, y_1, lambda_J):
    r"""Profile function for the phase soliton-antisoliton.
    y should be an ndarray
        .. math ::
            \phi(y) = 2\pi\left(\theta(y-y_0) - \theta(y-L) \right)
            
            \theta(0) =1/2
    """
    return phi_external + 2*np.arctan(np.exp(2*(y-y_0)/lambda_J)) - np.pi/2 + 2*np.arctan(np.exp(-2*(y-y_1)/lambda_J))

def phase_single_soliton_S(phi_external, y, y_0, phi_0, lambda_J):
    r"""Step function for the phase single soliton.
    y should be an ndarray
        .. math ::
            \begin{eqnarray}
                &\phi_1(x) = 2\arctan \left[ \tan\left(\frac{\phi_{0}}{2}\right)\tanh\left( \frac{|\sin \phi_{0}|}{2\sqrt{cos(\phi_0)}} \frac{x-x_1}{\lambda_J} \right) \right]\\
                &-\phi_0 \leq\phi\leq \phi_0\\
                &\phi_2(x) = \pi - 2\arctan \left[ \tan\left(\frac{\phi_{0}-\pi}{2}\right)\tanh\left( \sqrt{E_0}|\sin \phi_{0}| \frac{x-x_2}{\lambda_J} \right) \right]\\
                &\phi_0 \leq\phi\leq 2\pi-\phi_0
            \end{eqnarray}
    """
    return [phi_external + 2*np.arctan(np.tan(phi_0/2)*np.tanh(1/(2*np.sqrt(np.cos(phi_0)))*np.abs(np.sin(phi_0))*(y-y_0)/lambda_J)),
            phi_external + np.pi - 2*np.arctan(np.tan((phi_0-np.pi)/2)*np.tanh(1/(2*np.sqrt(np.cos(phi_0)))*np.abs(np.sin(phi_0))*(y-y_0)/lambda_J))]

def phase_soliton_antisoliton_S_around_zero(phi_external, phi_eq, y, y_0, y_1):
    r"""Profile function for the phase soliton-antisoliton.
    y should be an ndarray
        .. math ::
            \phi(y) = \phi_{ext} - \phi_{eq} + 2\phi_{eq}\left(\theta(y-y_0) - \theta(y-y_1) \right)
            
            \theta(0) = 1/2
    """
    return phi_external - phi_eq + 2*phi_eq*( np.heaviside(y-y_0, 1/2) - np.heaviside(y-y_1, 1/2) )

def phase_soliton_antisoliton_S_around_pi(phi_external, phi_eq, y, y_0, y_1):
    r"""Profile function for the phase soliton-antisoliton.
    y should be an ndarray
        .. math ::
            \phi(y) = \phi_{ext} + \phi_{eq} + 2(\pi-\phi_{eq})\left(\theta(y-y_0) - \theta(y-y_1) \right)
            
            \theta(0) = 1/2
    """
    return phi_external + phi_eq + 2*(np.pi-phi_eq)*( np.heaviside(y-y_0, 1/2) - np.heaviside(y-y_1, 1/2) )
