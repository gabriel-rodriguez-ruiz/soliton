#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 11:30:38 2023

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
     
     \text{index}(i,j,\alpha,L_x,L_y) = \alpha + 4\left(L_y(i-1) + j-1\right)
     
     \text{real space}
     
     (c_{1L_y} &... c_{L_xL_y})
                      
     (c_{11} &... c_{L_x1})
  """
  if (i>L_x or j>L_y):
      raise Exception("Site index should not be greater than samplesize.")
  return alpha + 4*( L_y*(i-1) + j - 1)

def index_k(i, alpha, L):
  r"""Return the index of basis vector given the site (i,j)
  and spin index alpha in {0,1,2,3} for i in {1, ..., L}. The site 1 corresponds to the left real space position.
  
  .. math ::
     (c_{1}, c_{2}, ..., c_{L})^T
     
     \text{index}(i,j,\alpha,L_x,L_y) = \alpha + 4(i-1)
     
     \text{real space}
     
     (c_{1} &... c_{L})
     """
  if (i>L):
      raise Exception("Site index should not be greater than samplesize.")
  return alpha + 4*(i-1)

def index_periodic(i, j, alpha, L_x, L_y):
  r"""Return the index of basis vector given the site (i,j)
  and spin index alpha in {0,1,2,3} for i in {1, ..., L_x} and
  j in {1, ..., L_y+1}. The site (1,1) corresponds to the lower left real space position.
    
  .. math ::
     (c_{11}, c_{12}, ..., c_{1L_y}, c_{21}, ..., c_{L_xL_y})^T
     
     c_{i1} = c_{i,L_y+1} \text{(periodic boundary condition)}
     
     \text{index}(i,j,\alpha,L_x,L_y) = \alpha + 4\left(L_y(i-1) + j-1 -\delta_{j,L_y+1}L_y\right)
     
     \text{real space}
     
     (c_{1L_y} &... c_{L_xL_y})
                      
     (c_{11} &... c_{L_x1})
  """
  if (i>L_x or j>L_y+1):
      raise Exception("Site index should not be greater than samplesize.")
  if j==L_y+1:
      return alpha + 4*( L_y*(i-1) + j - 1 - L_y)
  else:
      return alpha + 4*( L_y*(i-1) + j - 1)

def index_periodic_in_x(i, j, alpha, L_x, L_y):
  r"""Return the index of basis vector given the site (i,j)
  and spin index alpha in {0,1,2,3} for i in {1, ..., L_x} and
  j in {1, ..., L_y+1}. The site (1,1) corresponds to the lower left real space position.
    
  .. math ::
     (c_{11}, c_{12}, ..., c_{1L_y}, c_{21}, ..., c_{L_xL_y})^T
     
     c_{1j} = c_{L_x+1,j} \text{(periodic boundary condition)}
     
     \text{index}(i,j,\alpha,L_x,L_y) = \alpha + 4\left(L_y(i-1) + j-1 -\delta_{i,L_x+1}L_yL_x\right)
     
     \text{real space}
     
     (c_{1L_y} &... c_{L_xL_y})
                      
     (c_{11} &... c_{L_x1})
  """
  if (i>L_x+1 or j>L_y):
      raise Exception("Site index should not be greater than samplesize.")
  if i==L_x+1:
      return alpha + 4*(j - 1)
  else:
      return alpha + 4*( L_y*(i-1) + j - 1)

def Hamiltonian_A1u_sparse(t, mu, L_x, L_y, Delta):
    r"""Return the matrix for A1u model with:
        
    .. math ::
       \vec{c_{n,m}} = (c_{n,m,\uparrow},
                        c_{n,m,\downarrow},
                        c^\dagger_{n,m,\downarrow},
                        -c^\dagger_{n,m,\uparrow})^T
       
       H = \frac{1}{2} \sum_n^{L_x} \sum_m^{L_y} (-\mu \vec{c}^\dagger_{n,m} \tau_z\sigma_0  \vec{c}_{n,m}) +
           \frac{1}{2} \sum_n^{L_x-1} \sum_m^{L_y} \left( \vec{c}^\dagger_{n,m}\left[ 
            -t\tau_z\sigma_0 -
            i\frac{\Delta}{2} \tau_x\sigma_x \right] \vec{c}_{n+1,m} + H.c. \right) +
           \frac{1}{2} \sum_n^{L_x} \sum_m^{L_y-1} \left( \vec{c}^\dagger_{n,m}\left[ 
            -t\tau_z\sigma_0 -
            i\frac{\Delta}{2} \tau_x\sigma_y \right] \vec{c}_{n,m+1} + H.c. \right) 
    """
    M = scipy.sparse.lil_matrix((4*L_x*L_y, 4*L_x*L_y), dtype=complex)
    onsite = -mu/4 * np.kron(tau_z, sigma_0)   # para no duplicar al sumar la traspuesta
    for i in range(1, L_x+1):
      for j in range(1, L_y+1):
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j, beta, L_x, L_y)] = onsite[alpha, beta]   
    hopping_x = -t/2 * np.kron(tau_z, sigma_0) - 1j*Delta/4 * np.kron(tau_x, sigma_x)
    for i in range(1, L_x):
      for j in range(1, L_y+1):    
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i+1, j, beta, L_x, L_y)] = hopping_x[alpha, beta]
    hopping_y = -t/2 * np.kron(tau_z, sigma_0) - 1j*Delta/4 * np.kron(tau_x, sigma_y)
    for i in range(1, L_x+1):
      for j in range(1, L_y): 
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j+1, beta, L_x, L_y)] = hopping_y[alpha, beta]
    return scipy.sparse.csr_matrix(M + M.conj().T)

def Hamiltonian_A1u_junction_sparse(t, mu, L_x, L_y, Delta, t_J, Phi):
    r"""Return the matrix for A1u model in a junction with a superconductor with:
    
       .. math ::
           \vec{c_{n,m}} = (c_{n,m,\uparrow},
                            c_{n,m,\downarrow},
                            c^\dagger_{n,m,\downarrow},
                            -c^\dagger_{n,m,\uparrow})^T
           
           H = H_{A1u, 1} + H_{A1u, 2} + H_J
           
           H_{A1u} = \frac{1}{2} \sum_{n=1}^{L_x} \sum_{m=1}^{L_y} (-\mu \vec{c}^\dagger_{n,m} \tau_z\sigma_0  \vec{c}_{n,m}) +
               \frac{1}{2} \sum_{n=1}^{L_x-1} \sum_{m=1}^{L_y} \left( \vec{c}^\dagger_{n,m}\left[ 
                -t\tau_z\sigma_0 -
                i\frac{\Delta}{2} \tau_x\sigma_x \right] \vec{c}_{n+1,m} + H.c. \right) +
               \frac{1}{2} \sum_{n=1}^{L_x} \sum_{m=1}^{L_y-1} \left( \vec{c}^\dagger_{n,m}\left[ 
                -t\tau_z\sigma_0 -
                i\frac{\Delta}{2} \tau_x\sigma_y \right] \vec{c}_{n,m+1} + H.c. \right) 
         
           H_J = t_J/2\sum_m^{L_y}[\vec{c}_{L_x-1,m}^\dagger(cos(\phi/2)\tau_z\sigma_0+isin(\phi/2)\tau_0\sigma_0)\vec{c}_{L_x,m}+H.c.]
    """
    phi = Phi[::-1]  # I have inverted the y axis
    M = scipy.sparse.lil_matrix((4*L_x*L_y, 4*L_x*L_y), dtype=complex)
    onsite_A1u = -mu/4 * np.kron(tau_z, sigma_0)   # para no duplicar al sumar la traspuesta
    for i in range(1, L_x+1):
      for j in range(1, L_y+1):
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j, beta, L_x, L_y)] = onsite_A1u[alpha, beta]   
    hopping_x_A1u = 1/2*( -t * np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_x) )
    for i in range(1, L_x//2):
      for j in range(1, L_y+1):    
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i+1, j, beta, L_x, L_y)] = hopping_x_A1u[alpha, beta]
    for i in range(L_x//2+1, L_x):
      for j in range(1, L_y+1):    
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i+1, j, beta, L_x, L_y)] = hopping_x_A1u[alpha, beta]
    hopping_y_A1u = 1/2* ( -t/2 * np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_y) )
    for i in range(1, L_x+1):
      for j in range(1, L_y): 
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j+1, beta, L_x, L_y)] = hopping_y_A1u[alpha, beta]
    for j in range(1, L_y+1):
        hopping_junction_x = t_J/2 * (np.cos(phi[j-1]/2)*np.kron(tau_z, sigma_0)+1j*np.sin(phi[j-1]/2)*np.kron(tau_0, sigma_0))
        for alpha in range(4):
            for beta in range(4):
                M[index(L_x//2, j, alpha, L_x, L_y), index(L_x//2+1, j, beta, L_x, L_y)] = hopping_junction_x[alpha, beta]
    return scipy.sparse.csr_matrix(M + M.conj().T)

def Hamiltonian_A1u_S(t, mu, L_x, L_y, Delta, t_J, Phi):
   r"""Return the matrix for A1u model in a junction with a superconductor:

   .. math ::
      \vec{c_{n,m}} = (c_{n,m,\uparrow},
                       c_{n,m,\downarrow},
                       c^\dagger_{n,m,\downarrow},
                       -c^\dagger_{n,m,\uparrow})^T
             
          H = H_{A1u} + H_{S} + H_J
          
          H_{A1u} = \frac{1}{2} \sum_{n=1}^{L_x} \sum_{m=1}^{L_y} (-\mu \vec{c}^\dagger_{n,m} \tau_z\sigma_0  \vec{c}_{n,m}) +
              \frac{1}{2} \sum_{n=1}^{L_x-1} \sum_{m=1}^{L_y} \left( \vec{c}^\dagger_{n,m}\left[ 
               -t\tau_z\sigma_0 -
               i\frac{\Delta}{2} \tau_x\sigma_x \right] \vec{c}_{n+1,m} + H.c. \right) +
              \frac{1}{2} \sum_{n=1}^{L_x} \sum_{m=1}^{L_y-1} \left( \vec{c}^\dagger_{n,m}\left[ 
               -t\tau_z\sigma_0 -
               i\frac{\Delta}{2} \tau_x\sigma_y \right] \vec{c}_{n,m+1} + H.c. \right) 
          
           H_{S} = \frac{1}{2}\sum_m^{L_y} \mathbf{c}^\dagger_{L_x,m}\left[ 
                       -\mu\tau_z\sigma_0 + \Delta_0\tau_x\sigma_0 \right] \mathbf{c}_{L_x,m}\nonumber
           				+ \frac{1}{2}
                       \sum_m^{L_y-1}\left[\mathbf{c}^\dagger_{L_x,m}\left(-t\tau_z\sigma_0 \right)\mathbf{c}_{L_x,m+1}
                       + H.c.\right]
           
          H_J = t_J/2\sum_m^{L_y}[\vec{c}_{L_x-1,m}^\dagger(cos(\phi/2)\tau_z\sigma_0+isin(\phi/2)\tau_0\sigma_0)\vec{c}_{L_x,m}+H.c.]
   """
   phi = Phi[::-1]  # I have inverted the y axis
   M = scipy.sparse.lil_matrix((4*(L_x)*L_y, 4*(L_x)*L_y), dtype=complex)
   onsite_A1u = -mu/4 * np.kron(tau_z, sigma_0)   # para no duplicar al sumar la traspuesta
   for i in range(1, L_x):
     for j in range(1, L_y+1):
       for alpha in range(4):
         for beta in range(4):
           M[index(i, j, alpha, L_x, L_y), index(i, j, beta, L_x, L_y)] = onsite_A1u[alpha, beta]   
   onsite_S = 1/4*( -mu * np.kron(tau_z, sigma_0) + Delta*np.kron(tau_x, sigma_0) )
   for j in range(1, L_y+1):
     for alpha in range(4):
       for beta in range(4):
         M[index(L_x, j, alpha, L_x, L_y), index(L_x, j, beta, L_x, L_y)] = onsite_S[alpha, beta]
   hopping_x_A1u = 1/2*(-t * np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_x) )
   for i in range(1, L_x-1):
     for j in range(1, L_y+1):    
       for alpha in range(4):
         for beta in range(4):
           M[index(i, j, alpha, L_x, L_y), index(i+1, j, beta, L_x, L_y)] = hopping_x_A1u[alpha, beta]
   hopping_y_A1u = 1/2*( -t * np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_y) )
   for i in range(1, L_x):
     for j in range(1, L_y+1): 
       for alpha in range(4):
         for beta in range(4):
           M[index(i, j, alpha, L_x, L_y), index(i, j+1, beta, L_x, L_y)] = hopping_y_A1u[alpha, beta]
   hopping_y_S = -t/2 * np.kron(tau_z, sigma_0)    
   for j in range(1, L_y+1): 
     for alpha in range(4):
       for beta in range(4):
         M[index(L_x, j, alpha, L_x, L_y), index(L_x, j+1, beta, L_x, L_y)] = hopping_y_S[alpha, beta]
   for j in range(1, L_y+1):
       hopping_junction_x = t_J/2 * (np.cos(phi[j-1]/2)*np.kron(tau_z, sigma_0)+1j*np.sin(phi[j-1]/2)*np.kron(tau_0, sigma_0))
       for alpha in range(4):
           for beta in range(4):
               M[index(L_x-1, j, alpha, L_x, L_y), index(L_x, j, beta, L_x, L_y)] = hopping_junction_x[alpha, beta]
   return scipy.sparse.csr_matrix(M + M.conj().T)

def Zeeman(theta, phi, Delta_Z, L_x, L_y):
    r""" Return the Zeeman Hamiltonian matrix in 2D.
    
    .. math::
        H_Z = \frac{\Delta_Z}{2} \sum_n^{L_x} \sum_m^{L_y} \vec{c}^\dagger_{n,m}
        \tau_0(\cos(\varphi)\sin(\theta)\sigma_x + \sin(\varphi)\sin(\theta)\sigma_y + \cos(\theta)\sigma_z)\vec{c}_{n,m}
    
        \vec{c}_{n,m} = (c_{n,m,\uparrow},
                         c_{n,m,\downarrow},
                         c^\dagger_{n,m,\downarrow},
                         -c^\dagger_{n,m,\uparrow})^T
    """
    M = scipy.sparse.lil_matrix((4*L_x*L_y, 4*L_x*L_y), dtype=complex)
    onsite = Delta_Z/2*( np.cos(phi)*np.sin(theta)*np.kron(tau_0, sigma_x) +
                        np.sin(phi)*np.sin(theta)*np.kron(tau_0, sigma_y) +
                        np.cos(theta)*np.kron(tau_0, sigma_z))
    for i in range(1, L_x+1):
      for j in range(1, L_y+1):
          for alpha in range(4):
              for beta in range(4):
                  M[index(i, j, alpha, L_x, L_y), index(i, j, beta, L_x, L_y)] = onsite[alpha, beta]
    return scipy.sparse.csr_matrix(M)

def charge_conjugation_operator(L_x, L_y):
    """
    Return the charge conjugation operator.
    """
    M = np.zeros((4*L_x*L_y, 4*L_x*L_y), dtype=complex)
    C = np.kron(tau_y, sigma_y)     #charge conjugation operator
    for i in range(1, L_x+1):
      for j in range(1, L_y+1):
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j, beta, L_x, L_y)] = C[alpha, beta]   
    return M

def charge_conjugation_operator_sparse(L_x, L_y):
    """
    Return the charge conjugation operator in a sparse way.
    """
    M = scipy.sparse.lil_matrix((4*L_x*L_y, 4*L_x*L_y), dtype=complex)
    C = np.kron(tau_y, sigma_y)     #charge conjugation operator
    for i in range(1, L_x+1):
      for j in range(1, L_y+1):
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j, beta, L_x, L_y)] = C[alpha, beta]   
    return M

def charge_conjugation(H, L_x, L_y):
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
    M = charge_conjugation_operator(L_x, L_y)
    return np.all(np.linalg.inv(M) @ H @ M == -H.conj())

def Hamiltonian_A1u_semi_infinite_sparse(k, t, mu, L_x, Delta):
    r"""Returns the H_k matrix for A1u model with:

    .. math::
        H_{A1u} = \frac{1}{2}\sum_k H_{A1u,k}
        
        H_{A1u,k} = \sum_n^L \vec{c}^\dagger_n\left[ 
            \xi_k\tau_z\sigma_0 +
            \Delta sin(k_y)\tau_x\sigma_y \right] +
            \sum_n^{L-1}\left(\vec{c}^\dagger_n(-t\tau_z\sigma_0 + \frac{\Delta}{2i}\tau_x\sigma_x)\vec{c}_{n+1}
            + H.c. \right)
            
        """
    M = scipy.sparse.lil_matrix((4*L_x, 4*L_x), dtype=complex)
    chi_k = -mu-2*t*np.cos(k)
    onsite = 1/4*( chi_k * np.kron(tau_z, sigma_0) + Delta*np.sin(k)*np.kron(tau_x, sigma_y) )  # para no duplicar al sumar la traspuesta
    for i in range(1, L_x+1):
        for alpha in range(4):
            for beta in range(4):
                M[index_k(i, alpha, L_x), index_k(i, beta, L_x)] = onsite[alpha, beta] 
    hopping = 1/2*(-t * np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_x) )
    for i in range(1, L_x):
        for alpha in range(4):
            for beta in range(4):
                M[index_k(i, alpha, L_x), index_k(i+1, beta, L_x)] = hopping[alpha, beta]
    return scipy.sparse.csr_matrix(M + M.conj().T)

def Hamiltonian_A1u_semi_infinite(k, t, mu, L_x, Delta):
    r"""Returns the H_k matrix for A1u model with:

    .. math::
        H_{A1u} = \frac{1}{2}\sum_k H_k
        
        H_k = \sum_n^L \vec{c}^\dagger_n\left[ 
            \xi_k\tau_z\sigma_0 +
            \Delta sin(k_y)\tau_x\sigma_y \right] \vec{c}_n +
            \sum_n^{L-1}\vec{c}^\dagger_n(-t\tau_z\sigma_0 + \frac{\Delta}{2i}\tau_x\sigma_x)\vec{c}_{n+1}
            + H.c.
        
        \xi_k = -2t\cos(k)-\mu    
        
       \vec{c} = (c_{k,\uparrow}, c_{k,\downarrow},c^\dagger_{-k,\downarrow},-c^\dagger_{-k,\uparrow})^T
    """
    M = np.zeros((4*L_x, 4*L_x), dtype=complex)
    onsite = 1/4*((-mu - 2*t*np.cos(k)) * np.kron(tau_z, sigma_0) + Delta*np.sin(k)*np.kron(tau_x, sigma_y) )   # para no duplicar al sumar la traspuesta
    for i in range(1, L_x+1):
        for alpha in range(4):
            for beta in range(4):
                M[index_k(i, alpha, L_x), index_k(i, beta, L_x)] = onsite[alpha, beta] 
    hopping = -t/2 * np.kron(tau_z, sigma_0) - 1j*Delta/4 * np.kron(tau_x, sigma_x)
    for i in range(1, L_x):
        for alpha in range(4):
            for beta in range(4):
                M[index_k(i, alpha, L_x), index_k(i+1, beta, L_x)] = hopping[alpha, beta]
    return (M + M.conj().T)


def charge_conjugation_operator_sparse_1D(L_x):
    """
    Return the charge conjugation operator in a sparse way.
    """
    M = scipy.sparse.lil_matrix((4*L_x, 4*L_x), dtype=complex)
    C = np.kron(tau_y, sigma_y)     #charge conjugation operator
    for i in range(1, L_x+1):
        for alpha in range(4):
            for beta in range(4):
                M[index_k(i, alpha, L_x), index_k(i, beta, L_x)] = C[alpha, beta]   
    return M

def Hamiltonian_A1u_junction_sparse_periodic(t, mu, L_x, L_y, Delta, t_J, Phi):
    r"""Return the matrix for A1u model in a junction with a superconductor with:
    
       .. math ::
           \vec{c_{n,m}} = (c_{n,m,\uparrow},
                            c_{n,m,\downarrow},
                            c^\dagger_{n,m,\downarrow},
                            -c^\dagger_{n,m,\uparrow})^T
           
           H = H_{A1u, 1} + H_{A1u, 2} + H_J
           
           H_{A1u} = \frac{1}{2} \sum_{n=1}^{L_x} \sum_{m=1}^{L_y} (-\mu \vec{c}^\dagger_{n,m} \tau_z\sigma_0  \vec{c}_{n,m}) +
               \frac{1}{2} \sum_{n=1}^{L_x-1} \sum_{m=1}^{L_y} \left( \vec{c}^\dagger_{n,m}\left[ 
                -t\tau_z\sigma_0 -
                i\frac{\Delta}{2} \tau_x\sigma_x \right] \vec{c}_{n+1,m} + H.c. \right) +
               \frac{1}{2} \sum_{n=1}^{L_x} \sum_{m=1}^{L_y-1} \left( \vec{c}^\dagger_{n,m}\left[ 
                -t\tau_z\sigma_0 -
                i\frac{\Delta}{2} \tau_x\sigma_y \right] \vec{c}_{n,m+1} + H.c. \right) 
         
           H_J = t_J/2\sum_m^{L_y}[\vec{c}_{L_x-1,m}^\dagger(cos(\phi/2)\tau_z\sigma_0+isin(\phi/2)\tau_0\sigma_0)\vec{c}_{L_x,m}+H.c.]
    """
    phi = Phi[::-1]  # I have inverted the y axis
    M = scipy.sparse.lil_matrix((4*(L_x)*L_y, 4*(L_x)*L_y), dtype=complex)
    onsite_A1u = -mu/4 * np.kron(tau_z, sigma_0)   # para no duplicar al sumar la traspuesta
    for i in range(1, L_x+1):
      for j in range(1, L_y+1):
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j, beta, L_x, L_y)] = onsite_A1u[alpha, beta]   
    hopping_x_A1u = 1/2*( -t * np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_x) )
    for i in range(1, L_x//2):
      for j in range(1, L_y+1):    
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i+1, j, beta, L_x, L_y)] = hopping_x_A1u[alpha, beta]
    for i in range(L_x//2+1, L_x):
      for j in range(1, L_y+1):    
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i+1, j, beta, L_x, L_y)] = hopping_x_A1u[alpha, beta]
    hopping_y_A1u = 1/2* ( -t/2 * np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_y) )
    for i in range(1, L_x+1):
      for j in range(1, L_y): 
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j+1, beta, L_x, L_y)] = hopping_y_A1u[alpha, beta]
    for j in range(1, L_y+1):
        hopping_junction_x = t_J/2 * (np.cos(phi[j-1]/2)*np.kron(tau_z, sigma_0)+1j*np.sin(phi[j-1]/2)*np.kron(tau_0, sigma_0))
        for alpha in range(4):
            for beta in range(4):
                M[index(L_x//2, j, alpha, L_x, L_y), index(L_x//2+1, j, beta, L_x, L_y)] = hopping_junction_x[alpha, beta]
    return scipy.sparse.csr_matrix(M + M.conj().T)

def Hamiltonian_A1u_S_periodic(t, mu, L_x, L_y, Delta, t_J, Phi):
    r"""Return the matrix for A1u model in a junction with a superconductor with periodicity in y:

    .. math ::
       \vec{c_{n,m}} = (c_{n,m,\uparrow},
                        c_{n,m,\downarrow},
                        c^\dagger_{n,m,\downarrow},
                        -c^\dagger_{n,m,\uparrow})^T
              
           H = H_{A1u} + H_{S} + H_J
           
           H_{A1u} = \frac{1}{2} \sum_{n=1}^{L_x} \sum_{m=1}^{L_y} (-\mu \vec{c}^\dagger_{n,m} \tau_z\sigma_0  \vec{c}_{n,m}) +
               \frac{1}{2} \sum_{n=1}^{L_x-1} \sum_{m=1}^{L_y} \left( \vec{c}^\dagger_{n,m}\left[ 
                -t\tau_z\sigma_0 -
                i\frac{\Delta}{2} \tau_x\sigma_x \right] \vec{c}_{n+1,m} + H.c. \right) +
               \frac{1}{2} \sum_{n=1}^{L_x} \sum_{m=1}^{L_y-1} \left( \vec{c}^\dagger_{n,m}\left[ 
                -t\tau_z\sigma_0 -
                i\frac{\Delta}{2} \tau_x\sigma_y \right] \vec{c}_{n,m+1} + H.c. \right) 
           
            H_{S} = \frac{1}{2}\sum_m^{L_y} \mathbf{c}^\dagger_{L_x,m}\left[ 
                        -\mu\tau_z\sigma_0 + \Delta_0\tau_x\sigma_0 \right] \mathbf{c}_{L_x,m}\nonumber
            				+ \frac{1}{2}
                        \sum_m^{L_y-1}\left[\mathbf{c}^\dagger_{L_x,m}\left(-t\tau_z\sigma_0 \right)\mathbf{c}_{L_x,m+1}
                        + H.c.\right]
            
           H_J = t_J/2\sum_m^{L_y}[\vec{c}_{L_x-1,m}^\dagger(cos(\phi/2)\tau_z\sigma_0+isin(\phi/2)\tau_0\sigma_0)\vec{c}_{L_x,m}+H.c.]
    """
    phi = Phi[::-1]  # I have inverted the y axis
    M = scipy.sparse.lil_matrix((4*(L_x)*L_y, 4*(L_x)*L_y), dtype=complex)
    onsite_A1u = -mu/4 * np.kron(tau_z, sigma_0)   # para no duplicar al sumar la traspuesta
    for i in range(1, L_x):
      for j in range(1, L_y+1):
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j, beta, L_x, L_y)] = onsite_A1u[alpha, beta]   
    onsite_S = 1/4*( -mu * np.kron(tau_z, sigma_0) + Delta*np.kron(tau_x, sigma_0) )
    for j in range(1, L_y+1):
      for alpha in range(4):
        for beta in range(4):
          M[index(L_x, j, alpha, L_x, L_y), index(L_x, j, beta, L_x, L_y)] = onsite_S[alpha, beta]
    hopping_x_A1u = 1/2*(-t * np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_x) )
    for i in range(1, L_x-1):
      for j in range(1, L_y+1):    
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i+1, j, beta, L_x, L_y)] = hopping_x_A1u[alpha, beta]
    hopping_y_A1u = 1/2*( -t * np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_y) )
    for i in range(1, L_x):
      for j in range(1, L_y+1): 
        for alpha in range(4):
          for beta in range(4):
            M[index_periodic(i, j, alpha, L_x, L_y), index_periodic(i, j+1, beta, L_x, L_y)] = hopping_y_A1u[alpha, beta]
    hopping_y_S = -t/2 * np.kron(tau_z, sigma_0)    
    for j in range(1, L_y+1): 
      for alpha in range(4):
        for beta in range(4):
          M[index_periodic(L_x, j, alpha, L_x, L_y), index_periodic(L_x, j+1, beta, L_x, L_y)] = hopping_y_S[alpha, beta]
    for j in range(1, L_y+1):
        hopping_junction_x = t_J/2 * (np.cos(phi[j-1]/2)*np.kron(tau_z, sigma_0)+1j*np.sin(phi[j-1]/2)*np.kron(tau_0, sigma_0))
        for alpha in range(4):
            for beta in range(4):
                M[index(L_x-1, j, alpha, L_x, L_y), index(L_x, j, beta, L_x, L_y)] = hopping_junction_x[alpha, beta]
    return scipy.sparse.csr_matrix(M + M.conj().T)

def Hamiltonian_A1u(t, mu, L_x, L_y, Delta, t_J, Phi):
    r"""Return the matrix for A1u model in a junction with a superconductor with:
    
       .. math ::
           \vec{c_{n,m}} = (c_{n,m,\uparrow},
                            c_{n,m,\downarrow},
                            c^\dagger_{n,m,\downarrow},
                            -c^\dagger_{n,m,\uparrow})^T
           
           H_{A1u} = \frac{1}{2} \sum_{n=1}^{L_x} \sum_{m=1}^{L_y} (-\mu \vec{c}^\dagger_{n,m} \tau_z\sigma_0  \vec{c}_{n,m}) +
               \frac{1}{2} \sum_{n=1}^{L_x-1} \sum_{m=1}^{L_y} \left( \vec{c}^\dagger_{n,m}\left[ 
                -t\tau_z\sigma_0 -
                i\frac{\Delta}{2} \tau_x\sigma_x \right] \vec{c}_{n+1,m} + H.c. \right) +
               \frac{1}{2} \sum_{n=1}^{L_x} \sum_{m=1}^{L_y-1} \left( \vec{c}^\dagger_{n,m}\left[ 
                -t\tau_z\sigma_0 -
                i\frac{\Delta}{2} \tau_x\sigma_y \right] \vec{c}_{n,m+1} + H.c. \right) 
         """
    M = scipy.sparse.lil_matrix((4*L_x*L_y, 4*L_x*L_y), dtype=complex)
    onsite_A1u = -mu/4 * np.kron(tau_z, sigma_0)   # para no duplicar al sumar la traspuesta
    for i in range(1, L_x+1):
      for j in range(1, L_y+1):
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j, beta, L_x, L_y)] = onsite_A1u[alpha, beta]   
    hopping_x_A1u = 1/2*(-t * np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_x) )
    for i in range(1, L_x):
      for j in range(1, L_y+1):    
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i+1, j, beta, L_x, L_y)] = hopping_x_A1u[alpha, beta]
    hopping_y_A1u = 1/2*(-t * np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_y) )
    for i in range(1, L_x+1):
      for j in range(1, L_y): 
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j+1, beta, L_x, L_y)] = hopping_y_A1u[alpha, beta]
    return scipy.sparse.csr_matrix(M + M.conj().T)

def Hamiltonian_A1u_junction(t, mu, L_x, L_y, Delta, t_J, Phi):
    r"""Return the matrix for A1u model in a junction with a superconductor with:
    
       .. math ::
           \vec{c_{n,m}} = (c_{n,m,\uparrow},
                            c_{n,m,\downarrow},
                            c^\dagger_{n,m,\downarrow},
                            -c^\dagger_{n,m,\uparrow})^T
           
           H = H_{A1u, 1} + H_{A1u, 2} + H_J
           
           H_{A1u} = \frac{1}{2} \sum_{n=1}^{L_x} \sum_{m=1}^{L_y} (-\mu \vec{c}^\dagger_{n,m} \tau_z\sigma_0  \vec{c}_{n,m}) +
               \frac{1}{2} \sum_{n=1}^{L_x-1} \sum_{m=1}^{L_y} \left( \vec{c}^\dagger_{n,m}\left[ 
                -t\tau_z\sigma_0 -
                i\frac{\Delta}{2} \tau_x\sigma_x \right] \vec{c}_{n+1,m} + H.c. \right) +
               \frac{1}{2} \sum_{n=1}^{L_x} \sum_{m=1}^{L_y-1} \left( \vec{c}^\dagger_{n,m}\left[ 
                -t\tau_z\sigma_0 -
                i\frac{\Delta}{2} \tau_x\sigma_y \right] \vec{c}_{n,m+1} + H.c. \right) 
         
           H_J = t_J/2\sum_m^{L_y}[\vec{c}_{L_x-1,m}^\dagger(cos(\phi/2)\tau_z\sigma_0+isin(\phi/2)\tau_0\sigma_0)\vec{c}_{L_x,m}+H.c.]
    """
    phi = Phi[::-1]  # I have inverted the y axis
    M = np.zeros((4*L_x*L_y, 4*L_x*L_y), dtype=complex)    
    onsite_A1u = -mu/4 * np.kron(tau_z, sigma_0)   # para no duplicar al sumar la traspuesta
    for i in range(1, L_x+1):
      for j in range(1, L_y+1):
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j, beta, L_x, L_y)] = onsite_A1u[alpha, beta]   
    hopping_x_A1u = 1/2*( -t * np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_x) )
    for i in range(1, L_x//2):
      for j in range(1, L_y+1):    
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i+1, j, beta, L_x, L_y)] = hopping_x_A1u[alpha, beta]
    for i in range(L_x//2+1, L_x):
      for j in range(1, L_y+1):    
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i+1, j, beta, L_x, L_y)] = hopping_x_A1u[alpha, beta]
    hopping_y_A1u = 1/2* ( -t/2 * np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_y) )
    for i in range(1, L_x+1):
      for j in range(1, L_y): 
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j+1, beta, L_x, L_y)] = hopping_y_A1u[alpha, beta]
    for j in range(1, L_y+1):
        hopping_junction_x = t_J/2 * (np.cos(phi[j-1]/2)*np.kron(tau_z, sigma_0)+1j*np.sin(phi[j-1]/2)*np.kron(tau_0, sigma_0))
        for alpha in range(4):
            for beta in range(4):
                M[index(L_x//2, j, alpha, L_x, L_y), index(L_x//2+1, j, beta, L_x, L_y)] = hopping_junction_x[alpha, beta]
    return M + M.conj().T

def Hamiltonian_A1u_junction_k(t, k, mu, L, Delta, phi, t_J):
    r"""Returns the H_k matrix for A1u-A1u junction with:

    .. math::
        H = \sum_k \frac{1}{2} (H_k^{S1} + H_k^{S2} + H_{J,k} )

        H_{A1u} = \frac{1}{2}\sum_k H_k
        
        H_{A1u,k} = \sum_n^L \vec{c}^\dagger_n\left[ 
            \xi_k\tau_z\sigma_0 +
            \Delta sin(k_y)\tau_x\sigma_y \right] +
            \sum_n^{L-1}\left(\vec{c}^\dagger_n(-t\tau_z\sigma_0 + \frac{\Delta}{2i}\tau_x\sigma_x)\vec{c}_{n+1}
            + H.c. \right)
                
        H_{J,k} = t_J\vec{c}_{S1,k,L}^{\dagger}\left( 
            \frac{\tau^z+\tau^0}{2} e^{i\phi/2} + \frac{\tau^z-\tau^0}{2} e^{-i\phi/2}
            \right)\vec{c}_{S2,k,1} + H.c.           
       
        \vec{c} = (c_{k,\uparrow}, c_{k,\downarrow},c^\dagger_{-k,\downarrow},-c^\dagger_{-k,\uparrow})^T
    """
    M = np.zeros((4*L, 4*L), dtype=complex)
    chi_k = -mu - 2*t * np.cos(k)
    onsite_A1u = 1/4*(chi_k * np.kron(tau_z, sigma_0) + \
            Delta*np.sin(k)* np.kron(tau_x, sigma_y) )  #divided by 2 because of the transpose
    for i in range(1, L+1):
        for alpha in range(4):
            for beta in range(4):
                M[index_k(i, alpha, L), index_k(i, beta, L)] = onsite_A1u[alpha, beta]
    hopping_A1u = 1/2*(-t*np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_x) )
    for i in range(1, L//2):
        for alpha in range(4):
            for beta in range(4):
                M[index_k(i, alpha, L), index_k(i+1, beta, L)] = hopping_A1u[alpha, beta]
    for i in range((L//2+1), L):
        for alpha in range(4):
            for beta in range(4):
                M[index_k(i, alpha, L), index_k(i+1, beta, L)] = hopping_A1u[alpha, beta]
    hopping_junction_x = t_J/2 * (np.cos(phi/2)*np.kron(tau_z, sigma_0)+1j*np.sin(phi/2)*np.kron(tau_0, sigma_0))
    for alpha in range(4):
        for beta in range(4):
            M[index_k(L//2, alpha, L), index_k(L//2+1, beta, L)] = hopping_junction_x[alpha, beta]                        
    return (M + M.conj().T)

def Hamiltonian_A1u_S_junction_k(t, k, mu, L_A1u, L_S, Delta_A1u, Delta_S, phi, t_J):
    r"""Returns the H_k matrix for junction bewtween A1u and S superconductors with:

    .. math::
        H = \sum_k \frac{1}{2} (H_k^{S1} + H_k^{S2} + H_{J,k} )

        H_{A1u} = \frac{1}{2}\sum_k H_{A1u,k}
        
        H_{A1u,k} = \sum_n^L \vec{c}^\dagger_n\left[ 
            \xi_k\tau_z\sigma_0 +
            \Delta sin(k_y)\tau_x\sigma_y \right] +
            \sum_n^{L-1}\left(\vec{c}^\dagger_n(-t\tau_z\sigma_0 + \frac{\Delta}{2i}\tau_x\sigma_x)\vec{c}_{n+1}
            + H.c. \right)
                
        H_{J,k} = t_J\vec{c}_{S1,k,L}^{\dagger}\left( 
            \frac{\tau^z+\tau^0}{2} e^{i\phi/2} + \frac{\tau^z-\tau^0}{2} e^{-i\phi/2}
            \right)\vec{c}_{S2,k,1} + H.c.            
        
        H_S = \frac{1}{2} \sum_k H_{S,k}
        
        H_{S,k} = \sum_n^L \vec{c}^\dagger_n\left[ 
            \xi_k\tau_z\sigma_0 +
            \Delta_0 \tau_x\sigma_0 \right] + \left(
            \sum_n^{L-1}\vec{c}^\dagger_n(-t\tau_z\sigma_0)\vec{c}_{n+1}
            + H.c. \right)
        
        \vec{c} = (c_{k,\uparrow}, c_{k,\downarrow},c^\dagger_{-k,\downarrow},-c^\dagger_{-k,\uparrow})^T
    
        \xi_k = -2tcos(k) - \mu
    """
    L = L_A1u + L_S
    M = np.zeros((4*L, 4*L), dtype=complex)
    chi_k = -mu - 2*t * np.cos(k)
    onsite_A1u = 1/4*(chi_k * np.kron(tau_z, sigma_0) + \
            Delta_A1u *np.sin(k)* np.kron(tau_x, sigma_y) )
    for i in range(1, L_A1u+1):
        for alpha in range(4):
            for beta in range(4):
                M[index_k(i, alpha, L), index_k(i, beta, L)] = onsite_A1u[alpha, beta]
    onsite_S = 1/4*(chi_k * np.kron(tau_z, sigma_0) + Delta_S*np.kron(tau_x, sigma_0) ) # I divide by 1/2 because of the transpose
    for i in range(L_A1u+1, L+1):
        for alpha in range(4):
            for beta in range(4):
                M[index_k(i, alpha, L), index_k(i, beta, L)] = onsite_S[alpha, beta]
    hopping_A1u = 1/2*(-t*np.kron(tau_z, sigma_0) - 1j*Delta_A1u/2 * np.kron(tau_x, sigma_x) )
    for i in range(1, L_A1u):
        for alpha in range(4):
            for beta in range(4):
                M[index_k(i, alpha, L), index_k(i+1, beta, L)] = hopping_A1u[alpha, beta]
    hopping_S = 1/2*(-t*np.kron(tau_z, sigma_0) )
    for i in range(L_A1u+1, L):
        for alpha in range(4):
            for beta in range(4):
                M[index_k(i, alpha, L), index_k(i+1, beta, L)] = hopping_S[alpha, beta]
    hopping_junction_x = t_J/2 * (np.cos(phi/2)*np.kron(tau_z, sigma_0)+1j*np.sin(phi/2)*np.kron(tau_0, sigma_0))
    for alpha in range(4):
        for beta in range(4):
            M[index_k(L_A1u, alpha, L), index_k(L_A1u+1, beta, L)] = hopping_junction_x[alpha, beta]                        
    return M + M.conj().T

def Hamiltonian_S(t, k, mu, L_x, Delta):
    r"""Returns the H_k matrix for an S trivial superconductor with:

    .. math::
        H_S = \frac{1}{2} \sum_k H_{S,k}
        
        H_{S,k} = \sum_n^L \vec{c}^\dagger_n\left[ 
            \xi_k\tau_z\sigma_0 +
            \Delta_0 \tau_x\sigma_0 \right] + \left(
            \sum_n^{L-1}\vec{c}^\dagger_n(-t\tau_z\sigma_0)\vec{c}_{n+1}
            + H.c. \right)
        
        \vec{c} = (c_{k,\uparrow}, c_{k,\downarrow},c^\dagger_{-k,\downarrow},-c^\dagger_{-k,\uparrow})^T
    
        \xi_k = -2tcos(k) - \mu
    """
    M = np.zeros((4*L_x, 4*L_x), dtype=complex)
    chi_k = -mu - 2*t * np.cos(k)
    onsite_S = 1/4*(chi_k * np.kron(tau_z, sigma_0) + \
            Delta*np.kron(tau_x, sigma_0) )  #divided by 2 because of the transpose
    for i in range(1, L_x+1):
        for alpha in range(4):
            for beta in range(4):
                M[index_k(i, alpha, L_x), index_k(i, beta, L_x)] = onsite_S[alpha, beta]
    hopping_S = 1/2*(-t*np.kron(tau_z, sigma_0) )
    for i in range(1, L_x):
        for alpha in range(4):
            for beta in range(4):
                M[index_k(i, alpha, L_x), index_k(i+1, beta, L_x)] = hopping_S[alpha, beta]
    return (M + M.conj().T)

def Hamiltonian_S_S_junction_k(t, k, mu, L_S_1, L_S_2, Delta_S_1, Delta_S_2, phi, t_J):
    r"""Returns the H_k matrix for junction bewtween A1u and S superconductors with:

    .. math::
        H = \sum_k \frac{1}{2} (H_k^{S1} + H_k^{S2} + H_{J,k} )
               
        H_{J,k} = t_J\vec{c}_{S1,k,L}^{\dagger}\left( 
            \frac{\tau^z+\tau^0}{2} e^{i\phi/2} + \frac{\tau^z-\tau^0}{2} e^{-i\phi/2}
            \right)\vec{c}_{S2,k,1} + H.c.            
        
        H_S = \frac{1}{2} \sum_k H_{S,k}
        
        H_{S,k} = \sum_n^L \vec{c}^\dagger_n\left[ 
            \xi_k\tau_z\sigma_0 +
            \Delta_0 \tau_x\sigma_0 \right] + \left(
            \sum_n^{L-1}\vec{c}^\dagger_n(-t\tau_z\sigma_0)\vec{c}_{n+1}
            + H.c. \right)
        
        \vec{c} = (c_{k,\uparrow}, c_{k,\downarrow},c^\dagger_{-k,\downarrow},-c^\dagger_{-k,\uparrow})^T
    
        \xi_k = -2tcos(k) - \mu
    """
    L = L_S_1 + L_S_2
    M = np.zeros((4*L, 4*L), dtype=complex)
    chi_k = -mu - 2*t * np.cos(k)
    onsite_S_1 = 1/4*(chi_k * np.kron(tau_z, sigma_0) + Delta_S_1*np.kron(tau_x, sigma_0) )
    for i in range(1, L_S_1+1):
        for alpha in range(4):
            for beta in range(4):
                M[index_k(i, alpha, L), index_k(i, beta, L)] = onsite_S_1[alpha, beta]
    onsite_S_2 = 1/4*(chi_k * np.kron(tau_z, sigma_0) + Delta_S_2*np.kron(tau_x, sigma_0) ) # I divide by 1/2 because of the transpose
    for i in range(L_S_1+1, L+1):
        for alpha in range(4):
            for beta in range(4):
                M[index_k(i, alpha, L), index_k(i, beta, L)] = onsite_S_2[alpha, beta]
    hopping_S = 1/2*(-t*np.kron(tau_z, sigma_0) )
    for i in range(1, L_S_1):
        for alpha in range(4):
            for beta in range(4):
                M[index_k(i, alpha, L), index_k(i+1, beta, L)] = hopping_S[alpha, beta]
    for i in range(L_S_1+1, L):
        for alpha in range(4):
            for beta in range(4):
                M[index_k(i, alpha, L), index_k(i+1, beta, L)] = hopping_S[alpha, beta]
    hopping_junction_x = t_J/2 * (np.cos(phi/2)*np.kron(tau_z, sigma_0)+1j*np.sin(phi/2)*np.kron(tau_0, sigma_0))
    for alpha in range(4):
        for beta in range(4):
            M[index_k(L_S_1, alpha, L), index_k(L_S_1+1, beta, L)] = hopping_junction_x[alpha, beta]                        
    return M + M.conj().T

def Hamiltonian_A1us_S_junction_k(t, k, mu, L_A1u, L_S, Delta_A1u, Delta_S, phi, t_J):
    r"""Returns the H_k matrix for junction bewtween A1u (with s-wave superconductivity) and S superconductors with:
    
    .. math::
        H = \sum_k \frac{1}{2} (H_k^{S1} + H_k^{S2} + H_{J,k} )
        H_{A1us} = \frac{1}{2}\sum_k H_{A1us,k}
        
        H_{A1us,k} = \sum_n^L \vec{c}^\dagger_n\left[ 
            \xi_k\tau_z\sigma_0 +
            \Delta sin(k_y)\tau_x\sigma_y + \Delta_0 \tau_x\sigma_0\right] +
            \sum_n^{L-1}\left(\vec{c}^\dagger_n(-t\tau_z\sigma_0 + \frac{\Delta}{2i}\tau_x\sigma_x)\vec{c}_{n+1}
            + H.c. \right)
                
        H_{J,k} = t_J\vec{c}_{S1,k,L}^{\dagger}\left( 
            \frac{\tau^z+\tau^0}{2} e^{i\phi/2} + \frac{\tau^z-\tau^0}{2} e^{-i\phi/2}
            \right)\vec{c}_{S2,k,1} + H.c.            
        
        H_S = \frac{1}{2} \sum_k H_{S,k}
        
        H_{S,k} = \sum_n^L \vec{c}^\dagger_n\left[ 
            \xi_k\tau_z\sigma_0 +
            \Delta_0 \tau_x\sigma_0 \right] + \left(
            \sum_n^{L-1}\vec{c}^\dagger_n(-t\tau_z\sigma_0)\vec{c}_{n+1}
            + H.c. \right)
        
        \vec{c} = (c_{k,\uparrow}, c_{k,\downarrow},c^\dagger_{-k,\downarrow},-c^\dagger_{-k,\uparrow})^T
    
        \xi_k = -2tcos(k) - \mu
    """
    L = L_A1u + L_S
    M = np.zeros((4*L, 4*L), dtype=complex)
    chi_k = -mu - 2*t * np.cos(k)
    onsite_A1u = 1/4*(chi_k * np.kron(tau_z, sigma_0) + \
            Delta_A1u*np.sin(k) * np.kron(tau_x, sigma_y) +
            Delta_S * np.kron(tau_x, sigma_0))
    for i in range(1, L_A1u+1):
        for alpha in range(4):
            for beta in range(4):
                M[index_k(i, alpha, L), index_k(i, beta, L)] = onsite_A1u[alpha, beta]
    onsite_S = 1/4*(chi_k * np.kron(tau_z, sigma_0) + Delta_S*np.kron(tau_x, sigma_0) ) # I divide by 1/2 because of the transpose
    for i in range(L_A1u+1, L+1):
        for alpha in range(4):
            for beta in range(4):
                M[index_k(i, alpha, L), index_k(i, beta, L)] = onsite_S[alpha, beta]
    hopping_A1u = 1/2*(-t*np.kron(tau_z, sigma_0) - 1j*Delta_A1u/2 * np.kron(tau_x, sigma_x) )
    for i in range(1, L_A1u):
        for alpha in range(4):
            for beta in range(4):
                M[index_k(i, alpha, L), index_k(i+1, beta, L)] = hopping_A1u[alpha, beta]
    hopping_S = 1/2*(-t*np.kron(tau_z, sigma_0) )
    for i in range(L_A1u+1, L):
        for alpha in range(4):
            for beta in range(4):
                M[index_k(i, alpha, L), index_k(i+1, beta, L)] = hopping_S[alpha, beta]
    hopping_junction_x = t_J/2 * (np.cos(phi/2)*np.kron(tau_z, sigma_0)+1j*np.sin(phi/2)*np.kron(tau_0, sigma_0))
    for alpha in range(4):
        for beta in range(4):
            M[index_k(L_A1u, alpha, L), index_k(L_A1u+1, beta, L)] = hopping_junction_x[alpha, beta]                        
    return M + M.conj().T

def Hamiltonian_ZKM_S_junction_k(t, k, mu, L_ZKM, L_S, Delta_0, Delta_1, phi, t_J, Lambda):
    r"""Returns the H_k matrix for junction between A1u and S superconductors with:
        
    .. math::
        H = \sum_k \frac{1}{2} (H_k^{S1} + H_k^{S2} + H_{J,k} )
        
        H_{ZKM} = \frac{1}{2}\sum_k H_{A1u,k}
        
        H_{ZKM,k} = \sum_n^L \vec{c}^\dagger_n\left[ 
            \xi_k\tau_z\sigma_0 -2\lambda sin(k_y)\tau_z\sigma_x +
            (\Delta_0 + 2\Delta_1 cos(k) )\tau_x\sigma_0 \right] +
            \sum_n^{L-1}\left(\vec{c}^\dagger_n(-t\tau_z\sigma_0 -i\lambda \tau_z\sigma_z +\Delta_1\tau_x\sigma_0)\vec{c}_{n+1}
            + H.c. \right)
                
        H_{J,k} = t_J\vec{c}_{S1,k,L}^{\dagger}\left( 
            \frac{\tau^z+\tau^0}{2} e^{i\phi/2} + \frac{\tau^z-\tau^0}{2} e^{-i\phi/2}
            \right)\vec{c}_{S2,k,1} + H.c.            
        
        H_S = \frac{1}{2} \sum_k H_{S,k}
        
        H_{S,k} = \sum_n^L \vec{c}^\dagger_n\left[ 
            \xi_k\tau_z\sigma_0 +
            \Delta_0 \tau_x\sigma_0 \right] + \left(
            \sum_n^{L-1}\vec{c}^\dagger_n(-t\tau_z\sigma_0)\vec{c}_{n+1}
            + H.c. \right)
        
        \vec{c} = (c_{k,\uparrow}, c_{k,\downarrow},c^\dagger_{-k,\downarrow},-c^\dagger_{-k,\uparrow})^T
    
        \xi_k = -2tcos(k) - \mu
    """
    L = L_ZKM + L_S
    M = np.zeros((4*L, 4*L), dtype=complex)
    chi_k = -mu - 2*t * np.cos(k)
    onsite_ZKM = 1/4*(chi_k * np.kron(tau_z, sigma_0) -\
            2*Lambda*np.sin(k)* np.kron(tau_z, sigma_x) +\
                (Delta_0+2*Delta_1*np.cos(k))*np.kron(tau_x, sigma_0) )
    for i in range(1, L_ZKM+1):
        for alpha in range(4):
            for beta in range(4):
                M[index_k(i, alpha, L), index_k(i, beta, L)] = onsite_ZKM[alpha, beta]
    onsite_S = 1/4*(chi_k * np.kron(tau_z, sigma_0) + Delta_0*np.kron(tau_x, sigma_0) ) # I divide by 1/2 because of the transpose
    for i in range(L_ZKM+1, L+1):
        for alpha in range(4):
            for beta in range(4):
                M[index_k(i, alpha, L), index_k(i, beta, L)] = onsite_S[alpha, beta]
    hopping_ZKM = 1/2*(-t*np.kron(tau_z, sigma_0) - 1j*Lambda * np.kron(tau_z, sigma_z) +\
                       Delta_1*np.kron(tau_x, sigma_0))
    for i in range(1, L_ZKM):
        for alpha in range(4):
            for beta in range(4):
                M[index_k(i, alpha, L), index_k(i+1, beta, L)] = hopping_ZKM[alpha, beta]
    hopping_S = 1/2*(-t*np.kron(tau_z, sigma_0) )
    for i in range(L_ZKM+1, L):
        for alpha in range(4):
            for beta in range(4):
                M[index_k(i, alpha, L), index_k(i+1, beta, L)] = hopping_S[alpha, beta]
    hopping_junction_x = t_J/2 * (np.cos(phi/2)*np.kron(tau_z, sigma_0)+1j*np.sin(phi/2)*np.kron(tau_0, sigma_0))
    for alpha in range(4):
        for beta in range(4):
            M[index_k(L_ZKM, alpha, L), index_k(L_ZKM+1, beta, L)] = hopping_junction_x[alpha, beta]                        
    return M + M.conj().T

def Hamiltonian_A1us_S_sparse(t, mu, L_x, L_y, Delta, Delta_0, t_J, Phi):
   r"""Return the matrix for A1us model with s and p wave pairing in a junction with an s-wave superconductor:

   .. math ::
      \vec{c_{n,m}} = (c_{n,m,\uparrow},
                       c_{n,m,\downarrow},
                       c^\dagger_{n,m,\downarrow},
                       -c^\dagger_{n,m,\uparrow})^T
             
          H = H_{A1us} + H_{S} + H_J
          
          H_{A1us} = \frac{1}{2} \sum_{n=1}^{L_x} \sum_{m=1}^{L_y} \vec{c}^\dagger_{n,m}(-\mu  \tau_z\sigma_0 + \Delta_0 \tau_x\sigma_0 )\vec{c}_{n,m} +
              \frac{1}{2} \sum_{n=1}^{L_x-1} \sum_{m=1}^{L_y} \left( \vec{c}^\dagger_{n,m}\left[ 
               -t\tau_z\sigma_0 -
               i\frac{\Delta}{2} \tau_x\sigma_x \right] \vec{c}_{n+1,m} + H.c. \right) +
              \frac{1}{2} \sum_{n=1}^{L_x} \sum_{m=1}^{L_y-1} \left( \vec{c}^\dagger_{n,m}\left[ 
               -t\tau_z\sigma_0 -
               i\frac{\Delta}{2} \tau_x\sigma_y \right] \vec{c}_{n,m+1} + H.c. \right) 
          
           H_{S} = \frac{1}{2}\sum_m^{L_y} \mathbf{c}^\dagger_{L_x,m}\left[ 
                       -\mu\tau_z\sigma_0 + \Delta_0\tau_x\sigma_0 \right] \mathbf{c}_{L_x,m}\nonumber
           				+ \frac{1}{2}
                       \sum_m^{L_y-1}\left[\mathbf{c}^\dagger_{L_x,m}\left(-t\tau_z\sigma_0 \right)\mathbf{c}_{L_x,m+1}
                       + H.c.\right]
           
          H_J = t_J/2\sum_m^{L_y}[\vec{c}_{L_x-1,m}^\dagger(cos(\phi/2)\tau_z\sigma_0+isin(\phi/2)\tau_0\sigma_0)\vec{c}_{L_x,m}+H.c.]
   """
   phi = Phi[::-1]  # I have inverted the y axis
   M = scipy.sparse.lil_matrix((4*(L_x)*L_y, 4*(L_x)*L_y), dtype=complex)
   onsite_A1us = 1/4*( -mu * np.kron(tau_z, sigma_0) + Delta_0*np.kron(tau_x, sigma_0) )  # para no duplicar al sumar la traspuesta
   for i in range(1, L_x):
     for j in range(1, L_y+1):
       for alpha in range(4):
         for beta in range(4):
           M[index(i, j, alpha, L_x, L_y), index(i, j, beta, L_x, L_y)] = onsite_A1us[alpha, beta]   
   onsite_S = 1/4*( -mu * np.kron(tau_z, sigma_0) + Delta_0*np.kron(tau_x, sigma_0) )
   for j in range(1, L_y+1):
     for alpha in range(4):
       for beta in range(4):
         M[index(L_x, j, alpha, L_x, L_y), index(L_x, j, beta, L_x, L_y)] = onsite_S[alpha, beta]
   hopping_x_A1us = 1/2*(-t * np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_x) )
   for i in range(1, L_x-1):
     for j in range(1, L_y+1):    
       for alpha in range(4):
         for beta in range(4):
           M[index(i, j, alpha, L_x, L_y), index(i+1, j, beta, L_x, L_y)] = hopping_x_A1us[alpha, beta]
   hopping_y_A1u = 1/2*( -t * np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_y) )
   for i in range(1, L_x):
     for j in range(1, L_y): 
       for alpha in range(4):
         for beta in range(4):
           M[index(i, j, alpha, L_x, L_y), index(i, j+1, beta, L_x, L_y)] = hopping_y_A1u[alpha, beta]
   hopping_y_S = -t/2 * np.kron(tau_z, sigma_0)    
   for j in range(1, L_y): 
     for alpha in range(4):
       for beta in range(4):
         M[index(L_x, j, alpha, L_x, L_y), index(L_x, j+1, beta, L_x, L_y)] = hopping_y_S[alpha, beta]
   for j in range(1, L_y+1):
       hopping_junction_x = t_J/2 * (np.cos(phi[j-1]/2)*np.kron(tau_z, sigma_0)+1j*np.sin(phi[j-1]/2)*np.kron(tau_0, sigma_0))
       for alpha in range(4):
           for beta in range(4):
               M[index(L_x-1, j, alpha, L_x, L_y), index(L_x, j, beta, L_x, L_y)] = hopping_junction_x[alpha, beta]
   return scipy.sparse.csr_matrix(M + M.conj().T)

def Hamiltonian_A1us_junction_sparse(t, mu, L_x, L_y, Delta, Delta_0, t_J, Phi):
    r"""Return the matrix for A1u model in a junction with a superconductor with:
    
       .. math ::
           \vec{c_{n,m}} = (c_{n,m,\uparrow},
                            c_{n,m,\downarrow},
                            c^\dagger_{n,m,\downarrow},
                            -c^\dagger_{n,m,\uparrow})^T
           
           H = H_{A1us, 1} + H_{A1us, 2} + H_J
           
           H_{A1us} = \frac{1}{2} \sum_{n=1}^{L_x} \sum_{m=1}^{L_y} \vec{c}^\dagger_{n,m}(-\mu  \tau_z\sigma_0 + \Delta_0 \tau_x\sigma_0 )\vec{c}_{n,m} +
               \frac{1}{2} \sum_{n=1}^{L_x-1} \sum_{m=1}^{L_y} \left( \vec{c}^\dagger_{n,m}\left[ 
                -t\tau_z\sigma_0 -
                i\frac{\Delta}{2} \tau_x\sigma_x \right] \vec{c}_{n+1,m} + H.c. \right) +
               \frac{1}{2} \sum_{n=1}^{L_x} \sum_{m=1}^{L_y-1} \left( \vec{c}^\dagger_{n,m}\left[ 
                -t\tau_z\sigma_0 -
                i\frac{\Delta}{2} \tau_x\sigma_y \right] \vec{c}_{n,m+1} + H.c. \right) 
         
           H_J = t_J/2\sum_m^{L_y}[\vec{c}_{L_x-1,m}^\dagger(cos(\phi/2)\tau_z\sigma_0+isin(\phi/2)\tau_0\sigma_0)\vec{c}_{L_x,m}+H.c.]
    """
    phi = Phi[::-1]  # I have inverted the y axis
    M = scipy.sparse.lil_matrix((4*L_x*L_y, 4*L_x*L_y), dtype=complex)
    onsite_A1us = 1/4*(-mu * np.kron(tau_z, sigma_0) + Delta_0*np.kron(tau_x, sigma_0))   # para no duplicar al sumar la traspuesta
    for i in range(1, L_x+1):
      for j in range(1, L_y+1):
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j, beta, L_x, L_y)] = onsite_A1us[alpha, beta]   
    hopping_x_A1us = 1/2*( -t * np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_x) )
    for i in range(1, L_x//2):
      for j in range(1, L_y+1):    
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i+1, j, beta, L_x, L_y)] = hopping_x_A1us[alpha, beta]
    for i in range(L_x//2+1, L_x):
      for j in range(1, L_y+1):    
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i+1, j, beta, L_x, L_y)] = hopping_x_A1us[alpha, beta]
    hopping_y_A1us = 1/2* ( -t/2 * np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_y) )
    for i in range(1, L_x+1):
      for j in range(1, L_y): 
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j+1, beta, L_x, L_y)] = hopping_y_A1us[alpha, beta]
    for j in range(1, L_y+1):
        hopping_junction_x = t_J/2 * (np.cos(phi[j-1]/2)*np.kron(tau_z, sigma_0)+1j*np.sin(phi[j-1]/2)*np.kron(tau_0, sigma_0))
        for alpha in range(4):
            for beta in range(4):
                M[index(L_x//2, j, alpha, L_x, L_y), index(L_x//2+1, j, beta, L_x, L_y)] = hopping_junction_x[alpha, beta]
    return scipy.sparse.csr_matrix(M + M.conj().T)

def Hamiltonian_A1us_junction_sparse_periodic(t, mu, L_x, L_y, Delta, Delta_0, t_J, Phi):
    r"""Return the matrix for A1u model in a junction with a superconductor with:
    
       .. math ::
           \vec{c_{n,m}} = (c_{n,m,\uparrow},
                            c_{n,m,\downarrow},
                            c^\dagger_{n,m,\downarrow},
                            -c^\dagger_{n,m,\uparrow})^T
           
           H = H_{A1us, 1} + H_{A1us, 2} + H_J
           
           H_{A1us} = \frac{1}{2} \sum_{n=1}^{L_x} \sum_{m=1}^{L_y} \vec{c}^\dagger_{n,m}(-\mu  \tau_z\sigma_0 + \Delta_0 \tau_x\sigma_0 )\vec{c}_{n,m} +
               \frac{1}{2} \sum_{n=1}^{L_x-1} \sum_{m=1}^{L_y} \left( \vec{c}^\dagger_{n,m}\left[ 
                -t\tau_z\sigma_0 -
                i\frac{\Delta}{2} \tau_x\sigma_x \right] \vec{c}_{n+1,m} + H.c. \right) +
               \frac{1}{2} \sum_{n=1}^{L_x} \sum_{m=1}^{L_y-1} \left( \vec{c}^\dagger_{n,m}\left[ 
                -t\tau_z\sigma_0 -
                i\frac{\Delta}{2} \tau_x\sigma_y \right] \vec{c}_{n,m+1} + H.c. \right) 
         
           H_J = t_J/2\sum_m^{L_y}[\vec{c}_{L_x-1,m}^\dagger(cos(\phi/2)\tau_z\sigma_0+isin(\phi/2)\tau_0\sigma_0)\vec{c}_{L_x,m}+H.c.]
    """
    phi = Phi[::-1]  # I have inverted the y axis
    M = scipy.sparse.lil_matrix((4*L_x*L_y, 4*L_x*L_y), dtype=complex)
    onsite_A1us = 1/4*(-mu * np.kron(tau_z, sigma_0) + Delta_0*np.kron(tau_x, sigma_0))   # para no duplicar al sumar la traspuesta
    for i in range(1, L_x+1):
      for j in range(1, L_y+1):
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j, beta, L_x, L_y)] = onsite_A1us[alpha, beta]   
    hopping_x_A1us = 1/2*( -t * np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_x) )
    for i in range(1, L_x//2):
      for j in range(1, L_y+1):    
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i+1, j, beta, L_x, L_y)] = hopping_x_A1us[alpha, beta]
    for i in range(L_x//2+1, L_x):
      for j in range(1, L_y+1):    
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i+1, j, beta, L_x, L_y)] = hopping_x_A1us[alpha, beta]
    hopping_y_A1us = 1/2* ( -t/2 * np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_y) )
    for i in range(1, L_x+1):
      for j in range(1, L_y+1): 
        for alpha in range(4):
          for beta in range(4):
            M[index_periodic(i, j, alpha, L_x, L_y), index_periodic(i, j+1, beta, L_x, L_y)] = hopping_y_A1us[alpha, beta]
    for j in range(1, L_y+1):
        hopping_junction_x = t_J/2 * (np.cos(phi[j-1]/2)*np.kron(tau_z, sigma_0)+1j*np.sin(phi[j-1]/2)*np.kron(tau_0, sigma_0))
        for alpha in range(4):
            for beta in range(4):
                M[index(L_x//2, j, alpha, L_x, L_y), index(L_x//2+1, j, beta, L_x, L_y)] = hopping_junction_x[alpha, beta]
    return scipy.sparse.csr_matrix(M + M.conj().T)

def Hamiltonian_A1us_junction_sparse_periodic_in_x_and_y(t, mu, L_x, L_y, Delta, Delta_0, t_J, Phi):
    r"""Return the matrix for A1u model in a junction with a superconductor with:
    
       .. math ::
           \vec{c_{n,m}} = (c_{n,m,\uparrow},
                            c_{n,m,\downarrow},
                            c^\dagger_{n,m,\downarrow},
                            -c^\dagger_{n,m,\uparrow})^T
           
           H = H_{A1us, 1} + H_{A1us, 2} + H_J
           
           H_{A1us} = \frac{1}{2} \sum_{n=1}^{L_x} \sum_{m=1}^{L_y} \vec{c}^\dagger_{n,m}(-\mu  \tau_z\sigma_0 + \Delta_0 \tau_x\sigma_0 )\vec{c}_{n,m} +
               \frac{1}{2} \sum_{n=1}^{L_x-1} \sum_{m=1}^{L_y} \left( \vec{c}^\dagger_{n,m}\left[ 
                -t\tau_z\sigma_0 -
                i\frac{\Delta}{2} \tau_x\sigma_x \right] \vec{c}_{n+1,m} + H.c. \right) +
               \frac{1}{2} \sum_{n=1}^{L_x} \sum_{m=1}^{L_y-1} \left( \vec{c}^\dagger_{n,m}\left[ 
                -t\tau_z\sigma_0 -
                i\frac{\Delta}{2} \tau_x\sigma_y \right] \vec{c}_{n,m+1} + H.c. \right) 
         
           H_J = t_J/2\sum_m^{L_y}[\vec{c}_{L_x-1,m}^\dagger(cos(\phi/2)\tau_z\sigma_0+isin(\phi/2)\tau_0\sigma_0)\vec{c}_{L_x,m}+H.c.]
    """
    phi = Phi[::-1]  # I have inverted the y axis
    M = scipy.sparse.lil_matrix((4*L_x*L_y, 4*L_x*L_y), dtype=complex)
    onsite_A1us = 1/4*(-mu * np.kron(tau_z, sigma_0) + Delta_0*np.kron(tau_x, sigma_0))   # para no duplicar al sumar la traspuesta
    for i in range(1, L_x+1):
      for j in range(1, L_y+1):
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j, beta, L_x, L_y)] = onsite_A1us[alpha, beta]   
    hopping_x_A1us = 1/2*( -t * np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_x) )
    for i in range(1, L_x//2):
      for j in range(1, L_y+1):    
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i+1, j, beta, L_x, L_y)] = hopping_x_A1us[alpha, beta]
    for i in range(L_x//2+1, L_x+1):
      for j in range(1, L_y+1):    
        for alpha in range(4):
          for beta in range(4):
            M[index_periodic_in_x(i, j, alpha, L_x, L_y), index_periodic_in_x(i+1, j, beta, L_x, L_y)] = hopping_x_A1us[alpha, beta]
    hopping_y_A1us = 1/2* ( -t/2 * np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_y) )
    for i in range(1, L_x+1):
      for j in range(1, L_y+1): 
        for alpha in range(4):
          for beta in range(4):
            M[index_periodic(i, j, alpha, L_x, L_y), index_periodic(i, j+1, beta, L_x, L_y)] = hopping_y_A1us[alpha, beta]
    for j in range(1, L_y+1):
        hopping_junction_x = t_J/2 * (np.cos(phi[j-1]/2)*np.kron(tau_z, sigma_0)+1j*np.sin(phi[j-1]/2)*np.kron(tau_0, sigma_0))
        for alpha in range(4):
            for beta in range(4):
                M[index(L_x//2, j, alpha, L_x, L_y), index(L_x//2+1, j, beta, L_x, L_y)] = hopping_junction_x[alpha, beta]
    return scipy.sparse.csr_matrix(M + M.conj().T)

def Hamiltonian_A1us_S_sparse_periodic(t, mu, L_x, L_y, Delta, Delta_0, t_J, Phi):
   r"""Return the matrix for A1us model with s and p wave pairing in a junction with an s-wave superconductor:

   .. math ::
      \vec{c_{n,m}} = (c_{n,m,\uparrow},
                       c_{n,m,\downarrow},
                       c^\dagger_{n,m,\downarrow},
                       -c^\dagger_{n,m,\uparrow})^T
             
          H = H_{A1us} + H_{S} + H_J
          
          H_{A1us} = \frac{1}{2} \sum_{n=1}^{L_x} \sum_{m=1}^{L_y} \vec{c}^\dagger_{n,m}(-\mu  \tau_z\sigma_0 + \Delta_0 \tau_x\sigma_0 )\vec{c}_{n,m} +
              \frac{1}{2} \sum_{n=1}^{L_x-1} \sum_{m=1}^{L_y} \left( \vec{c}^\dagger_{n,m}\left[ 
               -t\tau_z\sigma_0 -
               i\frac{\Delta}{2} \tau_x\sigma_x \right] \vec{c}_{n+1,m} + H.c. \right) +
              \frac{1}{2} \sum_{n=1}^{L_x} \sum_{m=1}^{L_y-1} \left( \vec{c}^\dagger_{n,m}\left[ 
               -t\tau_z\sigma_0 -
               i\frac{\Delta}{2} \tau_x\sigma_y \right] \vec{c}_{n,m+1} + H.c. \right) 
          
           H_{S} = \frac{1}{2}\sum_m^{L_y} \mathbf{c}^\dagger_{L_x,m}\left[ 
                       -\mu\tau_z\sigma_0 + \Delta_0\tau_x\sigma_0 \right] \mathbf{c}_{L_x,m}\nonumber
           				+ \frac{1}{2}
                       \sum_m^{L_y-1}\left[\mathbf{c}^\dagger_{L_x,m}\left(-t\tau_z\sigma_0 \right)\mathbf{c}_{L_x,m+1}
                       + H.c.\right]
           
          H_J = t_J/2\sum_m^{L_y}[\vec{c}_{L_x-1,m}^\dagger(cos(\phi/2)\tau_z\sigma_0+isin(\phi/2)\tau_0\sigma_0)\vec{c}_{L_x,m}+H.c.]
   """
   phi = Phi[::-1]  # I have inverted the y axis
   M = scipy.sparse.lil_matrix((4*(L_x)*L_y, 4*(L_x)*L_y), dtype=complex)
   onsite_A1us = 1/4*( -mu * np.kron(tau_z, sigma_0) + Delta_0*np.kron(tau_x, sigma_0) )  # para no duplicar al sumar la traspuesta
   for i in range(1, L_x):
     for j in range(1, L_y+1):
       for alpha in range(4):
         for beta in range(4):
           M[index(i, j, alpha, L_x, L_y), index(i, j, beta, L_x, L_y)] = onsite_A1us[alpha, beta]   
   onsite_S = 1/4*( -mu * np.kron(tau_z, sigma_0) + Delta_0*np.kron(tau_x, sigma_0) )
   for j in range(1, L_y+1):
     for alpha in range(4):
       for beta in range(4):
         M[index(L_x, j, alpha, L_x, L_y), index(L_x, j, beta, L_x, L_y)] = onsite_S[alpha, beta]
   hopping_x_A1us = 1/2*(-t * np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_x) )
   for i in range(1, L_x-1):
     for j in range(1, L_y+1):    
       for alpha in range(4):
         for beta in range(4):
           M[index(i, j, alpha, L_x, L_y), index(i+1, j, beta, L_x, L_y)] = hopping_x_A1us[alpha, beta]
   hopping_y_A1u = 1/2*( -t * np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_y) )
   for i in range(1, L_x):
     for j in range(1, L_y+1): 
       for alpha in range(4):
         for beta in range(4):
           M[index_periodic(i, j, alpha, L_x, L_y), index_periodic(i, j+1, beta, L_x, L_y)] = hopping_y_A1u[alpha, beta]
   hopping_y_S = -t/2 * np.kron(tau_z, sigma_0)    
   for j in range(1, L_y+1): 
     for alpha in range(4):
       for beta in range(4):
         M[index_periodic(L_x, j, alpha, L_x, L_y), index_periodic(L_x, j+1, beta, L_x, L_y)] = hopping_y_S[alpha, beta]
   for j in range(1, L_y+1):
       hopping_junction_x = t_J/2 * (np.cos(phi[j-1]/2)*np.kron(tau_z, sigma_0)+1j*np.sin(phi[j-1]/2)*np.kron(tau_0, sigma_0))
       for alpha in range(4):
           for beta in range(4):
               M[index(L_x-1, j, alpha, L_x, L_y), index(L_x, j, beta, L_x, L_y)] = hopping_junction_x[alpha, beta]
   return scipy.sparse.csr_matrix(M + M.conj().T)

def Hamiltonian_A1us_S_sparse_extended(t, mu, L_S, L_A1u, L_y, Delta, Delta_0, t_J, Phi):
   r"""Return the matrix for A1us model with s and p wave pairing in a junction with an s-wave superconductor:

   .. math ::
      \vec{c_{n,m}} = (c_{n,m,\uparrow},
                       c_{n,m,\downarrow},
                       c^\dagger_{n,m,\downarrow},
                       -c^\dagger_{n,m,\uparrow})^T
             
          H = H_{A1us} + H_{S} + H_J
          
          H_{A1us} = \frac{1}{2} \sum_{n=1}^{L_x} \sum_{m=1}^{L_y} \vec{c}^\dagger_{n,m}(-\mu  \tau_z\sigma_0 + \Delta_0 \tau_x\sigma_0 )\vec{c}_{n,m} +
              \frac{1}{2} \sum_{n=1}^{L_x-1} \sum_{m=1}^{L_y} \left( \vec{c}^\dagger_{n,m}\left[ 
               -t\tau_z\sigma_0 -
               i\frac{\Delta}{2} \tau_x\sigma_x \right] \vec{c}_{n+1,m} + H.c. \right) +
              \frac{1}{2} \sum_{n=1}^{L_x} \sum_{m=1}^{L_y-1} \left( \vec{c}^\dagger_{n,m}\left[ 
               -t\tau_z\sigma_0 -
               i\frac{\Delta}{2} \tau_x\sigma_y \right] \vec{c}_{n,m+1} + H.c. \right) 
          
           H_{S} = \frac{1}{2}\sum_m^{L_y} \mathbf{c}^\dagger_{L_x,m}\left[ 
                       -\mu\tau_z\sigma_0 + \Delta_0\tau_x\sigma_0 \right] \mathbf{c}_{L_x,m}\nonumber
           				+ \frac{1}{2}
                       \sum_m^{L_y-1}\left[\mathbf{c}^\dagger_{L_x,m}\left(-t\tau_z\sigma_0 \right)\mathbf{c}_{L_x,m+1}
                       + H.c.\right]
           
          H_J = t_J/2\sum_m^{L_y}[\vec{c}_{L_x-1,m}^\dagger(cos(\phi/2)\tau_z\sigma_0+isin(\phi/2)\tau_0\sigma_0)\vec{c}_{L_x,m}+H.c.]
   """
   phi = Phi[::-1]  # I have inverted the y axis
   L_x = L_A1u + L_S
   M = scipy.sparse.lil_matrix((4*(L_x)*L_y, 4*(L_x)*L_y), dtype=complex)
   onsite_A1us = 1/4*( -mu * np.kron(tau_z, sigma_0) + Delta_0*np.kron(tau_x, sigma_0) )  # para no duplicar al sumar la traspuesta
   for i in range(1, L_x//2+1):
       for j in range(1, L_y+1):
           for alpha in range(4):
               for beta in range(4):
                   M[index(i, j, alpha, L_x, L_y), index(i, j, beta, L_x, L_y)] = onsite_A1us[alpha, beta]   
   onsite_S = 1/4*( -mu * np.kron(tau_z, sigma_0) + Delta_0*np.kron(tau_x, sigma_0) )
   for i in range(L_x//2+1, L_x+1):    
       for j in range(1, L_y+1):
           for alpha in range(4):
               for beta in range(4):
                   M[index(L_x, j, alpha, L_x, L_y), index(L_x, j, beta, L_x, L_y)] = onsite_S[alpha, beta]
   hopping_x_A1us = 1/2*(-t * np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_x) )
   for i in range(1, L_x//2):
       for j in range(1, L_y+1):    
           for alpha in range(4):
               for beta in range(4):
                   M[index(i, j, alpha, L_x, L_y), index(i+1, j, beta, L_x, L_y)] = hopping_x_A1us[alpha, beta]
   hopping_x_S = 1/2*(-t * np.kron(tau_z, sigma_0))
   for i in range(L_A1u+1, L_x):
       for j in range(1, L_y+1):    
           for alpha in range(4):
               for beta in range(4):
                   M[index(i, j, alpha, L_x, L_y), index(i+1, j, beta, L_x, L_y)] = hopping_x_S[alpha, beta]   
   hopping_y_A1u = 1/2*( -t * np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_y) )
   for i in range(1, L_x//2+1):
       for j in range(1, L_y): 
           for alpha in range(4):
               for beta in range(4):
                   M[index(i, j, alpha, L_x, L_y), index(i, j+1, beta, L_x, L_y)] = hopping_y_A1u[alpha, beta]
   hopping_y_S = -t/2 * np.kron(tau_z, sigma_0)    
   for i in range(L_x//2+1, L_x+1):
       for j in range(1, L_y): 
           for alpha in range(4):
               for beta in range(4):
                   M[index(i, j, alpha, L_x, L_y), index(i, j+1, beta, L_x, L_y)] = hopping_y_S[alpha, beta]
   for j in range(1, L_y+1):
       hopping_junction_x = t_J/2 * (np.cos(phi[j-1]/2)*np.kron(tau_z, sigma_0)+1j*np.sin(phi[j-1]/2)*np.kron(tau_0, sigma_0))
       for alpha in range(4):
           for beta in range(4):
               M[index(L_A1u, j, alpha, L_x, L_y), index(L_A1u+1, j, beta, L_x, L_y)] = hopping_junction_x[alpha, beta]
   return scipy.sparse.csr_matrix(M + M.conj().T)

def Hamiltonian_A1us_S_sparse_extended_periodic(t, mu, L_A1u, L_S, L_y, Delta, Delta_0_A1u, Delta_0_S, t_J, Phi):
   r"""Return the matrix for A1us model with s and p wave pairing in a junction with an s-wave superconductor:

   .. math ::
      \vec{c_{n,m}} = (c_{n,m,\uparrow},
                       c_{n,m,\downarrow},
                       c^\dagger_{n,m,\downarrow},
                       -c^\dagger_{n,m,\uparrow})^T
             
          H = H_{A1us} + H_{S} + H_J
          
          H_{A1us} = \frac{1}{2} \sum_{n=1}^{L_x} \sum_{m=1}^{L_y} \vec{c}^\dagger_{n,m}(-\mu  \tau_z\sigma_0 + \Delta_0 \tau_x\sigma_0 )\vec{c}_{n,m} +
              \frac{1}{2} \sum_{n=1}^{L_x-1} \sum_{m=1}^{L_y} \left( \vec{c}^\dagger_{n,m}\left[ 
               -t\tau_z\sigma_0 -
               i\frac{\Delta}{2} \tau_x\sigma_x \right] \vec{c}_{n+1,m} + H.c. \right) +
              \frac{1}{2} \sum_{n=1}^{L_x} \sum_{m=1}^{L_y-1} \left( \vec{c}^\dagger_{n,m}\left[ 
               -t\tau_z\sigma_0 -
               i\frac{\Delta}{2} \tau_x\sigma_y \right] \vec{c}_{n,m+1} + H.c. \right) 
          
           H_{S} = \frac{1}{2}\sum_m^{L_y} \mathbf{c}^\dagger_{L_x,m}\left[ 
                       -\mu\tau_z\sigma_0 + \Delta_0\tau_x\sigma_0 \right] \mathbf{c}_{L_x,m}\nonumber
           				+ \frac{1}{2}
                       \sum_m^{L_y-1}\left[\mathbf{c}^\dagger_{L_x,m}\left(-t\tau_z\sigma_0 \right)\mathbf{c}_{L_x,m+1}
                       + H.c.\right]
           
          H_J = t_J/2\sum_m^{L_y}[\vec{c}_{L_x-1,m}^\dagger(cos(\phi/2)\tau_z\sigma_0+isin(\phi/2)\tau_0\sigma_0)\vec{c}_{L_x,m}+H.c.]
   """
   phi = Phi[::-1]  # I have inverted the y axis
   L_x = L_A1u + L_S
   M = scipy.sparse.lil_matrix((4*(L_x)*L_y, 4*(L_x)*L_y), dtype=complex)
   onsite_A1us = 1/4*( -mu * np.kron(tau_z, sigma_0) + Delta_0_A1u*np.kron(tau_x, sigma_0) )  # para no duplicar al sumar la traspuesta
   for i in range(1, L_A1u+1):
       for j in range(1, L_y+1):
           for alpha in range(4):
               for beta in range(4):
                   M[index(i, j, alpha, L_x, L_y), index(i, j, beta, L_x, L_y)] = onsite_A1us[alpha, beta]   
   onsite_S = 1/4*( -mu * np.kron(tau_z, sigma_0) + Delta_0_S*np.kron(tau_x, sigma_0) )
   for i in range(L_A1u+1, L_x+1):    
       for j in range(1, L_y+1):
           for alpha in range(4):
               for beta in range(4):
                   M[index(L_x, j, alpha, L_x, L_y), index(L_x, j, beta, L_x, L_y)] = onsite_S[alpha, beta]
   hopping_x_A1us = 1/2*(-t * np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_x) )
   for i in range(1, L_A1u):
       for j in range(1, L_y+1):    
           for alpha in range(4):
               for beta in range(4):
                   M[index(i, j, alpha, L_x, L_y), index(i+1, j, beta, L_x, L_y)] = hopping_x_A1us[alpha, beta]
   hopping_x_S = 1/2*(-t * np.kron(tau_z, sigma_0))
   for i in range(L_A1u+1, L_x):
       for j in range(1, L_y+1):    
           for alpha in range(4):
               for beta in range(4):
                   M[index(i, j, alpha, L_x, L_y), index(i+1, j, beta, L_x, L_y)] = hopping_x_S[alpha, beta]
   hopping_y_A1u = 1/2*( -t * np.kron(tau_z, sigma_0) - 1j*Delta/2 * np.kron(tau_x, sigma_y) )
   for i in range(1, L_A1u+1):
       for j in range(1, L_y+1): 
           for alpha in range(4):
               for beta in range(4):
                   M[index_periodic(i, j, alpha, L_x, L_y), index_periodic(i, j+1, beta, L_x, L_y)] = hopping_y_A1u[alpha, beta]
   hopping_y_S = -t/2 * np.kron(tau_z, sigma_0)    
   for i in range(L_A1u+1, L_x+1):
       for j in range(1, L_y+1): 
           for alpha in range(4):
               for beta in range(4):
                   M[index_periodic(i, j, alpha, L_x, L_y), index_periodic(i, j+1, beta, L_x, L_y)] = hopping_y_S[alpha, beta]
   for j in range(1, L_y+1):
       hopping_junction_x = t_J/2 * (np.cos(phi[j-1]/2)*np.kron(tau_z, sigma_0)+1j*np.sin(phi[j-1]/2)*np.kron(tau_0, sigma_0))
       for alpha in range(4):
           for beta in range(4):
               M[index(L_A1u, j, alpha, L_x, L_y), index(L_A1u+1, j, beta, L_x, L_y)] = hopping_junction_x[alpha, beta]
   return scipy.sparse.csr_matrix(M + M.conj().T)

def Hamiltonian_A1us_k(t, k, mu, L, Delta_A1u, Delta_S):
    r"""Returns the H_k matrix for A1u (with s-wave superconductivity) with:
    
    .. math::
        H_{A1us} = \frac{1}{2}\sum_k H_{A1us,k}
        
        H_{A1us,k} = \sum_n^L \vec{c}^\dagger_n\left[ 
            \xi_k\tau_z\sigma_0 +
            \Delta sin(k_y)\tau_x\sigma_y + \Delta_0 \tau_x\sigma_0\right] +
            \sum_n^{L-1}\left(\vec{c}^\dagger_n(-t\tau_z\sigma_0 + \frac{\Delta}{2i}\tau_x\sigma_x)\vec{c}_{n+1}
            + H.c. \right)
        
        \vec{c} = (c_{k,\uparrow}, c_{k,\downarrow},c^\dagger_{-k,\downarrow},-c^\dagger_{-k,\uparrow})^T
    
        \xi_k = -2tcos(k) - \mu
    """
    M = np.zeros((4*L, 4*L), dtype=complex)
    chi_k = -mu - 2*t * np.cos(k)
    onsite_A1u = 1/4*(chi_k * np.kron(tau_z, sigma_0) + \
            Delta_A1u*np.sin(k) * np.kron(tau_x, sigma_y) +
            Delta_S * np.kron(tau_x, sigma_0))
    for i in range(1, L+1):
        for alpha in range(4):
            for beta in range(4):
                M[index_k(i, alpha, L), index_k(i, beta, L)] = onsite_A1u[alpha, beta]
    hopping_A1u = 1/2*(-t*np.kron(tau_z, sigma_0) - 1j*Delta_A1u/2 * np.kron(tau_x, sigma_x) )
    for i in range(1, L):
        for alpha in range(4):
            for beta in range(4):
                M[index_k(i, alpha, L), index_k(i+1, beta, L)] = hopping_A1u[alpha, beta]
    return M + M.conj().T

def Hamiltonian_S_sparse_periodic(t, mu, L_x, L_y, Delta_0):
   r"""Return the matrix for S model with s wave pairing:

   .. math ::
      \vec{c_{n,m}} = (c_{n,m,\uparrow},
                       c_{n,m,\downarrow},
                       c^\dagger_{n,m,\downarrow},
                       -c^\dagger_{n,m,\uparrow})^T
             
           H_{S} = \frac{1}{2}\sum_m^{L_y} \mathbf{c}^\dagger_{L_x,m}\left[ 
                       -\mu\tau_z\sigma_0 + \Delta_0\tau_x\sigma_0 \right] \mathbf{c}_{L_x,m}\nonumber
           				+ \frac{1}{2}
                       \sum_m^{L_y-1}\left[\mathbf{c}^\dagger_{L_x,m}\left(-t\tau_z\sigma_0 \right)\mathbf{c}_{L_x,m+1}
                       + H.c.\right]
           
   """
   M = scipy.sparse.lil_matrix((4*(L_x)*L_y, 4*(L_x)*L_y), dtype=complex)
   onsite_S = 1/4*( -mu * np.kron(tau_z, sigma_0) + Delta_0*np.kron(tau_x, sigma_0) )
   for i in range(1, L_x+1):    
       for j in range(1, L_y+1):
           for alpha in range(4):
               for beta in range(4):
                   M[index(L_x, j, alpha, L_x, L_y), index(L_x, j, beta, L_x, L_y)] = onsite_S[alpha, beta]
   hopping_x_S = 1/2*(-t * np.kron(tau_z, sigma_0))
   for i in range(1, L_x):
       for j in range(1, L_y+1):    
           for alpha in range(4):
               for beta in range(4):
                   M[index(i, j, alpha, L_x, L_y), index(i+1, j, beta, L_x, L_y)] = hopping_x_S[alpha, beta]
   hopping_y_S = -t/2 * np.kron(tau_z, sigma_0)    
   for i in range(1, L_x+1):
       for j in range(1, L_y+1): 
           for alpha in range(4):
               for beta in range(4):
                   M[index_periodic(i, j, alpha, L_x, L_y), index_periodic(i, j+1, beta, L_x, L_y)] = hopping_y_S[alpha, beta]
   return scipy.sparse.csr_matrix(M + M.conj().T)

def Hamiltonian_S_periodic(t, mu, L_x, L_y, Delta_0):
   r"""Return the matrix for S model with s wave pairing:

   .. math ::
      \vec{c_{n,m}} = (c_{n,m,\uparrow},
                       c_{n,m,\downarrow},
                       c^\dagger_{n,m,\downarrow},
                       -c^\dagger_{n,m,\uparrow})^T
             
           H_{S} = \frac{1}{2}\sum_m^{L_y} \mathbf{c}^\dagger_{L_x,m}\left[ 
                       -\mu\tau_z\sigma_0 + \Delta_0\tau_x\sigma_0 \right] \mathbf{c}_{L_x,m}\nonumber
           				+ \frac{1}{2}
                       \sum_m^{L_y-1}\left[\mathbf{c}^\dagger_{L_x,m}\left(-t\tau_z\sigma_0 \right)\mathbf{c}_{L_x,m+1}
                       + H.c.\right]
           
   """
   M = np.zeros((4*(L_x)*L_y, 4*(L_x)*L_y), dtype=complex)
   onsite_S = 1/4*( -mu * np.kron(tau_z, sigma_0) + Delta_0*np.kron(tau_x, sigma_0) )
   for i in range(1, L_x+1):    
       for j in range(1, L_y+1):
           for alpha in range(4):
               for beta in range(4):
                   M[index(i, j, alpha, L_x, L_y), index(i, j, beta, L_x, L_y)] = onsite_S[alpha, beta]
   hopping_x_S = 1/2*(-t * np.kron(tau_z, sigma_0))
   for i in range(1, L_x):
       for j in range(1, L_y+1):    
           for alpha in range(4):
               for beta in range(4):
                   M[index(i, j, alpha, L_x, L_y), index(i+1, j, beta, L_x, L_y)] = hopping_x_S[alpha, beta]
   hopping_y_S = -t/2 * np.kron(tau_z, sigma_0)    
   for i in range(1, L_x+1):
       for j in range(1, L_y): 
           for alpha in range(4):
               for beta in range(4):
                   M[index(i, j, alpha, L_x, L_y), index(i, j+1, beta, L_x, L_y)] = hopping_y_S[alpha, beta]
   return M + M.conj().T
