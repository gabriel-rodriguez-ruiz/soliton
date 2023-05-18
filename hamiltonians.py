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
     
     \text{real space}
     
     (c_{1L_y} &... c_{L_xL_y})
                      
     (c_{11} &... c_{L_x1})
  """
  if (i>L_x or j>L_y):
      raise Exception("Site index should not be greater than samplesize.")
  return alpha + 4*( L_y*(i-1) + j - 1)
    
def Hamiltonian_soliton_A1u(t, mu, L_x, L_y, Delta, t_J, Phi, L):
    r"""Return the matrix for A1u model in a junction with a superconductor with:

    .. math ::
       \vec{c_{n,m}} = (c_{n,m,\uparrow},
                        c_{n,m,\downarrow},
                        c^\dagger_{n,m,\downarrow},
                        -c^\dagger_{n,m,\uparrow})^T
       
       H_{A1u} = \frac{1}{2} \sum_n^{L_x-1} \sum_m^{L_y} (-\mu \vec{c}^\dagger_{n,m} \tau_z\sigma_0  \vec{c}_{n,m}) +
           \frac{1}{2} \sum_n^{L_x-2} \sum_m^{L_y} \left( \vec{c}^\dagger_{n,m}\left[ 
            -t\tau_z\sigma_0 -
            i\frac{\Delta}{2} \tau_x\sigma_x \right] \vec{c}_{n+1,m} + H.c. \right) +
           \frac{1}{2} \sum_n^{L_x-1} \sum_m^{L_y-1} \left( \vec{c}^\dagger_{n,m}\left[ 
            -t\tau_z\sigma_0 -
            i\frac{\Delta}{2} \tau_x\sigma_y \right] \vec{c}_{n,m+1} + H.c. \right) 
       
        H_J = t_J/2\sum_m^{L_y}[\vec{c}_{L_x-1,m}^\dagger(cos(\phi/2)\tau_0\sigma_0+(1+2\left[\theta(m-(\frac{L_y+L}{2}))-\theta(m-(\frac{L_y-L}{2}))\right])isin(\phi/2)\tau_z\sigma_0)\vec{c}_{L_x,m}+H.c.]
    """
    M = np.zeros((4*(L_x)*L_y, 4*(L_x)*L_y), dtype=complex)
    onsite_A1u = -mu/4 * np.kron(tau_z, sigma_0)   # para no duplicar al sumar la traspuesta
    for i in range(1, L_x+1):
      for j in range(1, L_y+1):
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j, beta, L_x, L_y)] = onsite_A1u[alpha, beta]   
    hopping_x_A1u = -t/2 * np.kron(tau_z, sigma_0) - 1j*Delta/4 * np.kron(tau_x, sigma_x)
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
    hopping_y_A1u = -t/2 * np.kron(tau_z, sigma_0) - 1j*Delta/4 * np.kron(tau_x, sigma_y)
    for i in range(1, L_x+1):
      for j in range(1, L_y): 
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j+1, beta, L_x, L_y)] = hopping_y_A1u[alpha, beta]
    hopping_junction_x = t_J/2 * (np.cos(Phi/2)*np.kron(tau_0, sigma_0) + 1j*np.sin(Phi/2)*np.kron(tau_z, sigma_0))
    for j in range(1, L_y+1): 
      for alpha in range(4):
        for beta in range(4):
            if j<=(L_y-L)//2:
                M[index(L_x//2, j, alpha, L_x, L_y), index(L_x//2+1, j, beta, L_x, L_y)] = hopping_junction_x[alpha, beta]
            elif j>(L_y-L)//2 and j<=(L_y+L)//2:
                M[index(L_x//2, j, alpha, L_x, L_y), index(L_x//2+1, j, beta, L_x, L_y)] = hopping_junction_x[alpha, beta].conj()
            else:
                M[index(L_x//2, j, alpha, L_x, L_y), index(L_x//2+1, j, beta, L_x, L_y)] = hopping_junction_x[alpha, beta]
    return M + M.conj().T

def Hamiltonian_soliton_A1u_sparse(t, mu, L_x, L_y, Delta, t_J, Phi, L):
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
         
            H_J = t_J/2\sum_m^{L_y}[\vec{c}_{L_x-1,m}^\dagger(\left(-1+2(\theta(m-\frac{L_y-L}{2})-\theta(m-\frac{L_y+L}{2})\right)sin(\epsilon/2)\tau_z\sigma_0+icos(\epsilon/2)\tau_0\sigma_0)\vec{c}_{L_x,m}+H.c.]
    """
    M = scipy.sparse.lil_matrix((4*(L_x)*L_y, 4*(L_x)*L_y), dtype=complex)
    onsite_A1u = -mu/4 * np.kron(tau_z, sigma_0)   # para no duplicar al sumar la traspuesta
    for i in range(1, L_x+1):
      for j in range(1, L_y+1):
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j, beta, L_x, L_y)] = onsite_A1u[alpha, beta]   
    hopping_x_A1u = -t/2 * np.kron(tau_z, sigma_0) - 1j*Delta/4 * np.kron(tau_x, sigma_x)
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
    hopping_y_A1u = -t/2 * np.kron(tau_z, sigma_0) - 1j*Delta/4 * np.kron(tau_x, sigma_y)
    for i in range(1, L_x+1):
      for j in range(1, L_y): 
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j+1, beta, L_x, L_y)] = hopping_y_A1u[alpha, beta]
    #hopping_junction_x = t_J/2 * (np.cos(Phi/2)*np.kron(tau_0, sigma_0) + 1j*np.sin(Phi/2)*np.kron(tau_z, sigma_0))
    hopping_junction_x_far_away = t_J/2 * (-np.sin(Phi/2)*np.kron(tau_z, sigma_0) + 1j*np.sin(Phi/2)*np.kron(tau_0, sigma_0))
    hopping_junction_x_middle = t_J/2 * (np.sin(Phi/2)*np.kron(tau_z, sigma_0) + 1j*np.sin(Phi/2)*np.kron(tau_0, sigma_0))
    hopping_junction_x_zero = t_J/2 * 1j*np.sin(Phi/2)*np.kron(tau_0, sigma_0)
    for j in range(1, L_y+1): 
      for alpha in range(4):
        for beta in range(4):
            if j<(L_y-L)//2:
                M[index(L_x//2, j, alpha, L_x, L_y), index(L_x//2+1, j, beta, L_x, L_y)] = hopping_junction_x_far_away[alpha, beta]
            elif j==(L_y-L)//2:
                M[index(L_x//2, j, alpha, L_x, L_y), index(L_x//2+1, j, beta, L_x, L_y)] = hopping_junction_x_zero[alpha, beta]
            elif j>(L_y-L)//2 and j<(L_y+L)//2:
                M[index(L_x//2, j, alpha, L_x, L_y), index(L_x//2+1, j, beta, L_x, L_y)] = hopping_junction_x_middle[alpha, beta]
            elif j==(L_y+L)//2:
                M[index(L_x//2, j, alpha, L_x, L_y), index(L_x//2+1, j, beta, L_x, L_y)] = hopping_junction_x_zero[alpha, beta]
            else:
                M[index(L_x//2, j, alpha, L_x, L_y), index(L_x//2+1, j, beta, L_x, L_y)] = hopping_junction_x_far_away[alpha, beta]
    return scipy.sparse.csr_matrix(M + M.conj().T)

def Hamiltonian_A1u_single_step_sparse(t, mu, L_x, L_y, Delta, t_J, Phi):
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
         
            H_J = t_J/2\sum_m^{L_y}[\vec{c}_{L_x-1,m}^\dagger(\left(\theta(\frac{L_y}{2}-m)-\theta(m-\frac{L_y}{2})\right)sin(\phi/2)\tau_z\sigma_0+icos(\phi/2)\tau_0\sigma_0)\vec{c}_{L_x,m}+H.c.]
    """
    M = scipy.sparse.lil_matrix((4*(L_x)*L_y, 4*L_x*L_y), dtype=complex)
    onsite_A1u = -mu/4 * np.kron(tau_z, sigma_0)   # para no duplicar al sumar la traspuesta
    for i in range(1, L_x+1):
      for j in range(1, L_y+1):
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j, beta, L_x, L_y)] = onsite_A1u[alpha, beta]   
    hopping_x_A1u = -t/2 * np.kron(tau_z, sigma_0) - 1j*Delta/4 * np.kron(tau_x, sigma_x)
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
    hopping_y_A1u = -t/2 * np.kron(tau_z, sigma_0) - 1j*Delta/4 * np.kron(tau_x, sigma_y)
    for i in range(1, L_x+1):
      for j in range(1, L_y): 
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j+1, beta, L_x, L_y)] = hopping_y_A1u[alpha, beta]
    hopping_junction_x_less = t_J/2 * (-np.sin(Phi/2)*np.kron(tau_z, sigma_0) + 1j*np.cos(Phi/2)*np.kron(tau_0, sigma_0))
    hopping_junction_x_bigger = t_J/2 * (np.sin(Phi/2)*np.kron(tau_z, sigma_0) + 1j*np.cos(Phi/2)*np.kron(tau_0, sigma_0))
    hopping_junction_x_zero = t_J/2 * 1j*np.cos(Phi/2)*np.kron(tau_0, sigma_0)
    for j in range(1, L_y+1): 
      for alpha in range(4):
        for beta in range(4):
            if j<L_y//2:
                M[index(L_x//2, j, alpha, L_x, L_y), index(L_x//2+1, j, beta, L_x, L_y)] = hopping_junction_x_less[alpha, beta]
            elif j>L_y//2:
                M[index(L_x//2, j, alpha, L_x, L_y), index(L_x//2+1, j, beta, L_x, L_y)] = hopping_junction_x_bigger[alpha, beta]
            else:
                M[index(L_x//2, j, alpha, L_x, L_y), index(L_x//2+1, j, beta, L_x, L_y)] = hopping_junction_x_zero[alpha, beta]                
    return scipy.sparse.csr_matrix(M + M.conj().T)

def Hamiltonian_A1u_S(t, mu, L_x, L_y, Delta, t_J, Phi):
    r"""Return the matrix for A1u model in a junction with a superconductor with:

    .. math ::
       \vec{c_{n,m}} = (c_{n,m,\uparrow},
                        c_{n,m,\downarrow},
                        c^\dagger_{n,m,\downarrow},
                        -c^\dagger_{n,m,\uparrow})^T
       
       H_{A1u} = \frac{1}{2} \sum_n^{L_x-1} \sum_m^{L_y} (-\mu \vec{c}^\dagger_{n,m} \tau_z\sigma_0  \vec{c}_{n,m}) +
           \frac{1}{2} \sum_n^{L_x-2} \sum_m^{L_y} \left( \vec{c}^\dagger_{n,m}\left[ 
            -t\tau_z\sigma_0 -
            i\frac{\Delta}{2} \tau_x\sigma_x \right] \vec{c}_{n+1,m} + H.c. \right) +
           \frac{1}{2} \sum_n^{L_x-1} \sum_m^{L_y-1} \left( \vec{c}^\dagger_{n,m}\left[ 
            -t\tau_z\sigma_0 -
            i\frac{\Delta}{2} \tau_x\sigma_y \right] \vec{c}_{n,m+1} + H.c. \right) 
       
        H_J = t_J/2\sum_m^{L_y}[\vec{c}_{L_x-1,m}(cos(\phi/2)\tau_0\sigma_0+(\theta(L_y/2-m)-\theta(m-L_y/2))isin(\phi/2)\tau_z\sigma_0)\vec{c}_{L_x,m}+H.c.]
    """
    M = scipy.sparse.lil_matrix((4*(L_x)*L_y, 4*(L_x)*L_y), dtype=complex)
    onsite_A1u = -mu/4 * np.kron(tau_z, sigma_0)   # para no duplicar al sumar la traspuesta
    for i in range(1, L_x):
      for j in range(1, L_y+1):
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j, beta, L_x, L_y)] = onsite_A1u[alpha, beta]   
    onsite_S = -mu/4 * np.kron(tau_z, sigma_0) + Delta/4*np.kron(tau_x, sigma_0) 
    for j in range(1, L_y+1):
      for alpha in range(4):
        for beta in range(4):
          M[index(L_x, j, alpha, L_x, L_y), index(L_x, j, beta, L_x, L_y)] = onsite_S[alpha, beta]
    hopping_x_A1u = -t/2 * np.kron(tau_z, sigma_0) - 1j*Delta/4 * np.kron(tau_x, sigma_x)
    for i in range(1, L_x-1):
      for j in range(1, L_y+1):    
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i+1, j, beta, L_x, L_y)] = hopping_x_A1u[alpha, beta]
    hopping_y = -t/2 * np.kron(tau_z, sigma_0) - 1j*Delta/4 * np.kron(tau_x, sigma_y)
    for i in range(1, L_x):
      for j in range(1, L_y): 
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j+1, beta, L_x, L_y)] = hopping_y[alpha, beta]
    hopping_junction_x = t_J/2 * (np.cos(Phi/2)*np.kron(tau_0, sigma_0) + 1j*np.sin(Phi/2)*np.kron(tau_z, sigma_0))
    for j in range(1, L_y+1): 
      for alpha in range(4):
        for beta in range(4):
            if j<=L_y//2:
                M[index(L_x-1, j, alpha, L_x, L_y), index(L_x, j, beta, L_x, L_y)] = hopping_junction_x[alpha, beta]
            else:
                M[index(L_x-1, j, alpha, L_x, L_y), index(L_x, j, beta, L_x, L_y)] = hopping_junction_x[alpha, beta].conj()
    return scipy.sparse.csr_matrix(M + M.conj().T)

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

def Hamiltonian_A1u_S_double_soliton(t, mu, L_x, L_y, Delta, t_J, Phi, L):
    r"""Return the matrix for A1u model in a junction with a superconductor with:

    .. math ::
       \vec{c_{n,m}} = (c_{n,m,\uparrow},
                        c_{n,m,\downarrow},
                        c^\dagger_{n,m,\downarrow},
                        -c^\dagger_{n,m,\uparrow})^T
       
       H_{A1u} = \frac{1}{2} \sum_n^{L_x-1} \sum_m^{L_y} (-\mu \vec{c}^\dagger_{n,m} \tau_z\sigma_0  \vec{c}_{n,m}) +
           \frac{1}{2} \sum_n^{L_x-2} \sum_m^{L_y} \left( \vec{c}^\dagger_{n,m}\left[ 
            -t\tau_z\sigma_0 -
            i\frac{\Delta}{2} \tau_x\sigma_x \right] \vec{c}_{n+1,m} + H.c. \right) +
           \frac{1}{2} \sum_n^{L_x-1} \sum_m^{L_y-1} \left( \vec{c}^\dagger_{n,m}\left[ 
            -t\tau_z\sigma_0 -
            i\frac{\Delta}{2} \tau_x\sigma_y \right] \vec{c}_{n,m+1} + H.c. \right) 
       
        H_J = t_J/2\sum_m^{L_y}[\vec{c}_{L_x-1,m}(cos(\phi/2)\tau_0\sigma_0+[1+2(\theta(m-(L_y+L)/2)-\theta(m-(L_y-L)/2))isin(\phi/2)\tau_z\sigma_0)\vec{c}_{L_x,m}+H.c.]
    """
    M = scipy.sparse.lil_matrix((4*(L_x)*L_y, 4*(L_x)*L_y), dtype=complex)
    onsite_A1u = -mu/4 * np.kron(tau_z, sigma_0)   # para no duplicar al sumar la traspuesta
    for i in range(1, L_x):
      for j in range(1, L_y+1):
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j, beta, L_x, L_y)] = onsite_A1u[alpha, beta]   
    onsite_S = -mu/4 * np.kron(tau_z, sigma_0) + Delta/4*np.kron(tau_x, sigma_0) 
    for j in range(1, L_y+1):
      for alpha in range(4):
        for beta in range(4):
          M[index(L_x, j, alpha, L_x, L_y), index(L_x, j, beta, L_x, L_y)] = onsite_S[alpha, beta]
    hopping_x_A1u = -t/2 * np.kron(tau_z, sigma_0) - 1j*Delta/4 * np.kron(tau_x, sigma_x)
    for i in range(1, L_x-1):
      for j in range(1, L_y+1):    
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i+1, j, beta, L_x, L_y)] = hopping_x_A1u[alpha, beta]
    hopping_y = -t/2 * np.kron(tau_z, sigma_0) - 1j*Delta/4 * np.kron(tau_x, sigma_y)
    for i in range(1, L_x):
      for j in range(1, L_y): 
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j+1, beta, L_x, L_y)] = hopping_y[alpha, beta]
    hopping_junction_x = t_J/2 * (np.cos(Phi/2)*np.kron(tau_0, sigma_0) + 1j*np.sin(Phi/2)*np.kron(tau_z, sigma_0))
    for j in range(1, L_y+1): 
      for alpha in range(4):
        for beta in range(4):
            if j<=(L_y-L)//2:
                M[index(L_x-1, j, alpha, L_x, L_y), index(L_x, j, beta, L_x, L_y)] = hopping_junction_x[alpha, beta]
            elif j>(L_y-L)//2 and j<=(L_y+L)//2:
                M[index(L_x-1, j, alpha, L_x, L_y), index(L_x, j, beta, L_x, L_y)] = hopping_junction_x[alpha, beta].conj()
            else:
                M[index(L_x-1, j, alpha, L_x, L_y), index(L_x, j, beta, L_x, L_y)] = hopping_junction_x[alpha, beta]
    return scipy.sparse.csr_matrix(M + M.conj().T)
