#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:02:34 2023

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

# Delta = 100
# m_0 = 10
# L = 20

# Analytical parameters
Phi = 0.1*np.pi  #height of the phase soliton around flux pi
t_J = 1   #t/2
m_0 = t_J*np.sin(Phi/2)
L = 15      #L_y//2
Delta = 2

# Numerical parameters
L_x = 200
L_y = 400
t = 1
mu = -2  #-2
k = 8   #number of eigenvalues


def trascendental_equation(k, m_0, Delta, L):
    """
    Wavevector of the localized state.

    Parameters
    ----------
    m_0 : float
        Mass.
    Delta : floar
        Gap.
    L : float
        Length.

    Returns
    -------
    A function whose roots represents the trascendental equation.
        (m_0/Delta)**2 - k**2 - (m_0/Delta)**2 * np.exp(-2*k*L)=0
    """
    return (m_0/Delta)**2 - k**2 - (m_0/Delta)**2 * np.exp(-2*k*L)

def Kappa(m_0, Delta, L):
    """
    Wavevector of the localized state.

    Parameters
    ----------
    (y, kappa, m_0, Delta, L)m_0 : float
        Mass.
    Delta : floar
        Gap.
    L : float
        Length.

    Returns
    -------
    The wavevector k solving:
        (m_0/Delta)**2 - k**2 = (m_0/Delta)**2 * np.exp(-2*k*L)
    """
    return root(trascendental_equation, 1, args=(m_0, Delta, L)).x

#%% Plotting trascendental equation

kappa = Kappa(m_0, Delta, L)
alpha = kappa*Delta/m_0
K = np.linspace(0, kappa, 1000)
fig, ax = plt.subplots()
ax.plot(K, [(m_0/Delta)**2 - k**2 for k in K])
ax.plot(K, [(m_0/Delta)**2 * np.exp(-2*k*L) for k in K])

def psi_1_prime(y, kappa, m_0, Delta, L):
    alpha = kappa*Delta/m_0
    C_L = np.sqrt(kappa/(2*(1-alpha)*(alpha**2-np.exp(-2*kappa*L)*kappa*L)))
    if y<=0:
        return C_L*alpha*np.exp(kappa*(y-L))
    elif (y>0 and y<=L):
        return C_L*((alpha-1)*np.exp(kappa*(y-L)) + np.exp(-kappa*(y+L)) )
    else:
        return C_L*((alpha-1)*np.exp(kappa*L) + np.exp(-kappa*L)) * np.exp(-kappa*y)

def psi_2_prime_plus(y, kappa, m_0, Delta, L):
    alpha = kappa*Delta/m_0
    C_L = np.sqrt(kappa/(2*(1-alpha)*(alpha**2-np.exp(-2*kappa*L)*kappa*L)))
    if y<=0:
        return -np.sqrt((1-alpha)/(1+alpha))*psi_1_prime(y, kappa, m_0, Delta, L)
    elif (y>0 and y<=L):
        return -C_L*(np.sqrt(1-alpha**2)*np.exp(kappa*(y-L)) - np.sqrt((1-alpha)/(1+alpha))*np.exp(-kappa*(y+L)) )
    else:
        return -np.sqrt((1+alpha)/(1-alpha))*psi_1_prime(y, kappa, m_0, Delta, L)

def psi_2_prime_minus(y, kappa, m_0, Delta, L):
    alpha = kappa*Delta/m_0
    C_L = np.sqrt(kappa/(2*(1-alpha)*(alpha**2-np.exp(-2*kappa*L)*kappa*L)))
    if y<=0:
        return np.sqrt((1-alpha)/(1+alpha))*psi_1_prime(y, kappa, m_0, Delta, L)
    elif (y>0 and y<=L):
        return C_L*(np.sqrt(1-alpha**2)*np.exp(kappa*(y-L)) - np.sqrt((1-alpha)/(1+alpha))*np.exp(-kappa*(y+L)) )
    else:
        return np.sqrt((1+alpha)/(1-alpha))*psi_1_prime(y, kappa, m_0, Delta, L)


y = np.arange(-L_y//2+L//2, L_y//2+L//2)
fig, ax = plt.subplots()
# ax.plot(y, [psi_1_prime(y_value, kappa, m_0, Delta, L) for y_value in y])

def psi_1_plus(y, kappa, m_0, Delta, L):
    return 1/2*(-1j*psi_1_prime(y, kappa, m_0, Delta, L) + psi_2_prime_plus(y, kappa, m_0, Delta, L))
def psi_1_minus(y, kappa, m_0, Delta, L):
    return 1/2*(-1j*psi_1_prime(y, kappa, m_0, Delta, L) + psi_2_prime_minus(y, kappa, m_0, Delta, L))
def psi_2_plus(y, kappa, m_0, Delta, L):
    return 1/2*(psi_1_prime(y, kappa, m_0, Delta, L) -1j* psi_2_prime_plus(y, kappa, m_0, Delta, L))
def psi_2_minus(y, kappa, m_0, Delta, L):
    return 1/2*(psi_1_prime(y, kappa, m_0, Delta, L) -1j* psi_2_prime_minus(y, kappa, m_0, Delta, L))
def psi_3_plus(y, kappa, m_0, Delta, L):
    return 1/2*(-psi_1_prime(y, kappa, m_0, Delta, L) + 1j*psi_2_prime_plus(y, kappa, m_0, Delta, L))
def psi_3_minus(y, kappa, m_0, Delta, L):
    return 1/2*(-psi_1_prime(y, kappa, m_0, Delta, L) + 1j*psi_2_prime_minus(y, kappa, m_0, Delta, L))
def psi_4_plus(y, kappa, m_0, Delta, L):
    return 1/2*(1j*psi_1_prime(y, kappa, m_0, Delta, L) - psi_2_prime_plus(y, kappa, m_0, Delta, L))
def psi_4_minus(y, kappa, m_0, Delta, L):
    return 1/2*(1j*psi_1_prime(y, kappa, m_0, Delta, L) - psi_2_prime_minus(y, kappa, m_0, Delta, L))

def psi_plus(y, kappa, m_0, Delta, L):
    return 1/2*np.array([psi_1_plus(y, kappa, m_0, Delta, L), psi_2_plus(y, kappa, m_0, Delta, L), psi_3_plus(y, kappa, m_0, Delta, L), psi_4_plus(y, kappa, m_0, Delta, L)])

def psi_left_plus(y, kappa, m_0, Delta, L):
    r"""
    Positive energy wavefunction for the left part of the junction in the fermionic base.
    The normalization is 1.
    .. math ::
        (c_{l_\uparrow}, c_{l_\downarrow}, c^\dagger_{l_\downarrow}, -c^\dagger_{l_\uparrow})
    """
    return 1/np.sqrt(2)*np.array([psi_1_plus(y, kappa, m_0, Delta, L), psi_2_plus(y, kappa, m_0, Delta, L), -1j*psi_2_plus(y, kappa, m_0, Delta, L), -1j*psi_1_plus(y, kappa, m_0, Delta, L)])

# ax.plot(y, [np.abs(psi_1_plus(y_value, kappa, m_0, Delta, L)) for y_value in y])
# ax.plot(y, [np.abs(psi_1_minus(y_value, kappa, m_0, Delta, L)) for y_value in y])
# ax.plot(y, [np.abs(psi_3_plus(y_value, kappa, m_0, Delta, L)) for y_value in y])
# ax.plot(y, [np.abs(psi_3_minus(y_value, kappa, m_0, Delta, L)) for y_value in y])
# ax.plot(y, [np.abs(psi_2_plus(y_value, kappa, m_0, Delta, L)) for y_value in y])
# ax.plot(y, [np.abs(psi_2_minus(y_value, kappa, m_0, Delta, L)) for y_value in y])
# ax.plot(y, [np.abs(psi_4_plus(y_value, kappa, m_0, Delta, L)) for y_value in y])
# ax.plot(y, [np.abs(psi_4_minus(y_value, kappa, m_0, Delta, L)) for y_value in y])

ax.plot(y, [np.linalg.norm(psi_plus(y_value, kappa, m_0, Delta, L)) for y_value in y])

#%% Sparse diagonalization
from hamiltonians import Hamiltonian_soliton_A1u_sparse
import scipy
from functions import get_components



H = Hamiltonian_soliton_A1u_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi, L=L)
params = {"t": t, "mu": mu, "L_x": L_x, "L_y": L_y, "Delta": Delta, "t_J": t_J, "Phi": Phi, "L": L}
eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H, k=k, sigma=0) 
index = np.arange(k)   #which zero mode (less than k)
Probability_density = []

for i in index:
    destruction_up, destruction_down, creation_down, creation_up = get_components(eigenvectors_sparse[:,i], L_x, L_y)
    Probability_density.append(np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)

index = 4
fig, ax = plt.subplots()
image = ax.imshow(Probability_density[index], cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
plt.colorbar(image)
#ax.set_title(f"{params}")
ax.set_xlabel("x")
ax.set_ylabel("y")
#ax.text(5,25, rf'$index={index}; \Phi={np.round(Phi, 2)}$')
ax.text(5,25, rf'$\Phi={np.round(Phi, 2)}$; $index={index}$')
#plt.plot(probability_density[10,:,0])
ax.set_title("Probability density")
plt.tight_layout()

# probability_density_at_junction_left = probability_density[index][:, L_x//2-1]
probability_density_at_junction_left = Probability_density[index][:, L_x//2-1]/np.sum(Probability_density[index][:, L_x//2-1])

fig, ax = plt.subplots()
ax.plot(y , probability_density_at_junction_left, label="Numerical")
probability_density_analytical = np.array([np.linalg.norm(psi_left_plus(y_value, kappa, m_0, Delta, L))**2 for y_value in y])
ax.plot(y, probability_density_analytical, label="Analytical")
ax.set_xlabel("y")
ax.set_ylabel("Probability density")
ax.set_title(f"Probability density at the junction with L={L}, Phi={Phi:.2}, alpha={alpha[0]} and Delta={Delta}")
ax.legend()
plt.tight_layout()
