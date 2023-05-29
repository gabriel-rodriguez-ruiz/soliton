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
L = 30      #L_y//2
Delta = 1

# Numerical parameters
L_x = 200
L_y = 200
t = 1
mu = -2  #-2
k = 4   #number of eigenvalues


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
    return -psi_1_prime(-y+L, kappa, m_0, Delta, L)

def psi_2_prime_minus(y, kappa, m_0, Delta, L):
    return psi_1_prime(-y+L, kappa, m_0, Delta, L)

y = np.arange(-L_y//2+L//2, L_y//2+L//2)
fig, ax = plt.subplots()
ax.plot(y, [psi_1_prime(y_value, kappa, m_0, Delta, L) for y_value in y])
ax.plot(y, [psi_2_prime_plus(y_value, kappa, m_0, Delta, L) for y_value in y])
ax.plot(y, [psi_2_prime_minus(y_value, kappa, m_0, Delta, L) for y_value in y])


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

def psi_down_minus_right(y, kappa, m_0, Delta, L):
    r"""
    Negative energy wavefunction for the right edge (left junction) in the fermionic base.
    The normalization is 1.
    .. math ::
        (c_{r_\uparrow}, c_{r_\downarrow}, c^\dagger_{r_\downarrow}, -c^\dagger_{r_\uparrow})
    """
    return 1/np.sqrt(2)*np.array([0, -1*psi_1_minus(y, kappa, m_0, Delta, L)[0], -1j*psi_1_minus(y, kappa, m_0, Delta, L)[0], 0])

# ax.plot(y, [np.abs(psi_1_plus(y_value, kappa, m_0, Delta, L)) for y_value in y])
# ax.plot(y, [np.abs(psi_1_minus(y_value, kappa, m_0, Delta, L)) for y_value in y])
# ax.plot(y, [np.abs(psi_3_plus(y_value, kappa, m_0, Delta, L)) for y_value in y])
# ax.plot(y, [np.abs(psi_3_minus(y_value, kappa, m_0, Delta, L)) for y_value in y])
# ax.plot(y, [np.abs(psi_2_plus(y_value, kappa, m_0, Delta, L)) for y_value in y])
# ax.plot(y, [np.abs(psi_2_minus(y_value, kappa, m_0, Delta, L)) for y_value in y])
# ax.plot(y, [np.abs(psi_4_plus(y_value, kappa, m_0, Delta, L)) for y_value in y])
# ax.plot(y, [np.abs(psi_4_minus(y_value, kappa, m_0, Delta, L)) for y_value in y])

#%% Sparse diagonalization
from hamiltonians import Hamiltonian_soliton_A1u_sparse
import scipy
from functions import get_components



H = Hamiltonian_soliton_A1u_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi, L=L)
params = {"t": t, "mu": mu, "L_x": L_x, "L_y": L_y, "Delta": Delta, "t_J": t_J, "Phi": Phi, "L": L}
eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H, k=k, sigma=0) 
idx = eigenvalues_sparse.argsort()
eigenvalues_sparse_sorted = eigenvalues_sparse[idx]
eigenvectors_sparse_sorted = eigenvectors_sparse[:, idx]
eigenvalues_sparse = eigenvalues_sparse_sorted
eigenvectors_sparse = eigenvectors_sparse_sorted

index = np.arange(k)   #which zero mode (less than k)
probability_density = []
zero_state = []
localized_state_upper_left = [] # it is the site (L_x/2-1, (L_y+L)/2)
localized_state_upper_right = [] # it is the site (L_x/2, (L_y+L)/2)
localized_state_bottom_left = [] # it is the site (L_x/2-1, (L_y-L)/2)
localized_state_bottom_right = [] # it is the site (L_x/2, (L_y-L)/2)
localized_state_left = []
localized_state_right = []

for i in index:
    destruction_up, destruction_down, creation_down, creation_up = get_components(eigenvectors_sparse[:,i], L_x, L_y)
    probability_density.append((np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)/(np.linalg.norm(np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)))
    zero_state.append(np.stack((destruction_up, destruction_down, creation_down, creation_up), axis=2)) #positive energy eigenvector splitted in components
    localized_state_upper_left.append(zero_state[i][(L_y+L)//2, L_x//2-1,:])
    localized_state_upper_right.append(zero_state[i][(L_y+L)//2, L_x//2,:])
    localized_state_bottom_left.append(zero_state[i][(L_y-L)//2, L_x//2-1,:])
    localized_state_bottom_right.append(zero_state[i][(L_y-L)//2, L_x//2,:])
    localized_state_left.append(zero_state[i][:, L_x//2-1,:])
    localized_state_right.append(zero_state[i][:, L_x//2,:])

index = 0
fig, ax = plt.subplots()
image = ax.imshow(probability_density[index], cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
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
probability_density_at_junction_left = probability_density[index][:, L_x//2-1]/np.sum(probability_density[index][:, L_x//2-1])

fig, ax = plt.subplots()
ax.plot(y , probability_density_at_junction_left, "or", label="Numerical", markersize=4)
probability_density_analytical = np.array([np.linalg.norm(psi_down_minus_right(y_value, kappa, m_0, Delta, L))**2 for y_value in y])
ax.plot(y, probability_density_analytical, label="Analytical")
ax.set_xlabel("y")
ax.set_ylabel("Probability density")
ax.set_title(f"Probability density at the junction with L={L}, Phi={Phi:.2}, alpha={alpha[0]} and Delta={Delta}")
ax.legend()
plt.tight_layout()

#%% Spin determination
from functions import mean_spin_xy
spin = []
for i in range(k):
    spin.append(mean_spin_xy(zero_state[i]))

#%% down minus left
up_plus_left_particle = np.array(localized_state_right[2][:,0])
# Hice un cambio de left por right en la solución numérica
fig, ax = plt.subplots()
y = np.arange(-L_y//2+L//2, L_y//2+L//2)
ax.plot(y, np.real(up_plus_left_particle), "--", label="Real numerical")
ax.plot(y, np.imag(up_plus_left_particle), "--", label="Imaginary numerical")

phase_up_plus_right_particle = 1j*np.exp(1j*(np.angle(up_plus_left_particle[(L_y-L)//2]) + np.pi/2))
up_plus_right_particle_analytical = [phase_up_plus_right_particle/np.sqrt(2)*psi_3_plus(y_value, kappa, m_0, Delta, L) for y_value in y]
ax.plot(y, np.real(up_plus_right_particle_analytical), label="Real analytical")
ax.plot(y, np.imag(up_plus_right_particle_analytical), label="Imaginary analytical")
ax.legend()
plt.title("up_plus_right_particle_analytical")

#%%
up_plus_right_particle = np.array(localized_state_left[2][:,0])
fig, ax = plt.subplots()
y = np.arange(-L_y//2+L//2, L_y//2+L//2)
ax.plot(y, np.real(up_plus_right_particle), "--", label="Real numerical")
ax.plot(y, np.imag(up_plus_right_particle), "--", label="Imaginary numerical")

phase_up_plus_left_particle = np.exp(1j*(np.angle(up_plus_right_particle[(L_y-L)//2]) + np.pi/2))
up_plus_left_particle_analytical = [phase_up_plus_left_particle/np.sqrt(2)*psi_1_plus(y_value, kappa, m_0, Delta, L) for y_value in y]
ax.plot(y, np.real(up_plus_left_particle_analytical), label="Real analytical")
ax.plot(y, np.imag(up_plus_left_particle_analytical), label="Imaginary analytical")
ax.legend()
ax.set_xlabel("y")
plt.title("up_plus_left_particle_analytical")

#%%
up_plus_left_hole = np.array(localized_state_right[2][:,3])
fig, ax = plt.subplots()
y = np.arange(-L_y//2+L//2, L_y//2+L//2)
ax.plot(y, np.real(up_plus_left_hole), "--", label="Real numerical")
ax.plot(y, np.imag(up_plus_left_hole), "--", label="Imaginary numerical")

phase = -1j*phase_up_plus_right_particle
up_plus_right_hole_analytical = [phase/np.sqrt(2)*psi_3_plus(y_value, kappa, m_0, Delta, L) for y_value in y]
ax.plot(y, np.real(up_plus_right_hole_analytical), label="Real analytical")
ax.plot(y, np.imag(up_plus_right_hole_analytical), label="Imaginary analytical")
ax.legend()
ax.set_xlabel("y")
plt.title("up_plus_right_hole_analytical")

#%%
up_plus_right_hole = np.array(localized_state_left[2][:,3])
fig, ax = plt.subplots()
y = np.arange(-L_y//2+L//2, L_y//2+L//2)
ax.plot(y, np.real(up_plus_right_hole), "--", label="Real numerical")
ax.plot(y, np.imag(up_plus_right_hole), "--", label="Imaginary numerical")

phase = 1j*phase_up_plus_left_particle
up_plus_left_hole_analytical = [phase/np.sqrt(2)*psi_1_plus(y_value, kappa, m_0, Delta, L) for y_value in y]
ax.plot(y, np.real(up_plus_left_hole_analytical), label="Real analytical")
ax.plot(y, np.imag(up_plus_left_hole_analytical), label="Imaginary analytical")
ax.legend()
ax.set_xlabel("y")
plt.title("up_plus_left_hole_analytical")

#%%
down_plus_left_particle = np.array(localized_state_right[3][:,1])
# Hice un cambio de left por right en la solución numérica
fig, ax = plt.subplots()
y = np.arange(-L_y//2+L//2, L_y//2+L//2)
ax.plot(y, np.real(down_plus_left_particle), "--", label="Real numerical")
ax.plot(y, np.imag(down_plus_left_particle), "--", label="Imaginary numerical")

phase_down_plus_right_particle = -np.exp(1j*(np.angle(down_plus_left_particle[(L_y-L)//2]) + np.pi/2))
down_plus_right_particle_analytical = [phase_down_plus_right_particle/np.sqrt(2)*psi_4_plus(y_value, kappa, m_0, Delta, L) for y_value in y]
ax.plot(y, np.real(down_plus_right_particle_analytical), label="Real analytical")
ax.plot(y, np.imag(down_plus_right_particle_analytical), label="Imaginary analytical")
ax.legend()
plt.title("down_plus_right_particle_analytical")

#%%
down_plus_left_hole = np.array(localized_state_right[3][:,2])
# Hice un cambio de left por right en la solución numérica
fig, ax = plt.subplots()
y = np.arange(-L_y//2+L//2, L_y//2+L//2)
ax.plot(y, np.real(down_plus_left_hole), "--", label="Real numerical")
ax.plot(y, np.imag(down_plus_left_hole), "--", label="Imaginary numerical")

phase_down_plus_right_hole = -1j*phase_down_plus_right_particle
down_plus_right_hole_analytical = [phase_down_plus_right_hole/np.sqrt(2)*psi_4_plus(y_value, kappa, m_0, Delta, L) for y_value in y]
ax.plot(y, np.real(down_plus_right_hole_analytical), label="Real analytical")
ax.plot(y, np.imag(down_plus_right_hole_analytical), label="Imaginary analytical")
ax.legend()
plt.title("down_plus_right_hole_analytical")