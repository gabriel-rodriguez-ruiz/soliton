#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 15:44:08 2023

@author: gabriel
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root


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

