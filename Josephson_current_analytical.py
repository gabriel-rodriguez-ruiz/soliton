#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 18:21:19 2023

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt

Phi = 0.1*np.pi  #height of the phase soliton around flux pi
t_J = 1   #t/2
m_0 = t_J*np.sin(Phi/2)
L = 30      #L_y//2
Delta = 1

def Josephson_current(Phi, t_J, Delta, L):
    """ Analytical Josephson current in units of 2e/h
    """
    m_0 = t_J*np.sin(Phi/2)
    return t_J/4*np.cos(Phi/2) * ( m_0*L/Delta - np.exp(-m_0*L/Delta) )

fig, ax = plt.subplots()
phi = np.linspace(0, 2*np.pi, 1000)
ax.plot(phi, [Josephson_current(phi, t_J, Delta, L) for phi in phi])
plt.title("Josephson current with a soliton")
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$J(\phi)$")