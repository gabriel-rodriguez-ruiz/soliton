# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 14:42:34 2023

@author: gabri
"""
import numpy as np
import matplotlib.pyplot as plt

L = 10
t_J = 0.1
Delta = 1
zeta = t_J/Delta
E = 0.2  # si E+t_J/Delta>0

def Psi(y, i):
    """Wavefunction with index i for a specific range of energy."""
    chi_values = [(E+t_J)/Delta,  (E-t_J)/Delta, -(E+t_J)/Delta, -(E-t_J)/Delta]
    chi = chi_values[i]
    if y<0:
        return np.exp(chi*y)
    elif y>=0 and y<=L:
        #return ((1-np.exp(zeta*L))/(1-np.exp(2*zeta*L))*(np.exp(zeta*y)
        #            +np.exp(-zeta*(y-L))))
        #return 1/( np.exp(E/Delta*L)*np.sinh(t_J/Delta*L) )* \
         #       ( np.exp(E/Delta*y)*np.sinh(t_J/Delta*y) - \
          #       np.exp(E/Delta*(y+L))*np.sinh(t_J/Delta*(y-L)) )
        return (1-np.exp(chi*L))/(1-np.exp(2*chi*L)) * (np.exp(chi*y)+np.exp(-chi*(y-L)))
    else:
        return np.exp(-chi*(y-L))

y_values = np.linspace(L/2-2*L, L/2+2*L, 100)
fig, ax = plt.subplots()
for i in range(4):
    ax.plot(y_values, [Psi(y, i) for y in y_values])
ax.set_ylim([0,1])
ax.set_ylabel(r"$\Psi$")
ax.set_xlabel("y")
ax.text(0, 0.5, rf"L={L}, $t_J$={t_J}, $\Delta$={Delta}, E={E}")        