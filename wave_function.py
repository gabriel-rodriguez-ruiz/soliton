# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 14:42:34 2023

@author: gabri
"""
import numpy as np
import matplotlib.pyplot as plt

L = 10
t_J = 1
Delta = 1
zeta = t_J/Delta

def Psi(y):
    if y<0:
        return np.exp(zeta*y)
    elif y>=0 and y<=L:
        return ((1-np.exp(zeta*L))/(1-np.exp(2*zeta*L))*(np.exp(zeta*y)
                    +np.exp(-zeta*(y-L))))
    else:
        return np.exp(-zeta*(y-L))

y_values = np.linspace(-2*L,3*L, 100)
fig, ax = plt.subplots()
ax.plot(y_values, [Psi(y) for y in y_values])
        