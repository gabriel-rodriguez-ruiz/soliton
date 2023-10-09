import numpy as np
import matplotlib.pyplot as plt
import scipy
from phase_functions import phase_soliton_antisoliton
from hamiltonians import Hamiltonian_A1us_junction_sparse_periodic_in_x_and_y

L_x = 200
L_y = 200
t = 1
Delta = t/2
Delta_0 = t/10
mu = -2*t  #-2
phi_external = 0.
t_J = t/10    #t/2
k = 12
y = np.arange(1, L_y+1)
L_values = np.linspace(10, 90, 9, dtype=int)

eigenvalues = []

for L_value in L_values:
    y_0 = (L_y-L_value)//2
    y_1 = (L_y+L_value)//2
    Phi = phase_soliton_antisoliton(phi_external, y, y_0, y_1)
    # Phi = phase_single_soliton(phi_external, y, y_s)
    # H = Hamiltonian_A1us_junction_sparse_periodic(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, Delta_0=Delta_0, t_J=t_J, Phi=Phi)
    # H = Hamiltonian_A1us_junction_sparse(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, Delta_0=Delta_0, t_J=t_J, Phi=Phi)
    H = Hamiltonian_A1us_junction_sparse_periodic_in_x_and_y(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, Delta_0=Delta_0, t_J=t_J, Phi=Phi)
    eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H, k=k, sigma=0) 
    eigenvalues_sparse.sort()
    eigenvalues.append(eigenvalues_sparse)

index = np.arange(k)
E_numerical = []
for j in index:
    E_numerical.append(np.array([eigenvalues[i][j] for i in range(len(L_values))]))

#%% Plotting

plt.rc("font", family="serif")  # set font family
plt.rc("xtick", labelsize="large")  # reduced tick label size
plt.rc("ytick", labelsize="large")
plt.rc('font', size=18) #controls default text size
plt.rc('axes', titlesize=18) #fontsize of the title
plt.rc('axes', labelsize=18) #fontsize of the x and y labels
plt.rc("text", usetex=True) # for better LaTex (slower)
plt.rcParams['xtick.top'] = True    #ticks on top
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['ytick.right'] = True    #ticks on left
plt.rcParams['ytick.labelright'] = False
plt.rc('legend', fontsize=18) #fontsize of the legend


fig, ax = plt.subplots()
# I remove the L=100 distance and plot only zero-energy states
ax.plot(L_values, E_numerical[6], "o", label="Numerical")

ax.set_xlabel(r"$L$")
plt.yscale('log')
ax.set_ylabel("E")

from analytical_solution import Kappa

m_0 = t_J/2

def positive_energy(L, m_0):
    kappa_value = Kappa(m_0=m_0, Delta=Delta, L=L_value)
    return m_0*np.exp(-kappa_value*L)

E = []
for L_value in L_values:
    Energy = positive_energy(L=L_value, m_0=m_0)
    E.append(Energy[0])

E_analytical = np.array([E[i] for i in range(len(L_values))])
ax.plot(L_values, E_analytical, "ok", label="Analytical")

m_numerical, b_numerical = np.polyfit(L_values, np.log(E_numerical[6]), 1)
m_analytical, b_analytical = np.polyfit(L_values, np.log(E_analytical), 1)

ax.plot(L_values, np.exp(m_numerical*L_values + b_numerical), label=f"{m_numerical:.3}L{b_numerical:.3}")
ax.plot(L_values, np.exp(m_analytical*L_values + b_analytical), label=f"{m_analytical:.3}L{b_analytical:.3}")
ax.legend()
plt.title(r"$\phi_{ext}=$"+f"{phi_external:.2}, Delta={Delta}")
plt.tight_layout()