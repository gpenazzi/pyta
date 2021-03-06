"""
Example of homogeneous one-dimensional linear chain without scattering.
We apply Landauer-Caroli formula to calculate transmission
and density of states. Also the surface self-energy is shown.
"""

import numpy as np
import pyta.green
import pyta.lead
import matplotlib.pyplot as plt

#Set energy range
en_points = 200
energies, en_step = np.linspace(-3.00, 3.00, en_points, retstep = True)
#Lenght of chain
n = 10

#Set up Hamiltonian
onsite = 0.0
hopping_element = -1.0
ham = np.matrix(np.zeros((n,n)) * 1j)
for i in range(n-1):
    ham[i,i+1] = hopping_element
    ham[i+1,i] = hopping_element
for i in range(n-2):
    ham[i,i+2] = hopping_element*0.3
    ham[i+2,i] = hopping_element*0.3
green_solver = pyta.green.ElGreen(ham)

#Define leads

ham_pl = ham[0:2,0:2]
t_pl = ham[0:2, 2:4]
ham_dl = ham[0:2, 2:4]
pos = 0
left = pyta.lead.ElLead(pos, ham_pl, t_pl, ham_dl)
left.mu = -2.0
pos = n-2
#Note: this pl, dl ordering is somewhat the opposite of what I thought
right = pyta.lead.ElLead(pos, ham_pl, t_pl.T, ham_dl.T)
right.mu = 2.0

#Add contacts to green solver
leads = [left, right]
green_solver.leads = leads
trans = np.zeros(en_points)
loc_trans = np.zeros(en_points)
self_real = np.zeros(en_points)
self_imag = np.zeros(en_points)
for ind, en in enumerate(energies):
    green_solver.energy = en
    trans[ind] = green_solver.transmission() 
    loc_trans[ind] = np.real(np.sum(green_solver.local_currents[0,:]))
    loc_trans[ind] += np.real(np.sum(green_solver.local_currents[1,:]))
    loc_trans[ind] += np.real(np.sum(green_solver.local_currents[2,:]))
    self_real[ind] = right.sigma_ret[1,1]
    self_imag[ind] = right.sigma_ret[1,1]

plt.plot(energies, self_real)
plt.title('Real part of Self Energy')
plt.show()
plt.plot(energies, self_imag)
plt.title('Imaginary part of Self Energy')
plt.show()
plt.plot(energies, trans, label='Caroli')
plt.plot(energies, loc_trans, label='Energy resolved local flux')
plt.legend()
plt.title('Transmission and local transmissione')
plt.show()
