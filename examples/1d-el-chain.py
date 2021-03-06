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
en_points = 100
energies, en_step = np.linspace(-3.0, 3.0, en_points, retstep = True)
#Lenght of chain
n = 10

#Set up Hamiltonian
onsite = 0.0
hopping_element = 1.0
ham = np.zeros((n,n)) * 1j
for i in range(n-1):
    ham[i,i+1] = hopping_element
    ham[i+1,i] = hopping_element
green_solver = pyta.green.ElGreen(ham)

#Define leads

ham_pl = np.ndarray(shape=(1,1))
ham_pl[:,:] = onsite
t_pl = np.ndarray(shape=(1,1))
t_pl[:,:] = hopping_element
ham_dl = np.ndarray(shape=(1,1))
ham_dl[:,:] = hopping_element
pos = 0
left = pyta.lead.ElLead(pos, ham_pl, t_pl, ham_dl)
pos = n-1
right = pyta.lead.ElLead(pos, ham_pl, t_pl, ham_dl)

#Add contacts to green solver
leads = [left, right]
green_solver.leads = leads
trans = np.zeros(en_points)
self_real = np.zeros(en_points)
self_imag = np.zeros(en_points)
for ind, en in enumerate(energies):
    green_solver.energy = en
    trans[ind] = green_solver.transmission()
    self_real[ind] = left.sigma_ret.real[0,0]
    self_imag[ind] = left.sigma_ret.imag[0,0]

print('transmission', trans)

plt.plot(energies, self_real)
plt.title('Real part of Self Energy')
plt.show()
plt.plot(energies, self_imag)
plt.title('Imaginary part of Self Energy')
plt.show()
plt.plot(energies, trans)
plt.title('Transmission')
plt.show()
