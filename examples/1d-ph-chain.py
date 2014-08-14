"""
Example of homogeneous one-dimensional harmonic linear chain without scattering.
Masses can move only along the chain direction (Transverse phonons).
We apply Landauer-Caroli formula to calculate transmission
and density of states. Also the surface self-energy is shown.
"""

import numpy as np
import pyta.green
import pyta.lead
import matplotlib.pyplot as plt

#Set energy range
freq_points = 100
frequencies, freq_step = np.linspace(0.001, 5.0, freq_points, retstep = True)
#Lenght of chain
n = 10

#Set up Spring constants (mass are assumed unitary)
spring_constant = 1.0
spring = np.matrix(np.zeros((n,n)) * 1j)
for i in range(n-1):
    spring[i,i+1] = spring_constant
    spring[i+1,i] = spring_constant
green_obj = pyta.green.GreenPhonon(spring)

#Self energies of semi-infinite ideal chains
spring_pl = np.matrix([0.0])
spring_t = np.matrix([spring_constant])
spring_dl = np.matrix([spring_constant])
pos = 0
left = pyta.lead.PhysicalLeadPhonon(pos, spring_pl, spring_t, spring_dl)
left.set('temperature', 10.0)
pos = n-1
right = pyta.lead.PhysicalLeadPhonon(pos, spring_pl, spring_t, spring_dl)
right.set('temperature', 10.0)
#Note: chemical potential is 0 by default

#Add contacts to Green solver
leads = [left, right]
green_obj.set('leads', leads)
trans = np.zeros(freq_points)
self_real = np.zeros(freq_points)
self_imag = np.zeros(freq_points)
#Sweep on energy
for ind, freq in enumerate(frequencies):
    green_obj.set('frequency', freq)
    trans[ind] = green_obj.get('transmission')
    self_real[ind] = left.get('sigma_ret').real[0,0]
    self_imag[ind] = left.get('sigma_ret').imag[0,0]

plt.plot(frequencies, self_real)
plt.title('Real part of Self Energy')
plt.show()

plt.plot(frequencies, self_imag)
plt.title('Imaginary part of Self Energy')
plt.show()
plt.plot(frequencies, trans)
plt.title('Transmission')
plt.show()
