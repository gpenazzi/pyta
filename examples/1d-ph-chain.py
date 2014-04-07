"""
Example of homogeneous one-dimensional harmonic linear chain without scattering.
Masses can move only along the chain direction (Transverse phonons).
We apply Landauer-Caroli formula to calculate transmission
and density of states. Also the surface self-energy is shown.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyta.stats 
import pyta.green
import pyta.lead
import pyta.consts
import sys

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
left = pyta.lead.PhysicalLeadPhonon(pos, spring_pl, spring_t, spring_dl, 
        temp = 10.0)
pos = n-1
right = pyta.lead.PhysicalLeadPhonon(pos, spring_pl, spring_t, spring_dl,
        temp = 10.0)
#Note: chemical potential is 0 by default

#Add contacts to Green solver
leads = set([left, right])
green_obj.set_leads(leads)

trans = np.zeros(freq_points)
self_real = np.zeros(freq_points)
self_imag = np.zeros(freq_points)
#Sweep on energy
for ind, freq in enumerate(frequencies):
    green_obj.set_freq(freq)
    green = green_obj.get_eqgreen()
    #Caroli-Landauer
    trans[ind] = (np.trace(left.get_gamma(resize = n) * 
                  green_obj.get_eqgreen() * 
                  right.get_gamma(resize = n) * 
                  green_obj.get_eqgreen().H))
    self_real[ind] = left.get_sigma().real[0,0]
    self_imag[ind] = left.get_sigma().imag[0,0]

plt.plot(frequencies, self_real)
plt.show()

plt.plot(frequencies, self_imag)
plt.show()
plt.plot(frequencies, trans)
plt.show()
