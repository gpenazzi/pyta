"""
Example of homogeneous one-dimensional linear chain without scattering.
We apply Landauer-Caroli formula to calculate transmission
and density of states. Also the surface self-energy is shown.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyta.stats 
import pyta.green
import pyta.core.consts
import sys

#Set energy range
en_points = 100
energies, en_step = np.linspace(-5.0, 5.0, en_points, retstep = True)
#Lenght of chain
n = 10

#Set up Hamiltonian
onsite = 0.0
hopping_element = 1.0
ham = np.matrix(np.zeros((n,n)) * 1j)
for i in range(n-1):
    ham[i,i+1] = hopping_element
    ham[i+1,i] = hopping_element
green_obj = pyta.green.GreenFermion(ham)

#Self energies of semi-infinite ideal chains
ham_pl = np.matrix([onsite])
t_pl = np.matrix([hopping_element])
ham_dl = np.matrix([hopping_element])
pos = 0
left = pyta.green.PhysicalLeadFermion(pos, ham_pl, t_pl, ham_dl)
pos = n-1
right = pyta.green.PhysicalLeadFermion(pos, ham_pl, t_pl, ham_dl)
#Note: chemical potential is 0 by default

#Add contacts to Green solver
leads = [left, right]
green_obj.set_leads(leads)
trans = np.zeros(en_points)
self_real = np.zeros(en_points)
self_imag = np.zeros(en_points)
#Sweep on energy
for ind, en in enumerate(energies):
    green_obj.set_energy(en)
    green = green_obj.get_eqgreen()
    #Caroli-Landauer
    trans[ind] = (np.trace(left.get_gamma(resize=n) * 
                  green_obj.get_eqgreen() * 
                  right.get_gamma(resize=n) * 
                  green_obj.get_eqgreen().H))
    self_real[ind] = left.get_sigma().real[0,0]
    self_imag[ind] = left.get_sigma().imag[0,0]

plt.plot(energies, self_real)
plt.show()

plt.plot(energies, self_imag)
plt.show()
plt.plot(energies, trans)
plt.show()
