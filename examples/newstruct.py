"""
Example of homogeneous one-dimensional linear chain without scattering.
We apply Landauer-Caroli formula to calculate transmission
and density of states. Also the surface self-energy is shown.
"""

import numpy as np
import pyta.green

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
green_solver = pyta.green.GreenFermion(ham)
green_solver.set('energy', 10.0)

print('done')
gr=green_solver.get('eqgreen')
print('eqgreen',gr)

#Works till here

ham_pl = np.matrix([onsite])
t_pl = np.matrix([hopping_element])
ham_dl = np.matrix([hopping_element])
pos = 0
left = pyta.lead.PhysicalLeadFermion(pos, ham_pl, t_pl, ham_dl)

