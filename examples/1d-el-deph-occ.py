import numpy as np
import matplotlib.pyplot as plt
import pyta.green
import pyta.lead
import pyta.consts

"""Here we set a non homogeneous linear chain and we check the occupation
in the first and last site respect to the occupation in the reservoirs.

Switching on the dephasing lead to a variable occupation along the chain,
which drops according to the dephasing strength, the length of the chain and the
applied bias.
"""

#Set an energy range
en_points = 400
en, step = np.linspace(-0.2, 1.5, en_points, retstep = True)
#System length
n = 200
#Maximum number of Self Consistent Born Approximation steps and SCBA tolerance
scba_steps = 1000
scba_tol = 1e-5

#Device Hamiltonian and coupling parameters
ham = np.matrix(np.zeros((n,n)) * 1j)
hopping_element = 1.0
onsite = 1.0
for i in range(n-1):
    ham[i,i+1] = hopping_element
    ham[i+1,i] = hopping_element
    ham[i,i] = onsite
ham[n-1, n-1] = onsite
coupling_left = 1.0 #coupling with self-energy
coupling_right = 1.0
left_onsite = 0.00
right_onsite = 0.00
####################################

#Output variables
occ_l = np.zeros(en_points)
occ_r = np.zeros(en_points)
occ_l_lead = np.zeros(en_points)
occ_l2 = np.zeros(en_points)
occ_r2 = np.zeros(en_points)
occ_r_lead = np.zeros(en_points)
#####################################

# Declare left lead 
pl_ham_l = np.matrix([onsite])
pl_t = np.matrix([coupling_left])
pl_ld = np.matrix([coupling_left])
left = pyta.lead.ElLead(0, pl_ham_l, pl_t, pl_ld)
t = 100.0 * pyta.consts.kb_eV__K
left.set('temperature', t)
left.set('mu', 0.0)
####################

#Declare right lead
pl_ham_r = np.matrix([onsite])
pl_t = np.matrix([coupling_right])
pl_ld = np.matrix([coupling_right])
right = pyta.lead.ElLead(n-1, pl_ham_r, pl_t, pl_ld)
t = 100.0 * pyta.consts.kb_eV__K
right.set('temperature', t)
right.set('mu', 1.0)
###################

#Declare virtual lead
dephase_parameter = np.array(np.zeros(n))
dephase_parameter[0:n] = 1e-2
dephasing = pyta.lead.MRDephasing()
dephasing.set('coupling', dephase_parameter)
######################

# Declare Green's solver
green_obj = pyta.green.ElGreen(ham)
green_obj.set('leads', [left, right, dephasing])
####################################

# Energy loop
for ind, ener in enumerate(en):
    green_obj.set('energy', ener)
    #SCBA loop
    print('eq')
    green_obj.scba(dephasing,mode='equilibrium',maxiter=10, tolerance=1e-5)
    print('keldysh')
    green_obj.scba(dephasing,mode='keldysh',maxiter=10,tolerance=1e-3)
    #green_obj.set('leads', [left, right])
    #scba = pyta.green.SCBA(green_obj, dephasing, tol = scba_tol, maxiter=scba_steps, task='both')
    #scba.solve()

    #Occupation is determined by comparing the Non equilibrium Green's function
    #and the spectral density
    occ = green_obj.get('occupation')
    occl = left.get('occupation')
    occr = right.get('occupation')
    occ_l[ind] = occ[0,0]
    occ_r[ind] = occ[n-1,n-1]
    occ_l2[ind] = occ[50,50]
    occ_r2[ind] = occ[150,150]
    occ_l_lead[ind] = occl[0,0]
    occ_r_lead[ind] = occr[0,0]

#Plot the results
plt.figure()
plt.plot(en, occ_l, ':',  label='0 site')
plt.plot(en, occ_r, ':', label='200 site')
plt.plot(en, occ_l2, ':',  label='50 site')
plt.plot(en, occ_r2, ':', label='150 site')
plt.plot(en, occ_l_lead, label='Left contact')
plt.plot(en, occ_r_lead, label='Right contact')
legend = plt.legend(loc='upper right', shadow=True)
plt.show()
