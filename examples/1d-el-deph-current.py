import numpy as np
import matplotlib.pyplot as plt
import pyta.green
import pyta.lead
import pyta.consts

"""Here we set a non homogeneous linear chain and we calculate the current
by calling the Meir Wingreen formula from Green module.

For 0 dephasing, we should recover the ballistic current while for a finite dephasing
strenght the current is lower and should go down linearly with the chain length (diffusive limit)
"""

#Set an energy range
en_points = 500
en, step = np.linspace(-2.0, 2.0, en_points, retstep = True)
#System length
n = 100
#Maximum number of Self Consistent Born Approximation steps and SCBA tolerance
scba_steps = 1000
scba_tol = 1e-8

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
dephase_parameter[0:n] = 1e-10
dephasing = pyta.lead.MRDephasing()
dephasing.set('coupling', dephase_parameter)
######################

# Declare Green's solver
green_obj = pyta.green.ElGreen(ham)
green_obj.set('leads', [left, right])
####################################

#Assign green to virtual lead
dephasing.set('greensolver', green_obj)
curr = np.zeros(en_points)
total_current = 0.0

# Energy loop
for ind, ener in enumerate(en):
    green_obj.set('energy', ener)
    #SCBA loop
    green_obj.set('leads', [left, right])
    scba = pyta.green.SCBA(green_obj, dephasing, tol = scba_tol, maxiter=scba_steps, task='both')
    scba.do()
    curr[ind] = green_obj.get('meirwingreen', lead=left)
    total_current += curr[ind] * step

#Plot the results
print('total current',total_current)
plt.figure()
plt.plot(en, curr)
plt.title('Energy resolved current')
plt.show()
