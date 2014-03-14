import numpy as np
import matplotlib.pyplot as plt
import pyta.stats as stats
import pyta.green
import pyta.core.consts

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
scba_steps = range(1000)
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
left = pyta.green.PhysicalLeadFermion(0, pl_ham_l, pl_t, pl_ld, mu =
        0.0)
t_left = 100.0 * pyta.core.consts.kb_eV__K
left.set_temp(t_left)
####################

#Declare right lead
pl_ham_r = np.matrix([onsite])
pl_t = np.matrix([coupling_right])
pl_ld = np.matrix([coupling_right])
right = pyta.green.PhysicalLeadFermion(n-1, pl_ham_r, pl_t, pl_ld, mu =
        1.0)
###################

#Declare virtual lead
dephase_parameter = np.array(np.zeros(n))
dephase_parameter[0:n] = 1e-2
dephasing = pyta.green.MRDephasing(dephase_parameter)
t_right = 100.0 * pyta.core.consts.kb_eV__K
right.set_temp(t_right)
######################

# Declare Green's solver
green_obj = pyta.green.GreenFermion(ham)
leads = set([left, right])
green_obj.set_leads(leads)
####################################


# Energy loop
for ind, ener in enumerate(en):
    green_obj.set_energy(ener)
    green = green_obj.get_eqgreen()

    #SCBA loop
    scba = pyta.green.SCBA(green_obj, dephasing)
    scba.do()

    #Occupation is determined by comparing the Non equilibrium Green's function
    #and the spectral density
    occ_l[ind] = (np.imag(green_obj.get_green_lr()[0,0]) /
                         green_obj.get_spectral()[0,0])
    occ_r[ind] = (np.imag(green_obj.get_green_lr()[n-1,n-1]) /
                         green_obj.get_spectral()[n-1,n-1])
    occ_l2[ind] = (np.imag(green_obj.get_green_lr()[50,50]) /
                         green_obj.get_spectral()[50,50])
    occ_r2[ind] = (np.imag(green_obj.get_green_lr()[150,150]) /
                         green_obj.get_spectral()[150,150])
    occ_l_lead[ind] = (np.imag(left.get_sigma_lr()[0,0]) /
                         left.get_gamma()[0,0])
    occ_r_lead[ind] = (np.imag(right.get_sigma_lr()[0,0]) /
                         right.get_gamma()[0,0])

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
