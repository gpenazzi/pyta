import numpy as np
import matplotlib.pyplot as plt
import pyta.green
import pyta.lead
import pyta.core.consts

"""Here we set a non homogeneous linear chain and we check the occupation
in the first and last site respect to the occupation in the reservoirs.

Switching on the dephasing lead to a variable occupation along the chain,
which drops according to the dephasing strength, the length of the chain and the
applied bias.
"""

def noise(amplitude = 0.06):
    return (np.random.sample()*amplitude - amplitude/2.0)

def single_calc(n, deph):

    #Set an energy range
    en_points = 400
    en, step = np.linspace(-0.1, 0.1, en_points, retstep = True)
    #System length
    #n = 10
    #Maximum number of Self Consistent Born Approximation steps and SCBA tolerance
    scba_steps = range(10000)
    scba_tol = 0.01
    
    #Device Hamiltonian and coupling parameters
    #We add some noise to the off-diagonal
    ham = np.matrix(np.zeros((n,n)) * 1j)
    hopping_element = 0.06
    onsite = 0.0
    for i in range(n-1):
        added_noise = noise(amplitude=0.06)
        onsite_noise = noise(amplitude=0.5)
        ham[i,i+1] = hopping_element + added_noise 
        ham[i+1,i] = hopping_element + added_noise
        ham[i,i] = onsite + onsite_noise 
    ham[n-1, n-1] = onsite
    coupling_left = 0.1 #coupling with self-energy
    coupling_right = 0.1
    ####################################
    
    
    #Output variables
    current = np.zeros(en_points)
    #####################################
    
    # Declare left lead 
    #pl_ham_l = np.matrix([onsite])
    #pl_t = np.matrix([coupling_left])
    pl_ld = np.matrix([coupling_left])
    left = pyta.lead.WideBandFermion(0, 0.5, pl_ld, mu = -0.05)
    t_left = 300.0 * pyta.core.consts.kb_eV__K
    left.set_temp(t_left)
    ####################
    
    #Declare right lead
    #pl_ham_r = np.matrix([onsite])
    #pl_t = np.matrix([coupling_right])
    pl_ld = np.matrix([coupling_right])
    right = pyta.green.WideBandFermion(n-1, 0.5, pl_ld, mu = 0.05)
    ###################
    
    #Declare virtual lead
    dephase_parameter = np.array(np.zeros(n))
    dephase_parameter[0:n] = deph
    dephasing = pyta.green.MRDephasing(dephase_parameter)
    t_right = 300.0 * pyta.core.consts.kb_eV__K
    right.set_temp(t_right)
    ######################
    
    # Declare Green's solver
    green_obj = pyta.green.GreenFermion(ham)
    leads = [left, right]
    green_obj.set_leads(leads)
    ####################################
    
    
    # Energy loop
    mix = 0.05
    for ind, ener in enumerate(en):
        green_obj.set_energy(ener)
        green = green_obj.get_eqgreen()

        scba = pyta.green.SCBA(green_obj, dephasing, task='both')
        scba.do()

        #Occupation is determined by comparing the Non equilibrium Green's function
        #and the spectral density
        current[ind] = np.trace((left.get_sigma_lr(resize = n)*green_obj.get_green_gr() -
                left.get_sigma_gr(resize = n)*green_obj.get_green_lr()))
    
    tot_current = np.sum(current)*step
    conductance=-tot_current/0.1
    print('n',n)
    print('conductance',conductance)
    return conductance 
    #Plot the results
    #plt.figure()
    #plt.plot(en, current)
    #legend = plt.legend(loc='current', shadow=True)
    #plt.show()

sizes = range(11,21)
repetitions = 50
conds = np.zeros(repetitions)
for size in sizes:
    for i in range(repetitions):
        conds[i] = single_calc(size, 0.0001)
    filename = 'deph10meVscba_' + str(size) + '.txt'
    np.savetxt(filename, conds)

#print('conds', conds)
#plt.figure()
#plt.plot(sizes, conds)
#legend = plt.legend(loc='conductances', shadow=True)
#plt.show()
