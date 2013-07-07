import numpy as np

    #TODO: extend to ufunc

def fermi(energy, chempot, temppot = 0.0):
    """Return the value of the Fermi-Dirac distribution
    .. math::
            \frac{1.0}{exp(\frac{E-\mu}{k_{B}T}) + 1.0}
    The temperature must be expressed as energy (kb*T where kb is the Boltzmann
    constant).
    Zero temperature is allowed.
    To avoid exp overflow, if dE/kT > 100, the asymptotic value is returned
    """

    if temppot < 0.:
        raise ValueError("Negative temperature is not allowed")

    #Manage overflow and zero temperature issue
    if (temppot == 0.):
        if energy-chempot > 0.0:
            return 0.0
        if energy-chempot < 0.0:
            return 1.0
        if energy-chempot == 0.0:
            return 0.5
    if (energy-chempot)/temppot > 100.0:
        return 0.0
    if (energy-chempot)/temppot < -100.0:
        return 1.0

    #If no degenerate situation, make the normal calculation
    n = 1.0 / ((np.exp((energy - chemical_potential) / temp_potential) + 1.0))
    assert(n < 1. and n > 0.0 )
    return n

    
def bose(energy, chemical_potential = 0.0, temp_potential = 0.0):
    """Return the value of the Fermi-Dirac distribution
    .. math::
            \frac{1.0}{exp(\frac{E-\mu}{k_{B}T}) - 1.0}
    The temperature must be expressed as energy (kb*T where kb is the Boltzmann
    constant).
    Zero temperature is not allowed.
    To avoid exp overflow, if dE/kT > 100, the asymptotic value is returned
    """
    
    if temppot <= 0.0:
        raise ValueError("Negative and zero temperature is not allowed")
    if energy-chempot <= 0.0:
        raise ValueError("Negative (energy-chempot)  is not allowed")

    #Avoid overflow issue
    if (energy-chempot)/temppot > 100.0:
        return 0.0
    
    n = 1.0 / ((np.exp((energy - chemical_potential) / temp_potential) - 1.0))
    assert(n > 0.0 )
    return n
