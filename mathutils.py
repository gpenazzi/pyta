import numpy as np


def linear_mixer(func1, func2, var1_guess, alpha=1.0,
                 niter=None, maxiter=None, tolerance=1e-2):
    """
    Mix consistently two functions func1 and func2 in the following way:

    var1_0 = initial guess
    var2_0 = func2(var1)
    var1_1 = (1-alpha)var1_0 + alpha*func1(var2_0)
    var2_1 = func2(var2)
    ....
    Until the stopping condition is satisfied

    Args:
        func1 (function): the first function
        func2 (function): the second function
        var1_guess: the first guess to be used as argument to func2
        alpha: mixing parameter. 1.0 correspond to no memory
        niter (int): number of iterations. If niter is reached the function
            exit without error. If no number of iteration is specified then
            only the other stopping condition are used
        maxiter (int): maximum number of iterations. If niter is not specified
            and maxiter is reached without convergence, an exception is thrown

    Returns:
        var1, var2: return values of func1, func2 at stopping condition

        Note: var1_guess is not overwritten

    """

    # n = 0
    var1_n = var1_guess.copy()
    var2_n = func2(var1_n)

    if niter is not None:
        assert type(niter)==int and niter >= 0
        for ii in range(niter):
            var1_nplus1 = (1 - alpha) * var1_n + alpha * func1(var2_n)
            var2_nplus1 = func2(var1_nplus1)
            var1_n = var1_nplus1
            var2_n = var2_nplus1
        return var1_n, var2_n
    else:
        assert type(maxiter)==int and maxiter > 1
        for ii in range(maxiter):
            print('ii',ii)
            var1_nplus1 = (1 - alpha) * var1_n + alpha * func1(var2_n)
            var2_nplus1 = func2(var1_nplus1)
            error = np.amax(np.abs(var1_n - var1_nplus1))
            print('error',error)
            if error < tolerance:
                return var1_n, var2_n
            var1_n = var1_nplus1
            var2_n = var2_nplus1
        raise RuntimeError('Linear Mixer did not converge below tolerance')
