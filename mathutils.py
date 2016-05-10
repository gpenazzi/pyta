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
            var1_nplus1 = (1 - alpha) * var1_n + alpha * func1()
            var2_nplus1 = func2(var1_nplus1)
            error = np.amax(np.absolute(var1_n - var1_nplus1))
            var1_n = var1_nplus1
            var2_n = var2_nplus1
        return var1_n, var2_n
    else:
        assert type(maxiter)==int and maxiter > 1
        for ii in range(maxiter):
            var1_nplus1 = (1 - alpha) * var1_n + alpha * func1()
            var2_nplus1 = func2(var1_nplus1)
            error = np.amax(np.absolute(var1_n - var1_nplus1))
            #print("error: {}".format(error))
            if error < tolerance:
                print('SCBA Iterations: {}'.format(ii))
                return var1_n, var2_n
            var1_n = var1_nplus1
            var2_n = var2_nplus1
        raise RuntimeError('Linear Mixer did not converge below tolerance')


#I use this function to decorate matrices
def resize_matrix(n, pos, mat, dtype=None):
    """
    Resize mat as nxn matrix but include the matrix mat starting from
    position pos

    Args:
        n (integer): target size of nxn matrix
        pos (integer): starting index in target matrix
        mat (mxm numpy.ndarray): matrix to be decorated
        dtype: force an output type

    Returns:
        out (nxn numpy.ndarray): output matrix decorated with zeros
    """
    if dtype is None:
        dtype = mat.dtype
    if mat.shape == (n, n):
        return mat
    else:
        assert(mat.shape[0] == mat.shape[1])
        size = mat.shape[0]
        tmp = np.matrix(np.zeros((n, n)), dtype=dtype)
        tmp[pos:pos + size, pos:pos + size] = mat[:, :]
        return tmp

