import numpy as np
import scipy.optimize
import Utils
import sys, pdb, warnings
try:
    import numdifftools as nd
    nd_installed = True
except:
    nd_installed = False

"""
This module defines the optimization algorithms for MAP objects.
"""

def optimize( MAP, method='neldermead', maxiter=1000, ftol=None, verbose=False ):
    """
    Wrapper for the scipy optimizers.

    CALLING

      mp = pyhm.MAP( stochastic_dict )
      mp.fit( method='neldermead', maxiter=1000, ftol=0.0001, verbose=False )

    DESCRIPTION
      This routine takes pyhm MAP objects and prepares them so that the negative
      model log likelihood can be optimized using the standard scipy optimization
      routines. Currently, the scipy optimizers that are supported are:
        neldermead (fmin)
        powell (fmin_powell)
        conjgrad (fmin_cg)
     """
    
    model = MAP.model
    free_stochastics = Utils.unobserved_stochastics( model.stochastics )
    keys = free_stochastics.keys()

    # Go through each of the stochastics, and unpack all of
    # their values into a single array; as part of this, fill
    # a separate array of integers that index each element of
    # the value array as belonging to a specific stochastic;
    # this is important for stochastics that have n-array values:
    nstoch = len( keys )
    x0 = np.array( [] )
    stochixs = np.array( [] )
    for i in range( nstoch ):
        value = free_stochastics[keys[i]].value
        if np.rank( value )==0:
            x0 = np.concatenate( [ x0, [ value ] ] )
            stochixs = np.concatenate( [ stochixs, [ i ] ] )
        else:
            ndim = len( value.flatten() )
            x0 = np.concatenate( [ x0, value ] )
            stochixs = np.concatenate( [ stochixs, i*np.ones( ndim ) ] )
    x0 = np.array( x0 )
    stochixs = np.array( stochixs, dtype=int )

    # Define the negative log likelihood in the format
    # that the scipy optimizers require:
    def func( x, *args ):

        # Update the values of each stochastic using the
        # values provided in the input array x, making
        # use of the stochixs array (created above) to
        # map each element or group of elements back to
        # a specific stochastic:
        for i in range( nstoch ):
            ixs = ( stochixs==i )
            x_val = x[ixs]
            if ( np.rank( x_val )==1 )*( len( x_val )==1 ):
                model.stochastics[keys[i]].value = float( x[ixs] )
            else:
                model.stochastics[keys[i]].value = np.array( x[ixs] )
        return -model.logp()

    # Run the optimizer specified in the call:
    if method=='neldermead':
        if ftol==None:
            ftol = 0.01
        xopt = scipy.optimize.fmin( func, x0, ftol=ftol, maxiter=maxiter, full_output=0, disp=verbose )
    elif method=='powell':
        if ftol==None:
            ftol = 0.01
        xopt = scipy.optimize.fmin_powell( func, x0, ftol=ftol, maxiter=maxiter, full_output=0, disp=verbose )
    elif method=='conjgrad':
        if ftol!=None:
            print ''
            warn_str = '\nConjugate gradient does not accept ftol (ignoring)\n'
            warnings.warn( warn_str )
        xopt = scipy.optimize.fmin_cg( func, x0, maxiter=maxiter, full_output=0, disp=verbose )

    # Update the stochastic values to the best-fit values
    # obtained by the optimizer:
    for i in range( nstoch ):
        ixs = ( stochixs==i )
        xopt_i = xopt[ixs]
        if np.rank( xopt_i )==0:
            free_stochastics[keys[i]].value = float( xopt_i )
        else:
            if len( xopt_i )==1:
                free_stochastics[keys[i]].value = float( xopt_i )
            else:
                free_stochastics[keys[i]].value = np.array( xopt_i )                

    MAP.logp_hess = -hessian( func, xopt )
    #xdata = np.reshape(np.arange(0,1,0.1),(-1,1))
    #ydata = 1+2*np.exp(0.75*xdata)
    #fun = lambda c: (c[0]+c[1]*np.exp(c[2]*xdata) - ydata)**2
    #xx=np.array([1,2,0.75])
    #AA=hessian(fun,xx)
    #aa=AA(xx)
    #BB=nd.Jacobian(fun)
    #bb=BB(xx)
    #print aa-bb
    #print 'aaaaaa'
    #pdb.set_trace()
    try:
        MAP.pcov = np.linalg.inv( -MAP.logp_hess )
    except:
        MAP.pcov = np.inf
    return None


def hessian ( f, x0, epsilon=1.e-8, linear_approx=False ):
    """
    A numerical approximation to the Hessian matrix of cost function at
    location x0 (hopefully, the minimum). THIS NEEDS TO BE CHECKED!!!!
    """
    # ``calculate_cost_function`` is the cost function implementation
    # The next line calculates an approximation to the first
    # derivative
    f1 = scipy.optimize.approx_fprime( x0, f, epsilon )

    # This is a linear approximation. Obviously much more efficient
    # if cost function is linear
    if linear_approx:
        f1 = np.matrix(f1)
        return f1.transpose() * f1
    # Allocate space for the hessian
    n = x0.shape[0]
    hessian = np.zeros ( ( n, n ) )
    # The next loop fill in the matrix
    xx = x0
    for j in xrange( n ):
        xx0 = xx[j] # Store old value
        xx[j] = xx0 + epsilon # Perturb with finite difference
        # Recalculate the partial derivatives for this new point
        f2 = scipy.optimize.approx_fprime( x0, f, epsilon )
        hessian[:, j] = (f2 - f1)/epsilon # scale...
        xx[j] = xx0 # Restore initial value of x0
    return hessian
