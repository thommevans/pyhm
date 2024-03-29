from __future__ import print_function
import numpy as np
import scipy.optimize
from . import Utils
import sys, pdb, warnings

"""
This module defines the optimization algorithms for MAP objects.
"""

# TODO = Add functionality for minimising a vector of weighted residuals (currently only
# have optimisation routines that minimise scalar-values functions). The reason for wanting
# to find the root of vector-valued functions (i.e. those that return a vector of residuals)
# is because nonlinear root-finding algorithms seem to be much faster than nonlinear 
# optimisation algorithms.  

EPSILON = 1e-6

def optimize( MAP, method='neldermead', verbose=False, maxfun=10000, maxiter=10000, ftol=None, xtol=None, gtol=None, epsilon=EPSILON ):
    """
    Wrapper for the scipy optimizers.

    CALLING

      mp = pyhm.MAP( stochastic_dict )
      mp.fit( method='neldermead', maxiter=10000, maxfun=10000, ftol=0.0001, verbose=False )

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
    keys = list( free_stochastics.copy().keys() )

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
        if np.ndim( value )==0:
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
            if ( np.ndim( x_val )==1 )*( len( x_val )==1 ):
                model.stochastics[keys[i]].value = float( x[ixs] )
            else:
                model.stochastics[keys[i]].value = np.array( x[ixs] )
        return -model.logp()

    # Run the optimizer specified in the call:
    if method=='neldermead':
        if ftol is None:
            ftol = 1e-4
        if xtol is None:
            xtol = 1e-4
        #xopt = scipy.optimize.fmin( func, x0, xtol=xtol, ftol=ftol, maxiter=maxiter, maxfun=maxfun, \
        #                            full_output=0, disp=verbose )
        options = { 'maxiter':maxiter, 'maxfev':maxfun, 'xatol':xtol, 'fatol':ftol, 'disp':verbose }
        result = scipy.optimize.minimize( func, x0, method='Nelder-Mead', options=options )
    elif method=='powell':
        if ftol is None:
            ftol = 1e-4
        if xtol is None:
            xtol = 1e-4
        #xopt = scipy.optimize.fmin_powell( func, x0, ftol=ftol, maxiter=maxiter, full_output=0, disp=verbose )
        options = { 'maxiter':maxiter, 'maxfev':maxfun, 'xatol':xtol, 'fatol':ftol, 'disp':verbose }
        result = scipy.optimize.minimize( func, x0, method='Powell', options=options )
    elif method=='conjgrad':
        if ftol is not None:
            print( '' )
            warn_str = '\nConjugate gradient does not accept ftol (ignoring)\n'
            warnings.warn( warn_str )
            pdb.set_trace()
        if gtol is not None:
            gtol = 1e-4 # stop when gradient less than this
        #xopt = scipy.optimize.fmin_cg( func, x0, maxiter=maxiter, full_output=0, disp=verbose, gtol=gtol )
        options = { 'maxiter':maxiter, 'maxfev':maxfun, 'xatol':xtol, 'fatol':ftol, 'disp':verbose }
        result = scipy.optimize.minimize( func, x0, method='CG', options=options )
    else:
        pdb.set_trace() # method not recognised
    xopt = result['x']
    
    # Update the stochastic values to the best-fit values
    # obtained by the optimizer:
    for i in range( nstoch ):
        ixs = ( stochixs==i )
        xopt_i = xopt[ixs]
        if np.ndim( xopt_i )==0:
            free_stochastics[keys[i]].value = float( xopt_i )
        else:
            if len( xopt_i )==1:
                free_stochastics[keys[i]].value = float( xopt_i )
            else:
                free_stochastics[keys[i]].value = np.array( xopt_i )
    # Evaluate the Hessian matrix of logp at the location of the
    # maximum likelihood parameter values:
    MAP.logp_hess = hessian( func, xopt, MAP, epsilon=epsilon )

    # NOTE: Sometimes if the numerical approximation to the Hessian requires
    # evaluating func outside the allowed parameter range, the corresponding
    # column and row will be filled with infs/nans; filter these out so that
    # when the inverse is taken to get the covariance matrix it doesn't
    # become filled with nans; this is temporary, a better solution might be
    # to estimate the Hessian properly with the allowed parameter ranges
    # taken into account....
    ixs = np.arange( nstoch )[np.isfinite( np.diag( MAP.logp_hess ) )==True]
    # Start with matrix of nans
    MAP.pcov = np.nan*np.ones( [ nstoch, nstoch]  )
    try:
        # Attempt to invert the Hessian:
        y = np.linalg.inv( MAP.logp_hess[ixs,:][:,ixs] )
        # Fill in columns/rows that have finite values:
        for i in range( len( ixs ) ):
            for j in range( len( ixs ) ):
                MAP.pcov[ixs[i],ixs[j]] = y[i,j]
    except:
        # If the Hessian is singular, it cannot be inverted:
        warn_str = 'Could not invert Hessian - filling pcov with NaNs'
        warnings.warn( warn_str )

    return None

def hessian( f, x0, MAP, epsilon=EPSILON ):
    """
    Numerically approximate the Hessian matrix using finite differencing.
    Note that the function f is the negative log likelihood.
    """
    keys = MAP.model.stochastics.keys()
    orig_vals = {}
    for key in keys:
        orig_vals[key] = MAP.model.stochastics[key].value
    dim = len( x0 )
    deltaVec = epsilon*np.abs( x0 )
    ixs = ( deltaVec<epsilon )
    deltaVec[ixs] = epsilon
    deltaMat = np.diag( deltaVec )
    deltaVec = np.reshape( deltaVec, [ dim, 1 ] )
    fx = f( x0 )
    Hess = np.zeros( [ dim, dim ] )
    for m in range(dim):
        deltaUse=deltaMat[m,:]
        Hess[m,m] = (-f(x0+2*deltaUse)+16*f(x0+deltaUse)-30*fx+16*f(x0-deltaUse)-f(x0-2*deltaUse))/12.
        for n in range(m+1,dim):
            delta1 = deltaMat[m,:]
            delta2 = deltaMat[n,:]
            Hess[n,m] = (f(x0+delta1+delta2)-f(x0+delta1-delta2)-f(x0-delta1+delta2)+f(x0-delta1-delta2))/4.
            Hess[m,n] = Hess[n,m]
    for key in keys:
        MAP.model.stochastics[key].value = orig_vals[key]
    divid = np.dot(deltaVec,deltaVec.T)
    Hess *= 1./divid
    return Hess

