import numpy as np
import scipy.linalg
import pdb, os, sys, time
import matplotlib.pyplot as plt

# The discarded points make up the posterior, but need to be weighted by L*w/Z.
# Then perhaps you use that to calculate the weighted mean. Then you can probably
# just to really simple numerical integration away from the mean for each parameter
# to marginalise over +/-34% etc. Need to figure out how this works properly...


a = 2.0
b = 1.0
ndat = 1000
x = np.r_[ -10:10:1j*ndat ]
unc = 1
y = a*x + b + unc*np.random.randn( ndat )

# Generate an active set from the prior.
nactive = 30
a_sample = 1000*( -0.5 + np.random.random( nactive ) )
b_sample = 1000*( -0.5 + np.random.random( nactive ) )
A = np.row_stack( [ a_sample, b_sample ] )

def loglikelihood( a, b ):
    model = a*x + b
    resids = ( y - model ).flatten()
    n = len( resids )
    logp = -n*np.log( unc ) - 0.5*n*np.log( 2*np.pi ) - 0.5*np.sum( ( resids/unc )**2. )
    return logp

def sample( X, logp_min ):
    """
    Given X, where each row is a different input vector, this
    routine will take a random sample from the ellipsoid.
    """

    d, n = np.shape( X )
    mu = np.matrix( np.reshape( np.mean( X, axis=1 ), [ d, 1 ] ) )
    C = np.matrix( np.cov( X ) )
    eigvals, R = np.linalg.eig( C )
    M = np.matrix( np.sqrt( np.diag( eigvals ) ) )

    # calculate the outer products:
    L = np.array( np.linalg.cholesky( C ) )
    k = np.zeros( n )
    for i in range( n ):
        r = np.array( X[:,i] ).flatten() - np.array( mu ).flatten()
        B = scipy.linalg.lu_solve( scipy.linalg.lu_factor( L ), r )
        B = np.reshape( B, [ d, 1 ] )
        k[i] = float( np.matrix( B ).T*np.matrix( B ) )

    f = 1.06
    T = f*np.sqrt( k.max() )*( R.T*M*R )

    logp_new = logp_min - 1
    while logp_new<logp_min:
        w = np.random.randn( d )
        w /= np.sqrt( np.sum( w**2. ) )
        u = np.random.random()
        z = ( u**( 1./d ) )*w
        z = np.matrix( np.reshape( z, [ d, 1 ] ) )
        y = np.array( T*z + mu ) # this is the random sample drawn uniformly from the ellipsoid
        logp_new = loglikelihood( y[0], y[1] )
        
    return y, logp_new


# 1. Draw N samples randomly from the prior over the model's parameter space (the "active set").
# 2. Remove the sample from the active set that has the lowest likelihood L0.
# 3. Add a new random sample to the active set subject to the restriction that its likelihood be L>L0.
# 4. Iterate over steps 2 and 3 until some stopping criterion achieved.
# 5. Compute the evidence by integrating the sorted likelihoods that were discarded at each iteration in step 2.



# Compute the likelihoods of the active set samples.

logL_active =  np.zeros( nactive )
for i in range( nactive ):
    logL_active[i] = loglikelihood( A[0,i], A[1,i] )

nsamples = 5000
w = np.zeros( nsamples )
logL_chain = np.zeros( nsamples )
a_chain = np.zeros( nsamples )
b_chain = np.zeros( nsamples )
X = np.zeros( nsamples )
w = np.zeros( nsamples-1 )
logZ = np.zeros( nsamples-1 )
kappa = 0.1
for i in range( nsamples ):

    # Remove lowest likelihood point from active set.
    ix = np.argmin( logL_active )
    logL_chain[i] = logL_active[ix]
    a_chain[i] = A[0,ix]
    b_chain[i] = A[1,ix]
    
    # Generate a new sample point.
    A = np.column_stack( [ A[:,:ix], A[:,ix+1:] ] )
    logL_active = np.concatenate( [ logL_active[:ix], logL_active[ix+1:] ] )
    Anew, logL_new = sample( A, logL_active[ix] )
    A = np.column_stack( [ A, Anew ] )
    logL_active = np.concatenate( [ logL_active, [ logL_new ] ] )

    X[i] = ( float( nactive )/float( nactive+1 ) )**( i+1 )
    if i>0:
        w[i-1] = X[i-1]-X[i] # currently this is not strictly a correct implementation of the trapezoid rule;
                             # will be a simple adjustment to correct this, including overshoot conditions
                             # suggested in Sivia & Skilling
        logL_max = logL_chain[:i+1].max()
        logL_chain_normalised = logL_chain[:i+1] - logL_max
        L_chain_normalised = np.exp( logL_chain_normalised )
        Z_normalised = 0.5*np.sum( w[:i]*L_chain_normalised[1:i+1] )
        logZ[i-1] = logL_max + Z_normalised #-logL_max + Z_normalised
        print i+1, X[i], logZ[i-1], logZ[i-1]-logZ[i-2]
        if i>100:
            # Stopping criterion:
            if logL_active.max() + np.log( X[i] ) - logZ[i-1] < np.log( kappa ):
                print 'Stopping criterion met - breaking out'
                break
    #if i>1:
    #    LHS = logL_active.max() + np.log( X[i] ) - logZ[i]
    #    RHS = np.log( 1+np.exp( kappa ) )
    #    print i+1, LHS, RHS
    #    if LHS<RHS:
    #        break

