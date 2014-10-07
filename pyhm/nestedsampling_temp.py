from bayes.pyhm_dev import pyhm
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import pdb, os, sys

##########################################################################################

# Generate a toy dataset with linear trend:
a = 2.0
b = 1.0
ndat = 1000
x = np.r_[ -10:10:1j*ndat ]
unc = 1
y = a*x + b + unc*np.random.randn( ndat )


def logL( a, b ):
    """
    Given parameter values a and b, calculates
    the log likelihood.
    """
    model = a*x + b
    resids = ( y - model ).flatten()
    n = len( resids )
    return -n*np.log( unc ) - 0.5*n*np.log( 2*np.pi ) - 0.5*np.sum( ( resids/unc )**2. )

def sample_ellipsoid( V, logp_min ):
    """
    Given V, where each row is a different input vector, this
    routine will take a random sample from the ellipsoid.
    """

    d, n = np.shape( V )
    mu = np.matrix( np.reshape( np.mean( V, axis=1 ), [ d, 1 ] ) )
    C = np.matrix( np.cov( V ) )
    eigvals, R = np.linalg.eig( C )
    M = np.matrix( np.sqrt( np.diag( eigvals ) ) )

    # calculate the outer products:
    L = np.array( np.linalg.cholesky( C ) )
    k = np.zeros( n )
    for i in range( n ):
        r = np.array( V[:,i] ).flatten() - np.array( mu ).flatten()
        B = scipy.linalg.lu_solve( scipy.linalg.lu_factor( L ), r )
        B = np.reshape( B, [ d, 1 ] )
        k[i] = float( np.matrix( B ).T*np.matrix( B ) )

    f = 1.06 # ellipsoid expansion factor
    T = f*np.sqrt( k.max() )*( R.T*M*R )

    logp_new = logp_min - 1
    while logp_new<logp_min:
        w = np.random.randn( d )
        w /= np.sqrt( np.sum( w**2. ) )
        u = np.random.random()
        z = ( u**( 1./d ) )*w
        z = np.matrix( np.reshape( z, [ d, 1 ] ) )
        y = np.array( T*z + mu ).flatten() # this is the random sample drawn uniformly from the ellipsoid
        logp_new = logL( y[0], y[1] )

    return y, logp_new

########################################################################
# Sampling.

# Define the sampling parameters:
nactive = 100 # number of samples to generate (i.e. number of points in the active sample)
npar = 2 # dimensionality of the parameter space (i.e. the 'a' and 'b' parameters)

# Generate the active set:
a_active = 5*( -0.5 + np.random.random( nactive ) )
b_active = 5*( -0.5 + np.random.random( nactive ) )
A_active = np.row_stack( [ a_active, b_active ] )

# Evaluate the log likelihoods of the active set points:
logL_active = np.empty( nactive )
for i in range( nactive ):
    logL_active[i] = logL( A_active[0,i], A_active[1,i] )

def log_add( loga, logc ):
    """
    Given log(a) and log(c), returns log(a+c).
    """
    if loga>logc:
        return loga + np.log( 1+np.exp( logc-loga ) )
    else:
        return logc + np.log( 1+np.exp( loga-logc ) )
    
logL_chain = []
A_chain = []
X_chain = []
w_chain = []
logZ = -sys.float_info.max
i = 0
fstop = 0.05
terminate = False
while terminate==False:
    ix = np.argmin( logL_active )
    logL_chain += [ logL_active[ix] ]
    A_chain += [ A_active[:,ix] ]
    X_chain += [ np.exp( -float( i )/float( nactive ) ) ]
    if i>1:
        w_chain += [ 0.5*( X_chain[i-2] - X_chain[i] ) ]
        delta_logZ = logL_chain[i-1] + np.log( w_chain[-1] )
        logZ = log_add( logZ, delta_logZ )

        # Stopping criterion:
        if ( logL_active.max() + np.log( X_chain[i] ) - logZ < np.log( fstop ) ):
            terminate = True
        
    i += 1
    # Replace the lowest likelihood point from the active
    # with a new draw from the ellipsoid:
    A_active = np.column_stack( [ A_active[:,:ix], A_active[:,ix:] ] )
    logL_i = logL_active[ix]
    A_new, logL_new = sample_ellipsoid( A_active, logL_i )
    A_active[:,ix] = A_new
    logL_active[ix] = logL_new
    
logL_chain = np.array( logL_chain )
A_chain = np.column_stack( A_chain ).T
X_chain = np.array( X_chain )
w_chain = np.array( w_chain )

nsamples = len( logL_chain )
log_percentiles = np.zeros( [ nsamples-2, npar ] )
for i in range( npar ):
    ixs = np.argsort( A_chain[1:-1,i] )
    running_log_count = -sys.float_info.max
    for j in range( nsamples-2 ):
        print i, j
        delta_log_count = logL_chain[1:-1][ixs][j] + np.log( w_chain[ixs][j] )
        running_log_count = log_add( running_log_count, delta_log_count )
        log_percentiles[j,i] = running_log_count - logZ
percentiles = np.exp( log_percentiles )

def par_range( par_chain, percentiles, plusminus ):

    ix_med = np.argmin( np.abs( percentiles-0.5 ) )
    ix_low = np.argmin( np.abs( percentiles-(0.5-plusminus) ) )
    ix_upp = np.argmin( np.abs( percentiles-(0.5+plusminus) ) )    

    median = par_chain[ix_med]
    low = par_chain[ix_low]
    upp = par_chain[ix_upp]
    
    return median, low, upp

medians = np.zeros( npar )
for k in range( npar ):
    median, low, upp = par_range( A_chain[:,k], percentiles[:,k], 0.34 )
    print median, median-low, upp-median
    medians[k] = median
bests = np.zeros( npar )
ix = np.argmax( logL_chain )
bests[0] = A_chain[ix,0]
bests[1] = A_chain[ix,1]

plt.figure()
plt.plot( x, y, '.k' )
plt.plot( x, medians[0]*x + medians[1], '-r' )
plt.plot( x, bests[0]*x + bests[1], '-b' )


###############
# Now compare with MCMC:

a_var = pyhm.Uniform( 'a', lower=-2.5, upper=2.5 )
b_var = pyhm.Uniform( 'b', lower=-2.5, upper=2.5 )
parents = { 'a':a_var, 'b':b_var }

@pyhm.stochastic( observed=True )
def logL_mcmc( value=y, parents=parents ):
    def logp( value, parents=parents ):
        a_val = parents['a']
        b_val = parents['b']
        return logL( a_val, b_val )
model_bundle = { 'logL_mcmc':logL_mcmc, 'a':a_var, 'b':b_var }
mcmc = pyhm.Sampler( model_bundle )
mcmc.assign_step_method( pyhm.BuiltinStepMethods.MetropolisHastings )
mcmc.step_method.step_sizes['a'] = 1e-7
mcmc.step_method.step_sizes['b'] = 1e-7
mcmc.sample( nsteps=20000, ntune_iterlim=10000, tune_interval=42, nconsecutive=3, \
             verbose=1, pickle_chain=None )
