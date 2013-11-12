import numpy as np
import pdb
import math
import scipy.special
import ModelObjs, Utils

"""
This module contains definitions for Stochs with standard probability
distributions that are used regularly when defining models, such as Gaussian,
Uniform, and Gamma. Additional Stoch definitions can be trivially added.
"""

def Gaussian( name, mu=0.0, sigma=1.0, value=None, observed=False, dtype=float ):
    """
    Gaussian random variable.

    CALLING

      y = pyhm.Gaussian( 'y', mu=0, sig=1, value=3.4, observed=False, dtype=float )

    The above would generate an unobserved Stoch that has a Gaussian probability
    distribution with a mean of 0, a standard deviation of 1, and a current value of 3.4.
    """

    parents = { 'mu':mu, 'sigma':sigma }
    parent_values = Utils.extract_stochastics_values( parents )
    mu_value = parent_values['mu']
    sigma_value = parent_values['sigma']
    
    def logp( value=value, mu=mu_value, sigma=sigma_value ):
        if np.rank( value )==0:
            logp = -0.5*math.log( 2*np.pi*( sigma**2. ) ) \
                   - ( ( value - mu )**2. )/( 2*( sigma**2. ) )
        else:
            logp = np.sum( -0.5*np.log( 2*np.pi*( sigma**2. ) ) \
                           - ( ( value - mu )**2. )/( 2*( sigma**2. ) ) )
        return logp
        
    def random( mu=mu_value, sigma=sigma_value ):
        return np.random.normal( mu, sigma )

    if value==None:
        value = random( mu=mu_value, sigma=sigma_value )
    parents = { 'mu':mu, 'sigma':sigma }
    dictionary = { 'name':name, 'observed':observed, 'dtype':dtype, 'parents':parents, \
                   'value':value, 'logp':logp, 'random':random }

    return ModelObjs.Stoch( dictionary )


def Uniform( name, lower=0.0, upper=1.0, value=None, observed=False, dtype=float ):
    """
    Uniform random variable.

    CALLING

      y = pyhm.Uniform( 'y', lower=0, upper=1, value=0.4, observed=False, dtype=float )

    The above would generate an unobserved Stoch that has a Uniform probability
    distribution between 0 and 1, with a current value of 0.4.
    """

    parents = { 'lower':lower, 'upper':upper }
    parent_values = Utils.extract_stochastics_values( parents )
    lower_value = parent_values['lower']
    upper_value = parent_values['upper']

    if ( value!=None )*( np.rank( value )>0 ):
        n = len( value.flatten() )
    else:
        n = 1

    def logp( value=value, lower=lower_value, upper=upper_value ):
        if np.any( value>=lower )*np.any( value<=upper ):
            logp = 1.0
        else:
            logp = -np.inf
        return logp

    def random( lower=lower_value, upper=upper_value ):
        if n>1:
            return np.random.uniform( low=lower, high=upper, size=n )
        else:
            return np.random.uniform( low=lower, high=upper )

    if value==None:
        value = random( lower=lower_value, upper=upper_value )
    parents = { 'lower':lower, 'upper':upper }
    dictionary = { 'name':name, 'observed':observed, 'dtype':dtype, 'parents':parents, \
                   'value':value, 'logp':logp, 'random':random }

    return ModelObjs.Stoch( dictionary )


def Gamma( name, alpha=1, beta=1, value=None, observed=False, dtype=float ):
    """
    Gamma random variable.

    CALLING

      y = pyhm.Gamma( 'y', alpha=1, beta=10, value=3.4, observed=False, dtype=float )

    The above would generate an unobserved Stoch that has a Gamma probability
    distribution with a shape parameter of 1 and a scale parameter of 10. More explicitly,
    the probability distribution is defined as:

      log_pdf = sum( -log_gamma( alpha ) + alpha*np.log( beta ) \
                     + (alpha-1)*log( value ) - beta*value )
    """

    parents = { 'alpha':alpha, 'beta':beta }
    parent_values = Utils.extract_stochastics_values( parents )
    alpha_value = parent_values['alpha']
    beta_value = parent_values['beta']

    def logp( value=value, alpha=alpha_value, beta=beta_value ):

        if np.any( value<=0 ):
            logp_value = -np.inf
        else:

            # The Python math routine is faster than numpy for single-valued inputs:
            if np.rank( value )==0:
                logp_value = -math.lgamma( alpha ) + alpha*math.log( beta ) \
                             + (alpha-1)*math.log( value ) - beta*value

            # Numpy is much faster for arrays of inputs:
            else:
                logp_value = np.sum( -math.lgamma( alpha ) + alpha*np.log( beta ) \
                                     + (alpha-1)*np.log( value ) - beta*value )

        return float( logp_value )

    def random( alpha=alpha_value, beta=beta_value ):
        return np.random.gamma( shape=alpha, scale=1./beta )

    if ( alpha_value<=0 )+( beta_value<=0 ):
        err_str = 'alpha and beta parameters must both be >0'
        raise ValueError( err_str )

    if value==None:
        value = random( alpha=alpha_value, beta=beta_value )
    parents = { 'alpha':alpha, 'beta':beta }
    dictionary = { 'name':name, 'observed':observed, 'dtype':dtype, 'parents':parents, \
                   'value':value, 'logp':logp, 'random':random }

    return ModelObjs.Stoch( dictionary )

    
