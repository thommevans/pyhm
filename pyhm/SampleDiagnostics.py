import matplotlib.pyplot as plt
import numpy as np
import pdb
import copy
try:
    import emcee
    emcee_imported = True
except:
    emcee_imported = False

"""
This module defines various routines that are useful for quickly inspecting
model samples, e.g. plotting the chains directly, plotting the binned chains,
plotting the chain autocorrelations, computing the Gelman-Rubin statistic.
"""

def plot_running_chain_means( chain, thin_before_plotting=None ):
    """
    For each step in the chain, plots the mean of all steps up until
    that point. Plots are made separately for each parameter.

    CALLING (this needs updating...)

      sp = pyhm.Sampler( stochastic_dict )
      sp.sample( nsteps=1000 )
      pyhm.plot_running_chain_means( sp, thin_before_plotting=5 )

    The above would generate a figure for each unobserved Stoch
    in the model, after thinning the chains by a factor of 5.
    """

    chain = copy.deepcopy( chain )
    nsteps = len( chain['accepted'] )
    freepars = get_freepars( chain )
    npars = len( freepars )
    chainarr = np.zeros( [ nsteps, npars ] )
    for i in range( npars ):
        chainarr[:,i] = chain[freepars[i]]
    rmeans = np.cumsum( chainarr, axis=0 )
    stepcount = np.arange( 1, nsteps+1 )
    for i in range( npars ):
        rmeans[:,i] /= stepcount

    if thin_before_plotting==None:
        if nsteps<2000:
            thin = 1
        else:
            ixs = ( np.arange( nsteps )%2000==0 )
            thin = int( len( np.arange( nsteps )[ixs] ) )
    ixs_plot = ( np.arange( nsteps )%thin==0 )

    for i in range( npars ):
        fig = plt.figure()
        ax = fig.add_subplot( 111 )
        ax.plot( stepcount[ixs_plot], rmeans[ixs_plot,i], '-' )
        ax.set_title( 'Running mean for parameter \'{0}\''.format( freepars[i] ) )
        ax.set_xlabel( 'Step number' )
        ax.set_ylabel( 'Running mean' )

    return None


def plot_chain_traces( chain, nburn=0, show_burn=True, thin_before_plotting=None ):
    """
    Makes simple plots of the chains for each parameter.
 
    CALLING (this needs updating...)

      sp = pyhm.Sampler( stochastic_dict )
      sp.sample( nsteps=1000 )
      pyhm.plot_chain_traces( sp, nburn=100, show_burn=True, thin_before_plotting=5 )

    The above would plot the chain for each unobserved Stoch
    in the model on a separate axis, after discarding nburn steps at
    the start of the chain and thinning by a factor of 5. If show_burn
    is set to True, the burn-in steps will be shown with a different
    colour.
    """
    
    chain = copy.deepcopy( chain )
    nsteps = len( chain['accepted'] )
    freepars = get_freepars( chain )
    npars = len( freepars )
    stepcount = np.arange( 1, nsteps+1 )

    if thin_before_plotting==None:
        if nsteps<2000:
            thin_before_plotting = 1
        else:
            z = ( np.arange( nsteps )%2000==0 )
            thin_before_plotting = int( z.sum() )
    ixs_plot = ( np.arange( nsteps )%thin_before_plotting==0 )
    stepsplot = stepcount[ixs_plot]
    
    nax_perfig = 3
    nfigs = int( np.ceil( float( npars )/nax_perfig ) )

    figcount = 0
    for i in range( npars ):
        subaxn = i%nax_perfig + 1
        if subaxn==1:
            figcount += 1
            fig = plt.figure()
            fig.suptitle( 'Trace plots (figure {0} of {1})'.format( figcount, nfigs ) )
            ax = fig.add_subplot( nax_perfig, 1, subaxn )
            ax0 = ax
        else:
            ax = fig.add_subplot( nax_perfig, 1, subaxn, sharex=ax0 )
        chain_i = chain[freepars[i]][ixs_plot]
        if show_burn==True:
            burn_ixs = ( stepsplot<=nburn )
            ax.plot( stepsplot[burn_ixs], chain_i[burn_ixs], '-r' )
        postburn_ixs = ( stepsplot>nburn )
        ax.plot( stepsplot[postburn_ixs], chain_i[postburn_ixs], '-b' )
        if ( subaxn==nax_perfig )+( i==npars-1 ):
            ax.set_xlabel( 'Step number' )
        ax.set_ylabel( freepars[i] )

    return None


def plot_chain_densities( chain, nburn=0, nbins=30, show_burn=True, thin_before_plotting=None ):
    """
    Plots simple histograms for the chain samples of each parameter.

    CALLING (this needs updating...)

      sp = pyhm.Sampler( stochastic_dict )
      sp.sample( nsteps=1000 )
      pyhm.plot_chain_densities( sp, nburn=100, show_burn=True )

    The above would plot histograms of the binned samples for each
    unobserved Stoch in the model, after discarding nburn steps
    from the start of the chains and thinning by a factor of 5.
    If show_burn is set to True, histograms including the burn-in steps
    will be shown as dashed lines on the same axes.
    """

    chain = copy.deepcopy( chain )
    nsteps = len( chain['accepted'] )
    freepars = get_freepars( chain )
    npars = len( freepars )
    stepcount = np.arange( 1, nsteps+1 )

    if thin_before_plotting==None:
        if nsteps<2000:
            thin_before_plotting = 1
        else:
            z = ( np.arange( nsteps )%2000==0 )
            thin_before_plotting = int( z.sum() )
    ixs_plot = ( np.arange( nsteps )%thin_before_plotting==0 )
    stepsplot = stepcount[ixs_plot]
    
    nax_dims = [ 2, 2 ]
    nax_perfig = nax_dims[0]*nax_dims[1]
    nfigs = int( np.ceil( float( npars )/nax_perfig ) )

    figcount = 0
    for i in range( npars ):
        subaxn = i%nax_perfig + 1
        if subaxn==1:
            figcount += 1
            fig = plt.figure()
            fig.suptitle( 'Binned traces (figure {0} of {1})'.format( figcount, nfigs ) )
            ax = fig.add_subplot( nax_dims[0], nax_dims[1], subaxn )
            ax0 = ax
        else:
            ax = fig.add_subplot( nax_dims[0], nax_dims[1], subaxn )
        chain_i = chain[freepars[i]]
        if show_burn==True:
            ax.hist( chain_i, ls='dashed', bins=nbins, normed=True, color='r', histtype='step' )
        postburn_ixs = ( stepcount>nburn )
        ax.hist( chain_i[postburn_ixs], bins=nbins, normed=True, ls='solid', color='b', histtype='step' )
        ax.text( 0.05, 0.9, '{0}'.format( freepars[i] ), \
                 horizontalalignment='left', verticalalignment='bottom' , transform=ax.transAxes)

    return None


def plot_chain_autocorrs( chain, nburn=None, maxlag=None, thin_before_plotting=None ):
    """
    Plots the autocorrelations of the chains for each parameter.

    CALLING (this needs updating...)

      sp = pyhm.Sampler( stochastic_dict )
      sp.sample( nsteps=1000 )
      pyhm.plot_chain_autocorrs( sp, nburn=100, maxlag=50, thin_before_plotting=5 )

    The above would plot the autocorrelation out to 50 lags for each
    unobserved Stoch in the model on a separate axis, after
    discarding nburn steps at the start of the chain and then thinning
    by a factor of 5.
    """

    chain = copy.deepcopy( chain )
    nsteps = len( chain['accepted'] ) - nburn
    if maxlag==None:
        maxlag = min( [ nsteps, 200 ] )

    if nburn==None:
        nburn = 0

    freepars = get_freepars( chain )
    npars = len( freepars )
    stepcount = np.arange( 1, nsteps+1 )

    if thin_before_plotting==None:
        if nsteps<2000:
            thin_before_plotting = 1
        else:
            z = ( np.arange( nsteps )%2000==0 )
            thin_before_plotting = int( z.sum() )
    ixs_plot = ( np.arange( nsteps )%thin_before_plotting==0 )
    stepsplot = stepcount[ixs_plot]

    nax_dims = [ 2, 2 ]
    nax_perfig = nax_dims[0]*nax_dims[1]
    nfigs = int( np.ceil( float( npars )/nax_perfig ) )

    figcount = 0
    print '\nComputing autocorrelations:'
    for i in range( npars ):
        print '... parameter {0} of {1}'.format( i+1, npars )
        subaxn = i%nax_perfig + 1
        if subaxn==1:
            figcount += 1
            fig = plt.figure()
            fig.suptitle( 'Chain autocorrelations (figure {0} of {1})'.format( figcount, nfigs ) )
            ax = fig.add_subplot( nax_dims[0], nax_dims[1], subaxn )
            ax0 = ax
            ax0.set_xlim( [ -1, maxlag ] )
        else:
            ax = fig.add_subplot( nax_dims[0], nax_dims[1], subaxn )
        chain_i = chain[freepars[i]][nburn:]
        chain_i -= np.mean( chain_i )
        denom = np.sum( chain_i**2. )
        numer = np.correlate( chain_i, chain_i, mode='full' )[nsteps-1:]
        ac = numer/denom
        ax.plot( ac[:maxlag], '-b' )
        ax.axhline( 0, ls='--', c='k' )
        ax.text( 0.90, 0.85, '{0}'.format( freepars[i] ), \
                 horizontalalignment='right', verticalalignment='bottom' , transform=ax.transAxes)

    return None


def gelman_rubin( chain_list, nburn=0, thin=1 ):
    """
    Calculates the convergence statistic of Gelman & Rubin (1992).

    CALLING
    
        pyhm.gelman_rubin( chain_list, nburn=10000 )
    
      The above takes a list of chains as input and calculates the
      Gelman-Rubin convergence statistic for each parameter, after
      discarding nburn steps from the start of each chain. Note that
      each element of the chain list should be a dictionary, and each
      labelled element of each dictionary should be an array containing
      the samples for each parameter, i.e. the standard format in which
      chains are stored when they are generated as attributes of Sampler
      objects.

    DESCRIPTION

      The Gelman-Rubin convergence statistic is calculated as:

         B = ( n/( m-1 ) )*sum_j[ ( theta_j - theta_mean )^2  ]

         W = ( 1/m )*sum_j[ ( 1/( n-1 )*sum_i[ ( theta_ji - theta_j )^2 ] ) ]

         GelmanRubin = sqrt[ ( n-1 )/n + ( 1/n )*( B/W ) ]

      where m is the number of chains, n is the number of samples in each
      chain, the j summation is done over the m chains, the i summation is
      done over the n samples within the jth chain, theta_j is the mean of
      the jth chain, theta_mean is the mean of all samples across all m chains,
      and theta_ji is the ith sample of the jth chain.

      If the GelmanRubin value is close to 1, it suggests that the chains are
      all well-mixed and have likely reached a stable state. If the value is
      more than a few percent from 1, it suggests that the chains have not yet
      converged and more samples are required.

      Note that the formula given above does not include the scaling factor of
      df/(df-2) that Gelman & Rubin originally included. Brooks & Gelman (1998)
      subsequently pointed out that this factor was incorrect anyway, and that
      the correct factor should be (d+3)/(d+1). However, if the chains are close
      to converged, this scaling factor is very close to 1 anyway, which is why
      it is ignored in the pyhm implementation.
    """
    m = len( chain_list )

    keys = []
    for key in chain_list[0].keys():
        if ( key=='logp' )+( key=='accepted' ):
            continue
        else:
            keys += [ key ]

    n = len( chain_list[0][keys[0]] ) - nburn
    ixs = ( np.arange( n )%thin==0 )
    npars = len( keys )
    grs = {}
    for i in range( npars ):
        chains = []
        for j in range(m):
            chains += [ chain_list[j][keys[i]][nburn:][ixs] ]
        chains = np.column_stack(chains)
        W = np.mean( np.var( chains, axis=0, ddof=1 ) )
        B_over_n = np.var( np.mean( chains, axis=0 ), ddof=1 )
        sigma2 = ( ( n-1. )/n )*W + B_over_n
        Vhat = sigma2 + B_over_n/float( m )
        grs[keys[i]] = np.sqrt( Vhat/float( W ) )
                 
    return grs


def walker_chain_autocorr( walker_chain, nburn=None, maxlag=50 ):
    """
    Computes the autocorrelation function and integrated autocorrelation times
    for each parameter, using the routines from emcee.
    """

    if nburn==None:
        nburn = 0

    keys = walker_chain.keys()
    keys.remove( 'logp' )
    npar = len( keys )
    nsteps, nwalkers = np.shape( walker_chain['logp'] )
    y = np.zeros( [ nsteps-nburn, npar ] ) 
    for i in range( npar ):
        y[:,i] = np.mean( walker_chain[keys[i]][nburn:,:], axis=1 ) # average over walkers
    
    acor_func_arr = emcee.autocorr.function( y, axis=0, fast=False )
    acor_integ_arr = emcee.autocorr.integrated_time( y, axis=0, window=maxlag, fast=False )

    acor_func = {}
    acor_integ = {}
    for i in range( npar ):
        acor_func[keys[i]] = acor_func_arr[:,i]
        acor_integ[keys[i]] = acor_integ_arr[i]

    return acor_func, acor_integ
    

def chain_properties( chain, nburn=None, thin=None, print_to_screen=True ):

    freepars = get_freepars( chain )
    npars = len( freepars )

    parkey = []
    mean = {}
    median = {}
    stdev = {}
    l34 = {}
    u34 = {}

    if print_to_screen==True:
        print '\n{0}\nParameter --> Mean, Stdev:'.format( '#'*50 )
    for i in range( npars ):
        parkey += [ str( freepars[i] ) ]
        chain_i = chain[freepars[i]]
        if nburn!=None:
            chain_i = chain_i[nburn:]
        nsteps = len( chain_i )
        if thin!=None:
            ixs = ( np.arange( nsteps )%thin==0 )
            chain_i = chain_i[ixs]
        mean[parkey[i]] = np.mean( chain_i )
        stdev[parkey[i]] = np.std( chain_i )
        if print_to_screen==True:
            print '  {0} --> {1}, {2}'.format( freepars[i], mean[parkey[i]], stdev[parkey[i]] )

    if print_to_screen==True:
        print '\n{0}\nParameter --> Median, -34%, +34%:'.format( '#'*50 )
    for i in range( npars ):
        chain_i = chain[freepars[i]]
        if nburn!=None:
            chain_i = chain_i[nburn:]
        nsteps = len( chain_i )
        if thin!=None:
            ixs = ( np.arange( nsteps )%thin==0 )
            chain_i = chain_i[ixs]
        median[parkey[i]] = np.median( chain_i )
        deltas = chain_i - median[parkey[i]]
        ixsl = ( deltas<0 )
        ixsu = ( deltas>0 )
        n = len( deltas )
        n34 = int( np.round( 0.34*n ) )
        l34[parkey[i]] = deltas[ixsl][np.argsort( deltas[ixsl] )][-n34]
        u34[parkey[i]] = deltas[ixsu][np.argsort( deltas[ixsu] )][n34]
        if print_to_screen==True:
            print '  {0} --> {1}, {2}, {3}'.format( freepars[i], median[parkey[i]], \
                                                    l34[parkey[i]], u34[parkey[i]] )
    output = {}
    output['mean'] = {}
    output['median'] = {}
    output['stdev'] = {}
    output['l34'] = {}
    output['u34'] = {}
    for i in range( npars ):
        output['mean'][parkey[i]] = mean[parkey[i]]
        output['median'][parkey[i]] = median[parkey[i]]
        output['stdev'][parkey[i]] = stdev[parkey[i]]
        output['l34'][parkey[i]] = l34[parkey[i]]
        output['u34'][parkey[i]] = u34[parkey[i]]
        
    return output


def print_chain_properties_OLD( sampler, nburn=None, thin=None ):

    chain = copy.deepcopy( sampler.chain )
    freepars = get_freepars( chain )
    npars = len( freepars )

    print '\n{0}\nParameter --> Mean, Median, Stdev:'.format( '#'*50 )
    for i in range( npars ):
        chain_i = chain[freepars[i]]
        if nburn!=None:
            chain_i = chain_i[nburn:]
        nsteps = len( chain_i )
        if thin!=None:
            ixs = ( np.arange( nsteps )%thin==0 )
            chain_i = chain_i[ixs]
        mean = np.mean( chain_i )
        median = np.median( chain_i )
        stdev = np.std( chain_i )
        print '  {0} --> {1}, {2}, {3}'.format( freepars[i], mean, median, stdev )

    return None


def get_freepars( chain ):
    """
    This is a minor routine that is called upon by some of the
    routines in the SampleDiagnostics module. It identified the
    keys for the model parameters in the chain, i.e. everything
    except for the tallies of 'logp' and 'accepted'.
    """
    freepars = []
    for key in chain.keys():
        if np.any( key==np.array( [ 'logp', 'accepted' ] ) ):
            continue
        else:
            freepars += [ key ]
    return freepars
