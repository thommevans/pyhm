import numpy as np
import pdb
import Utils

"""
This module contains definitions for basic proposal distribution
classes to be used by the MetropolisHastings sampler. Note that 
these class definitions should contain pretune() methods, rather
than the MetropolisHastings class (which is what it used to be).
"""

class mv_gaussian():
    """
    Take a random draw from a multivariate normal distribution and
    add the perturbations to the corresponding Stoch values. covmatrix
    is an MxM covariance matrix where M is the number of Stochs, and
    covcols is a list containing the labels of each Stoch in the order
    corresponding to the covariance matrix columns.
    """
    def __init__( self, covmatrix=None, covcols=[] ):
        self.proposal_kwargs = { 'covmatrix':covmatrix, 'covcols':covcols }

    def step( self, stochs, **kwargs ):
        keys = stochs.keys()
        npar = len( keys )
        meanvec = np.zeros( npar )
        steps = np.random.multivariate_normal( meanvec, kwargs['covmatrix'] )
        for i in range( npar ):
            key = kwargs['covcols'][i]
            stochs[key].value += steps[i]
    

    def pretune( self, mcmc, covcols=None, covinit=None, ntune_min=10000, ntune_max=30000, \
                  tune_interval=1000, nconsecutive=1, verbose=True, pretune_individ_step_sizes=False ):
        npar = len( covcols )
        if pretune_individ_step_sizes==True:
            step_sizes_init = {}
            for i in range( npar ):
                step_sizes_init[covcols[i]] = np.sqrt( covinit[i,i] )
            step_sizes_tuned = tune_diagonal_gaussian_step_sizes( mcmc, step_sizes_init, \
                                                                  ntune_iterlim=ntune_max, \
                                                                  tune_interval=tune_interval, \
                                                                  rescale_all_together=False, \
                                                                  verbose=verbose, nconsecutive=nconsecutive )
            for i in range( npar ):
                covinit[i,i] = step_sizes_tuned[covcols[i]]**2.
        self.proposal_kwargs = { 'covmatrix':covinit, 'covcols':covcols }
        covtuned = tune_proposal_covmatrix( mcmc, ntune_min=ntune_min, ntune_max=ntune_max, \
                                            tune_interval=tune_interval, nconsecutive=nconsecutive, \
                                            verbose=verbose )
        self.proposal_kwargs = { 'covmatrix':covtuned, 'covcols':covcols }


class diagonal_gaussian():
    """
    Add a random Gaussian perturbation to each Stoch value, where
    step_sizes is a dictionary containing the widths of each
    Gaussian perturbation.
    """

    def __init__( self, step_sizes={} ):
        self.proposal_kwargs = { 'step_sizes':step_sizes }

    def step( self, stochs, **kwargs ):
        keys = stochs.keys()
        for key in keys:
            stochs[key].value += Utils.gaussian_random_draw( mu=0.0, sigma=kwargs['step_sizes'][key] )
        
    def pretune( self, mcmc, ntune_iterlim=0, tune_interval=None, verbose=False, nconsecutive=4 ):
        keys = mcmc.model.free.keys()
        if self.proposal_kwargs['step_sizes']==None:
            self.proposal_kwargs['step_sizes'] = {}
            for key in keys:
                self.proposal_kwargs['step_sizes'][key] = 1.
        untuned_step_sizes = self.proposal_kwargs['step_sizes']
        tuned_step_sizes = tune_diagonal_gaussian_step_sizes( mcmc, untuned_step_sizes, \
                                                              ntune_iterlim=ntune_iterlim, \
                                                              tune_interval=tune_interval, \
                                                              rescale_all_together=True, \
                                                              verbose=verbose, nconsecutive=nconsecutive )
        self.proposal_kwargs['step_sizes'] = tuned_step_sizes


def tune_proposal_covmatrix( mcmc, ntune_min=0, ntune_max=0, tune_interval=500, nconsecutive=3, verbose=False ):
    """
    Tunes a general multivariate normal proposal distribution. 

    TODO: I find that the rescale_factors seem fairly robust
    to different problems/datasets/analyses. However, they're
    not based on anything scientific, just guesses. So it 
    might be worth considering how to make it more principled
    in the future...
    """
    nsteps = 0
    covcols = mcmc.step_method.proposal_distribution.proposal_kwargs['covcols']
    npar = len( covcols )
    chains = []
    accfrac = 0
    nsuccess = 0
    rescale_factor = ( 2.4**2 )/float( npar )
    mcmc.step_method.proposal_distribution.proposal_kwargs['covmatrix'] *= rescale_factor
    while ( nsteps<ntune_min )+( ( nsteps<ntune_max )*( nsuccess<nconsecutive ) ):    
        Utils.mcmc_sampling( mcmc, nsteps=tune_interval, verbose=verbose )
        mcmc._overwrite_existing_chains = False
        cov1 = mcmc.step_method.proposal_distribution.proposal_kwargs['covmatrix']
        nsteps += tune_interval
        chains = []
        for key in covcols:
            chains += [ mcmc.chain[key][-tune_interval:] ]
        chains = np.row_stack( chains )
        cov2 = np.cov( chains )
        covnew = 0.3*cov1 + 0.7*cov2
        nacc = mcmc.chain['accepted'][-tune_interval:].sum()
        accfrac = float( nacc )/float( tune_interval )
        if ( accfrac>=0.2 )*( accfrac<=0.4 ):
            nsuccess +=1
        else:
            nsuccess = 0
        rescale_factor *= ( 1./0.25 )*np.min( [ 0.9, np.max( [ 0.1, float( accfrac ) ] ) ] )
        covtuned = covnew*rescale_factor
        mcmc.step_method.proposal_distribution.proposal_kwargs['covmatrix'] = covtuned
        if verbose==True:
            print '\nAcceptance fraction = {0:.2f} (from last {1:.0f} steps)'.format( accfrac, tune_interval )
            print 'nconsecutive = {0:.0f} (require {1:.0f})'.format( nsuccess, nconsecutive )
            print 'Total tuning steps taken = {0:.0f}'.format( nsteps )
            print '(ntune_min={0:.0f}, ntune_max={1:.0f})\n'.format( ntune_min, ntune_max )
    if nsteps>ntune_max:
        warn_str = '\nAborting tuning - maximum step limit reached'
        warnings.warn( warn_str )
    elif verbose==True:
        print '\nTuning finished with step acceptance rate of {0:.2f}%'.format( accfrac*100 ) 
    mcmc.step_method.proposal_distribution.proposal_kwargs['covmatrix'] = cov1

    return covtuned


def tune_diagonal_gaussian_step_sizes( mcmc, step_sizes, ntune_iterlim=0, tune_interval=None, \
                                       verbose=False, nconsecutive=4, rescale_all_together=True ):
    """
    Tunes a Gaussian proposal distribution with a diagonal covariance matrix.
    This is done by holding all parameters fixed except for one, and tuning 
    the latter. tune_interval steps are taken in the free parameter, the acceptance
    rate is calculated, and the step size is rescaled appropriately. The free
    parameter is then returned to its starting value, and the process is repeated.
    This is continued until nconsecutive iterations produce acceptance rates 
    in the range 20-35%. This process is repeated for each parameter in turn.
    """

    unobs_stochs = mcmc.model.free
    keys = unobs_stochs.keys()
    m = ntune_iterlim
    n = tune_interval
    npars = len( keys )

    # Make a record of the starting values for each parameter:
    orig_stoch_values = {}
    for key in keys:
        orig_stoch_values[key] = unobs_stochs[key].value

    # First of all, we will tune the relative step sizes for
    # all of the parameters by taking steps one parameter at
    # a time. Initialise the arrays that will record the results:
    tuning_chains = {}
    for key in keys:
        tuning_chains[key] = {}
    current_values = {}
    for key in keys:
        tuning_chains[key]['values'] = np.zeros( n, dtype=unobs_stochs[key].dtype )
        tuning_chains[key]['logp'] = np.zeros( n, dtype=float )
        tuning_chains[key]['accepted'] = np.zeros( n, dtype=int )
        current_values[key] = unobs_stochs[key].value

    # Define variables that track the total number of tuning
    # steps that have been taken and the consecutive number of
    # successful tune_intervals:
    for j in range( npars ):

        i = 0 # iteration counter
        nsuccess = 0 # number of consecutive successes
        key_j = keys[j]

        # Proceed to perturb the current parameter only, carrying
        # on until the iteration limit has been reached:
        accfrac_j = 0

        while i<m+1:

            step_size_j = step_sizes[key_j]

            # If there have been nconsecutive successful tune intervals
            # in a row, break the loop:
            if nsuccess>=nconsecutive:
                step_sizes[key_j] *= 0.3
                break

            # If the iteration limit has been reached, return an error:
            elif i==m:
                err_str = 'Aborting tuning - exceeded {0} steps'.format( m )
                err_str += '\n...consider reducing tune_interval'
                raise StandardError( err_str )

            # Otherwise, proceed with the tuning:
            else:
                k = i%n # iteration number within current tuning interval
                i += 1

                # If this is the first iteration in a new tuning interval,
                # reset all parameters to their original values to avoid
                # drifting into low likelihood regions of parameter space:
                if k==0:
                    for key in keys:
                        unobs_stochs[key].value = orig_stoch_values[key]
                    current_logp = mcmc.logp()

                # Take a step in the current parameter while holding the 
                # rest fixed:
                step_size_j = step_sizes[key_j]
                unobs_stochs[key_j].value += Utils.gaussian_random_draw( mu=0.0, sigma=step_size_j )

                # Decide if the step is to be accepted:
                new_logp = mcmc.logp()
                tuning_chains[key_j]['accepted'][k] = mcmc.step_method.decide( current_logp, new_logp )

                # Update the value of the associated stochastic object:
                if ( tuning_chains[key_j]['accepted'][k]==True ):
                    current_logp = new_logp
                    current_values[key_j] = unobs_stochs[key_j].value
                else:
                    unobs_stochs[key_j].value = current_values[key_j]

                # Add the result to the chain:
                tuning_chains[key_j]['values'][k] = current_values[key_j]
                tuning_chains[key_j]['logp'][k] = current_logp

                # If we have reached the end of the current tuning interval,
                # adjust the step size of the current parameter based on the
                # fraction of steps that were accepted:
                if k==n-1:
                    naccepted_j  = np.sum( tuning_chains[key_j]['accepted'] )
                    accfrac_j = naccepted_j/float( n )
                    if ( accfrac_j<=0.01 ):
                        step_sizes[key_j] /= 5.0
                    elif ( accfrac_j>0.01 )*( accfrac_j<=0.05 ):
                        step_sizes[key_j] /= 2.0
                    elif ( accfrac_j>0.05 )*( accfrac_j<=0.10 ):
                        step_sizes[key_j] /= 1.5
                    elif ( accfrac_j>0.10 )*( accfrac_j<=0.15 ):
                        step_sizes[key_j] /= 1.2
                    elif ( accfrac_j>0.15 )*( accfrac_j<0.2 ):
                        step_sizes[key_j] /= 1.1
                    elif ( accfrac_j>0.20 )*( accfrac_j<0.25 ):
                        step_sizes[key_j] /= 1.01
                    elif ( accfrac_j>0.35 )*( accfrac_j<=0.40 ):
                        step_sizes[key_j] *= 1.01
                    elif ( accfrac_j>0.40 )*( accfrac_j<=0.45 ):
                        step_sizes[key_j] *= 1.1
                    elif ( accfrac_j>0.45 )*( accfrac_j<=0.50 ):
                        step_sizes[key_j] *= 1.2
                    elif ( accfrac_j>0.50 )*( accfrac_j<=0.55 ):
                        step_sizes[key_j] *= 1.5
                    elif ( accfrac_j>0.55 )*( accfrac_j<=0.60 ):
                        step_sizes[key_j] *= 2.0
                    elif ( accfrac_j>0.60 ):
                        step_sizes[key_j] *= 5.0

            # If the end of a tune interval has been reached, check
            # if all the acceptance rates were in the required range:
            if ( k==n-1 ):
                if ( accfrac_j>=0.2 )*( accfrac_j<=0.40 ):
                    nsuccess += 1
                else:
                    nsuccess = 0
                if verbose==True:
                    print '\nPre-tuning update for parameter {0} ({1} of {2}):'\
                          .format( key_j, j+1, npars )
                    print 'Consecutive successes = {0}'.format( nsuccess )
                    print 'Accepted fraction from last {0} steps = {1}'\
                          .format( n, accfrac_j )
                    print '(require {0} consecutive intervals with acceptance rate 0.2-0.4)'\
                          .format( nconsecutive )
                    print 'Median value of last {0} steps: median( {1} )={2} '\
                          .format( n, key_j, np.median( current_values[key_j] ) )
                    print 'Starting value for comparison: {0}'.format( orig_stoch_values[key_j] )
                    print 'Current stepsize: {0}'.format( step_sizes[key_j] )

    # Having tuned the relative step sizes, now rescale them together
    # to refine the joint step sizes if requested:
    if rescale_all_together==True:
        i = 0
        nsuccess = 0
        rescale_factor = 1.0/np.sqrt( npars )
        tuning_chain = np.zeros( n, dtype=int )
        if verbose==True:
            print '\n\nNow tuning the step sizes simultaneously...\n'
        while i<m+1:

            # If there have been nconsecutive successful tune intervals
            # in a row, break the loop:
            if nsuccess>=nconsecutive:
                break

            # If the iteration limit has been reached, return an error:
            elif i==m:
                err_str = 'Aborting tuning - exceeded {0} steps'.format( m )
                err_str += '\n...consider reducing tune_interval'
                raise StandardError( err_str )

            # Otherwise, proceed with the tuning:
            else:
                k = i%n # iteration number within current tuning interval
                i += 1

                # If this is the first iteration in a new tuning interval,
                # reset all parameters to their original values to avoid
                # drifting into low likelihood regions of parameter space:
                if k==0:
                    for key in keys:
                        unobs_stochs[key].value = orig_stoch_values[key]
                    current_logp = mcmc.logp()

                # Take a step in all of the parameters simultaneously:
                for key in keys:
                    # If this is the first iteration in a new tuning interval,
                    # rescale the step sizes by a constant factor before
                    # taking the step:
                    if k==0:
                        step_sizes[key] *= rescale_factor
                    unobs_stochs[key].value += Utils.gaussian_random_draw( mu=0.0, sigma=step_sizes[key] )

                # Decide if the step is to be accepted:
                new_logp = mcmc.logp()
                tuning_chain[k] = mcmc.step_method.decide( current_logp, new_logp )
                if ( tuning_chain[k]==True ):
                    current_logp = new_logp
                    for key in keys:
                        current_values[key] = unobs_stochs[key].value
                else:
                    for key in keys:
                        unobs_stochs[key].value = current_values[key]

                # If we have reached the end of the current tuning interval,
                # adjust the step size rescaling factor based on the fraction
                # of steps that were accepted:
                if k==n-1:
                    naccepted = np.sum( tuning_chain )
                    accfrac = naccepted/float( n )
                    if ( accfrac>=0.2 )*( accfrac<=0.4 ):
                        nsuccess += 1
                        rescale_factor = 1.0
                    else:
                        nsuccess = 0
                        if ( accfrac<=0.01 ):
                            rescale_factor = 1./1.6
                        elif ( accfrac>0.01 )*( accfrac<=0.05 ):
                            rescale_factor = 1./1.4
                        elif ( accfrac>0.05 )*( accfrac<=0.10 ):
                            rescale_factor = 1./1.2
                        elif ( accfrac>0.10 )*( accfrac<=0.15 ):
                            rescale_factor = 1./1.1
                        elif ( accfrac>0.15 )*( accfrac<0.2 ):
                            rescale_factor = 1./1.01
                        elif ( accfrac>0.35 )*( accfrac<=0.45 ):
                            rescale_factor = 1.01
                        elif ( accfrac>0.45 )*( accfrac<=0.50 ):
                            rescale_factor = 1.1
                        elif ( accfrac>0.50 )*( accfrac<=0.55 ):
                            rescale_factor = 1.2
                        elif ( accfrac>0.55 )*( accfrac<=0.60 ):
                            rescale_factor = 1.4
                        elif ( accfrac>0.60 ):
                            rescale_factor = 1.6

                    if verbose==True:
                        print 'Consecutive successes = {0}'.format( nsuccess )
                        print 'Accepted fraction from last {0} steps = {1}'\
                              .format( n, accfrac )

        print 'Finished tuning with acceptance rate of {0:.1f}%'.format( accfrac*100 )

    for key in keys:
        unobs_stochs[key].value = orig_stoch_values[key]
    return step_sizes
        


