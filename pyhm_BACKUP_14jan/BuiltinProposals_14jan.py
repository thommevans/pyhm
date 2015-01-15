import numpy as np
import pdb
import Utils
import warnings

"""
This module contains definitions for basic proposal distribution
classes to be used by the MetropolisHastings sampler. Note that 
these class definitions should contain pretune() methods, rather
than the MetropolisHastings class (which is what it used to be).
"""

class diag_gauss():
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
        
    def pretune( self, mcmc, ntune_max=0, tune_interval=None, verbose=False, nconsecutive=4 ):
        keys = mcmc.model.free.keys()
        if self.proposal_kwargs['step_sizes']==None:
            self.proposal_kwargs['step_sizes'] = {}
            for key in keys:
                self.proposal_kwargs['step_sizes'][key] = 1.
        untuned_step_sizes = self.proposal_kwargs['step_sizes']
        tuned_step_sizes = tune_diag_gauss_step_sizes( mcmc, untuned_step_sizes, \
                                                       ntune_max=ntune_max, \
                                                       tune_interval=tune_interval, \
                                                       rescale_all_together=True, \
                                                       verbose=verbose, nconsecutive=nconsecutive )
        self.proposal_kwargs['step_sizes'] = tuned_step_sizes


class mv_gauss():
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
        covcols = kwargs['covcols']
        covmatrix = kwargs['covmatrix']
        npar = len( covcols )
        meanvec = np.zeros( npar )
        steps = np.random.multivariate_normal( meanvec, covmatrix )
        for i in range( npar ):
            #print 'CCCC', key, stochs[key].value, steps[i]
            stochs[covcols[i]].value += steps[i]
        #pdb.set_trace()

    def pretune( self, mcmc, pretune_individ_step_sizes=False, individ_step_sizes_init=None, nindivid_iters=1, \
                 covcols=None, covinit=None, ntune_min=0, ntune_max=0, tune_interval=None, nconsecutive=3, \
                 verbose=None ):
        npar = len( covcols )
        # If requested, refine the initial guesses for the individual parameter
        # step sizes one-at-a-time before tuning them simultaneously below:
        if pretune_individ_step_sizes==True:
            for i in range( nindivid_iters ):
                individ_step_sizes_tuned = tune_diag_gauss_step_sizes( mcmc, individ_step_sizes_init, \
                                                                       ntune_max=ntune_max, \
                                                                       tune_interval=tune_interval, \
                                                                       rescale_all_together=False, \
                                                                       verbose=verbose, \
                                                                       nconsecutive=nconsecutive )
                individ_step_sizes_init = individ_step_sizes_tuned
            covinit = np.eye( npar )
            for i in range( npar ):
                covinit[i,i] = individ_step_sizes_tuned[covcols[i]]**2.
        # Tune the full covariance matrix of the proposal distribution
        # by stepping in all parameters simultaneously and computing
        # sample covariances of the chains:
        self.proposal_kwargs = { 'covmatrix':covinit, 'covcols':covcols }
        #for i in range( npar ):
        #    print covcols[i], mcmc.model.free[covcols[i]].value, np.sqrt(covinit[i,i])
        #pdb.set_trace()
        covtuned = tune_mv_gauss_proposal_covmatrix( mcmc, ntune_min=ntune_min, ntune_max=ntune_max, \
                                                     tune_interval=tune_interval, nconsecutive=nconsecutive, \
                                                     verbose=verbose )
        self.proposal_kwargs = { 'covmatrix':covtuned, 'covcols':covcols }
        return None

def tune_mv_gauss_proposal_covmatrix( mcmc, ntune_min=0, ntune_max=0, tune_interval=500, \
                                      nconsecutive=3, verbose=False ):
    #mcmc._overwrite_existing_chains = True
    #Utils.mcmc_sampling( mcmc, nsteps=tune_interval, verbose=False )
    #nsteps = tune_interval
    nsteps = 0
    covcols = mcmc.step_method.proposal_distribution.proposal_kwargs['covcols']
    npar = len( covcols )
    Inpar = np.eye( npar )
    chains = []
    accfrac = 0
    nsuccess = 0
    #if verbose==True:
    #    print '\nTuning covariance matrix in blocks of {0:.0f} steps...'.format( tune_interval )
    #    print '\nSteps taken = {0:.0f} (ntune_min={1:.0f}, ntune_max={2:.0f})'\
    #          .format( nsteps, ntune_min, ntune_max )
    #    print 'Step acceptance = {0:.1f}% (for last {1:.0f} steps)'.format( accfrac*100, tune_interval )
    #    print '... for first block of steps\n'
    mcmc._overwrite_existing_chains = False
    while ( nsteps<ntune_max )*( ( nsuccess<nconsecutive )+( nsteps<ntune_min ) ):

        cov1 = mcmc.step_method.proposal_distribution.proposal_kwargs['covmatrix']
        print '\n\n STARTING COV', np.diag(cov1)        
        Utils.mcmc_sampling( mcmc, nsteps=tune_interval, verbose=verbose )
        nsteps += tune_interval
        chains = []
        for key in covcols:
            chains += [ mcmc.chain[key][-tune_interval:] ]
        chains = np.row_stack( chains )
        cov2 = np.cov( chains )
        print '\nCCC EMPIRICAL COV', np.diag(cov2)        
        print np.shape(chains )
        mcmc.step_method.proposal_distribution.proposal_kwargs['covmatrix'] = 0.2*cov1 + 0.8*cov2
        nacc = mcmc.chain['accepted'][-tune_interval:].sum()
        accfrac = float( nacc )/float( tune_interval )
        if ( accfrac>=0.2 )*( accfrac<=0.4 ):
            nsuccess +=1
            rescale_factor = 1
            #print 'IGNORE'
        else:
            nsuccess = 0
            nn=1./2.
            if ( accfrac<=0.01 ):
                rescale_factor = 1./2.
            elif ( accfrac>0.01 )*( accfrac<=0.05 ):
                rescale_factor = 1./1.5
            elif ( accfrac>0.05 )*( accfrac<=0.10 ):
                rescale_factor = 1./1.3
            elif ( accfrac>0.10 )*( accfrac<=0.15 ):
                rescale_factor = 1./1.2
            elif ( accfrac>0.15 )*( accfrac<0.2 ):
                rescale_factor = 1./1.1
            elif ( accfrac>0.35 )*( accfrac<=0.45 ):
                rescale_factor = 1.1
            elif ( accfrac>0.45 )*( accfrac<=0.50 ):
                rescale_factor = 1.2
            elif ( accfrac>0.50 )*( accfrac<=0.55 ):
                rescale_factor = 1.3
            elif ( accfrac>0.55 )*( accfrac<=0.60 ):
                rescale_factor = 1.5
            elif ( accfrac>0.60 ):
                rescale_factor = 2.
            #rescale_factor=1.
            #rescale_factor = 
            mcmc.step_method.proposal_distribution.proposal_kwargs['covmatrix'] *= rescale_factor#*Inpar#(1./0.23)*min(0.9,max(0.1,accfrac)) #rescale_factor**2.
            #mcmc.step_method.proposal_distribution.proposal_kwargs['covmatrix'] = cov2#*rescale_factor
            #mcmc.step_method.proposal_distribution.proposal_kwargs['covmatrix'] = covnew*rescale_factor
            #cov1 = covnew*rescale_factor
        if verbose==True:
            print '\nSteps taken = {0:.0f} (ntune_min={1:.0f}, ntune_max={2:.0f})'\
                  .format( nsteps, ntune_min, ntune_max )
            print 'Step acceptance = {0:.1f}% (for last {1:.0f} steps)'.format( accfrac*100, tune_interval )
            print 'Consecutive successes = {0:.0f} (require {1:.0f})'.format( nsuccess, nconsecutive )
            print 'RESCALE FACTOR\n', rescale_factor
        #print rescale_factor, accfrac
        #pdb.set_trace()

    if nsteps>=ntune_max:
        print '\n'
        warn_str = 'Maximum number of tuning steps exceeded!'
        warnings.warn( warn_str )
    
    return mcmc.step_method.proposal_distribution.proposal_kwargs['covmatrix']


def tune_diag_gauss_step_sizes( mcmc, step_sizes, ntune_max=0, tune_interval=None, \
                                verbose=False, nconsecutive=4, rescale_all_together=True ):

        unobs_stochs = mcmc.model.free
        keys = unobs_stochs.keys()
        m = ntune_max
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
