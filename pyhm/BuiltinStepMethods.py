from __future__ import print_function
import numpy as np
import pdb
from . import BuiltinProposals
from . import Utils

"""
This module contains definitions for algorithms to be used by MCMC objects
to sample from the posterior distribution of models. Currently, it only contains
the definition for a basic Metropolis sampler. 
"""


class MetropolisHastings():
    """
    Metropolis-Hastings sampling algorithm with Gaussian proposal distributions
    for each of the free parameters.
    """

    def __init__( self ):
        """
        Initialises the sampling algorithm.
        """
        self.proposal_distribution = None


    def assign_proposal_distribution( self, proposal_distribution ):
        """
        Assigns the specified proposal distribution to the Sampler.StepMethod object.
        """
        Utils.assign_proposal_distribution( self, proposal_distribution )


    def propose( self, unobs_stochs ):
        """
        Proposes a step in the parameter space.
        """
        proposal_kwargs = self.proposal_distribution.proposal_kwargs
        self.proposal_distribution.step( unobs_stochs, **proposal_kwargs )


    def decide( self, current_logp, new_logp ):
        """
        Decides whether or not to accept the current step.
        """
        beta = new_logp - current_logp
        if beta>0:
            decision = True
        else: 
            alpha = np.exp( beta )
            z = np.random.random()
            if z<=alpha:
                decision = True
            else:
                decision = False
        return decision

    def pretune( self, mcmc, **kwargs ):
        self.proposal_distribution.pretune( mcmc, **kwargs )

    
class AffineInvariant():
    """
    """

    def __init__( self ):
        """
        Initialises the sampling algorithm.
        """
        self.proposal_distribution = None



    
