import numpy as np 
import inspect
import sys, pdb
import Utils
import Optimizers
import BuiltinStepMethods

"""
This module defines the fundamental objects used for building models and
maximising/sampling from their posterior distributions. Namely:
  Sampler
  MAP
  Model
  Stoch
See the documentation for each of these object classes for more information.
"""

# NOTE: This used to be 'class Sampler()' ...
class MCMC():
    """
    Class definition for a Sampler object.

    CALLING 
      sampler = pyhm.Sampler( stochastic_dict )

    BUILT-IN ROUTINES
      use_step_method
      sample
      draw_from_prior
      logp

    DESCRIPTION
      An MCMC object takes a dictionary of Stochs as input, where the
      Stochs comprise a self-consistent hierarchical model,
      i.e. containing at least one observed Stoch that depends upon one
      or more unobserved Stochs. The primary purpose of the MCMC 
      object is to take random samples from the posterior distribution of the
      hierarchical model. Currently, it is only possible to generate these samples
      using the standard Metropolis-Hastings algorithm. However, the code is 
      intended to be extensible - other sampling algorithms should be added
      For instance, it may be possible to wrap the affine-invariant package
      emcee inside this class.
    """
    
    def __init__( self, stochastics, overwrite_existing_chains=False ):
        """
        Initialises a blank MCMC object.
        """
        self.model = Model( stochastics )
        Utils.update_attributes( self, stochastics )
        self.chain = {}
        self._chain_exists = False
        self._overwrite_existing_chains = overwrite_existing_chains
        Utils.assign_step_method( self, BuiltinStepMethods.MetropolisHastings )
        self.show_progressbar = True
            
    def assign_step_method( self, step_method, **kwargs ):
        """
        Assigns the specified step method to the Sampler object.
        """
        Utils.assign_step_method( self, step_method, **kwargs )

    def sample( self, nsteps=1000, ntune_iterlim=None, tune_interval=None, nconsecutive=4, \
                show_progressbar=True, pickle_chain=None, thin_before_pickling=1, verbose=False, \
                overwrite_existing_chains=False ):
        """
        Sample from the posterior distribution and optionally pickle the output.
        """        
        self._overwrite_existing_chains = overwrite_existing_chains
        self.show_progressbar = show_progressbar
        Utils.mcmc_sampling( self, nsteps=nsteps, ntune_iterlim=ntune_iterlim, nconsecutive=nconsecutive, \
                             tune_interval=tune_interval, verbose=verbose )
        if pickle_chain!=None:
            Utils.pickle_chain( self, pickle_chain=pickle_chain, thin_before_pickling=thin_before_pickling )

    def draw_from_prior( self ):
        """
        Draw a random sample from the model prior.
        """
        self.model.draw_from_prior()
        return None

    def logp( self ):
        """
        Evaluate the log likelihood of the model.
        """
        return self.model.logp()


class NestedSampler():

    def __init__( self, stochastics ):
        """
        Initialises a blank NestedSampler object.
        """
        self.model = Model( stochastics )
        Utils.update_attributes( self, stochastics )

    def sample( self, n_active=100, stopping_criterion=[ 'Z_convergence', 0.01 ], \
                pickle_chain=None, thin_before_pickling=1, verbose=False ):

        Utils.nested_sampling( self, n_active, stopping_criterion=stopping_criterion, verbose=verbose )
        if pickle_chain!=None:
            Utils.pickle_chain( self, pickle_chain=pickle_chain, thin_before_pickling=thin_before_pickling )

    def draw_from_prior( self ):
        """
        Draw a random sample from the model prior.
        """
        self.model.draw_from_prior()
        return None

    def logp( self ):
        """
        Evaluate the log likelihood of the model.
        """
        return self.model.logp()

class MAP():
    """
    Class definition for a maximum a posteriori (MAP) object.

    CALLING
      mp = pyhm.MAP( stochastic_dict )

    BUILT-IN ROUTINES
      fit
      draw_from_prior
      logp

    DESCRIPTION
      A MAP object takes a dictionary of Stochs as input, where the
      StochasticsObjs comprise a self-consistent hierarchical model,
      i.e. containing at least one observed Stoch that depends upon
      one or more unobserved Stochs. The primary purpose of the MAP
      object is to identify the values of the unobserved Stochs that
      maximise the model posterior distribution. In practice, this is implemented
      using standard scipy optimisation routines.
    """

    def __init__( self, stochastics ):
        """
        Initialises a blank MAP object.
        """
        self.model = Model( stochastics )
        Utils.update_attributes( self, stochastics )

    def fit( self, method='neldermead', maxiter=1000, ftol=0.01 ):
        """
        Compute the maximum a posteriori solution using specified algorithm.
        """
        Optimizers.optimize( self, method=method, maxiter=maxiter, ftol=ftol )
        return None

    def draw_from_prior( self ):
        """
        Draw a random sample from the model prior.
        """
        self.model.draw_from_prior()
        return None
    
    def logp( self ):
        """
        Evaluate the log likelihood of the model.
        """
        return self.model.logp()


class Model():
    """
    Class definition for a Model object.

    A Model consists of a collection of Stochs contained within
    a dictionary. The dictionary can contain non-Stoch variables.
    Note that the user should not need to explicitly create Model objects
    - this is done internally by Sampler and MAP objects.
    """
    
    def __init__( self, stochastics ):
        self.free = Utils.unobserved_stochastics( stochastics )
        self.fixed = Utils.observed_stochastics( stochastics )
        if len( self.fixed )==0:
            err_str = 'Model must contain at least one observed stochastic'
            raise ValueError( err_str )
        self.stochastics = Utils.identify_stochastics( stochastics )
        Utils.check_model_stochastics_names( self )
        Utils.trace_ancestries( self )
        Utils.update_attributes( self, stochastics )
        
    def logp( self ):
        logp_value = 0.0
        keys = self.free.keys()
        # Ensure that the free parameters are valid
        # according to the defined priors:
        for key in self.free.keys():
            logp_i = self.free[key].logp()
            if np.isfinite( logp_i )==False:
                logp_value = -np.inf
                break
            else:
                logp_value += logp_i
        # Finish the logp calculation by evaluating
        # for the observed variables:
        if np.isfinite( logp_value )==True:
            for key in self.fixed.keys():
                logp_value += self.fixed[key].logp()
        return logp_value

    def logp_without_prior( self ):
        logp_value = 0.0
        # Evaluate logp for the observed variables:
        if np.isfinite( logp_value )==True:
            for key in self.fixed.keys():
                logp_value += self.fixed[key].logp()
        return logp_value

    def draw_from_prior( self ):
        Utils.random_draw_from_Model( self )
        return None


class Stoch():
    """
    Class definition for a Stoch.

    There are two broad categories of Stochs:
        1. unobserved - A model variable for which the values of its
        parents do not uniquely determine its value.
        2. observed - A variable with fixed value. In practice, the 
        observed dataset is always defined as an observed Stoch.

    Stochs can be created by defining a function of the form:
        
        def A( name, par1=0.0, par2=1.0, value=xdata, observed=False, dtype=float ):

            def logp( value=value, par1=par1, par2=par2 ):
                ...
                return logp_value

            def random( par1=par1, par2=par2 ):
                ...
                return random_draw

            parents = { 'par1':par1, 'par2':par2 }
            dictionary = { 'name':name, 'observed':observed, 'dtype':dtype, \
                           'parents':parents, 'value':value, 'logp':logp, \
                           'random':random }

            return ModelObjs.Stoch( dictionary )

    where:
        - 'A' is the name of the Stoch, and can be anything
        - name is a string containing the key used to reference
          the Stoch in dictionaries
        - 'par1', 'par2', ... are the parameters of the Stoch,
          which can be numerical values or other Stochs.
        - 'value' is the current numerical value of the Stoch
        - 'observed' is either True or False depending on whether the
          Stoch is observed or not (see above)
        - 'dtype' specifies the data type of the Stoch
        - 'logp' is a function that defines the loglikelihood of the
          Stoch
        - 'random' is an optional function that generates a random sample
          from the Stoch distribution
        - 'parents' is a dictionary containing the variables that define
          the probability distribution of the Stoch
        - 'dictionary' is a dictionary containing the variables required
          to fully define a Stoch

    A simpler way to define a Stoch is to use the decorator syntax:

        @pyhm.stochastic( observed=True, dtype=float ):
            
            def A( value=xdata, par1=par1, par2=par2 ):

                def logp( value, par1=par1, par2=par2 ):
                    ...
                    return logp_value

                def random( par1=par1, par2=par2 ):
                    ...
                    return random_draw

    In addition, a number of basic Stochs are built-in to pyhm in the
    BuiltinStochastics module, including those with Gaussian, Uniform, and Gamma
    probability distributions.
    """

    def __init__( self, dictionary ):
        self.is_stochastic = True
        self.name = dictionary['name']
        self.observed = dictionary['observed']
        self.dtype = dictionary['dtype']        
        self.parents = dictionary['parents']
        self.value = dictionary['value']
        self._logp_basefunc = dictionary['logp']
        self._random_basefunc = dictionary['random']
        if ( self.observed==True )*( self.value==None ):
            err_str = 'Observed variables must have value defined'
            raise ValueError(err_str)

    def logp( self ):
        logp_func = self._logp_basefunc
        if self.value==None:
            err_str = '\nStochastic {0} value not defined - can\'t compute logp'.\
                      format( self.name )
            raise ValueError(err_str)
        parent_vals = Utils.extract_stochastics_values( self.parents )
        logp_value = logp_func( self.value, parent_vals )
        return logp_value

    def random( self ):
        random_func = self._random_basefunc
        if random_func==None:
            random_draw = Utils.blank_random
        else:
            #kwargs = Utils.extract_stochastics_values( self.parents )
            #random_draw = random_func( **kwargs )
            parent_vals = Utils.extract_stochastics_values( self.parents )
            random_draw = random_func( parent_vals )
        self.value = random_draw
        return random_draw

