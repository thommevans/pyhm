import sys, inspect, pdb
from . import ModelObjs

"""
This module contains definitions for function decorators that are used
to instantiate instances of objects such as Stochs.
"""

def stochastic( label=None, func=None, observed=False, dtype=float ):
    """
    Decorator for the Stoch class.

    CALLING

      For an unobserved Stoch:

        @pyhm.stochastic( observe=False, dtype=float )
        def A( value=xvalue, parents=parents ):

            def logp( value, parents ):
                ...
                return logp_value

            def random( parents ):
                ...
                return random_draw

      The above will instantiate an unobserved Stoch named A (this will be
      both the name of the variable and it's identity key) with current value equal
      to xvalue, and with parameters par1 and par2.

      Note that xvalue can be a single number or an array. However, it is optional
      to provide a value for an unobserved Stoch - it can also be set to None.
      The external  arguments par1 and par2 can be Stochs, or else anything
      that will be accepted by the logp() function. The random() function is optional,
      and defines how to generate random samples from the Stoch probability
      distribution.

      Alternatively, for an observed Stoch:

        @pyhm.stochastic( observe=True, dtype=float )
        def A( value=xdata, par1=par1, par2=par2 ):

            def logp( value, par1=par1, par2=par2 ):
                ...
                return logp_value

      Unlike the unobserved Stoch case, it is necessary to provide an external
      argument for the value. A random() function is not needed, as the value of an
      observed Stoch is fixed to its 'observed' value.
    """

    def instantiate_stochastic( func, label=label ):

        dictionary = {}

        # Extract the basic properties:
        if label is not None:
            dictionary['name'] = label
        else:
            dictionary['name'] = func.__name__
        dictionary['observed'] = observed
        dictionary['dtype'] = dtype

        # Identify if logp and random functions
        # have been passed in:
        dictionary['logp'] = None
        dictionary['random'] = None
        keys = ['logp', 'random']
        def probe_func( frame, event, arg ):
            if event=='return':
                l = frame.f_locals
                for key in keys:
                    dictionary[key] = l.get( key )
                sys.settrace( None )
            return probe_func
        sys.settrace( probe_func )
        func()

        if dictionary['logp'] is None:
            err_str = '\nStochastic {0} logp not defined'\
                      .format( dictionary['name'] )
            raise ValueError(err_str)
        if ( dictionary['random'] is not None )*( dictionary['observed']==True ):
            err_str = '\nCan\'t have random function defined for Stochastic {0}'\
                      .format( dictionary['name'] )
            err_str += 'because \'observed\' is set to True'
            raise ValueError(err_str)
        
        # Unpack the value and parents inputs.        
        # Work out the parents of the stochastic, which are
        # provided in the parents input dictionary:
        parents = {}
        ( args, varargs, varkw, defaults ) = inspect.getargspec( func )
        if defaults is None:
            defaults = []

        # Check if value has been provided:
        if ( 'value' in args ):
            value_included = True
        else:
            value_included = False

        # Check if parents have been provided:
        if ( 'parents' in args ):
            parents_included = True
        else:
            parents_included = False
            
        # Raise error if value or parents haven't been provided:
        if ( value_included==False )+( parents_included==False ):
            err_str = 'Stochastic {0} value and/or parents not defined properly'\
                      .format( dictionary['name'] )
            raise ValueError( err_str )
        else:
            for i in range( len( args ) ):
                if args[i]=='parents':
                    for key in defaults[i].keys():
                        parents[key] = defaults[i][key]
                elif args[i]=='value':
                    dictionary['value'] = defaults[i]
        dictionary['parents'] = parents

        return ModelObjs.Stoch( dictionary )

    if func:
        return instantiate_stochastic( func )
    else:
        return instantiate_stochastic
    #pdb.set_trace()
    return stochastic_object 
