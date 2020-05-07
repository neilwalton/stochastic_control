# __init__.py

__all__ = ['a', 'b', 'c']
__version__ = '0.1'
__author__ = 'Neil Walton'



from .discrete_control.discrete_control import *
#from .continuous_control import continuous_control
from .bandits.bandits import *
from .tabular_rl import *
from .approximate_rl import cross_entropy_method
from .plotting.plotting import *