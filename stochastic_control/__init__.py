# __init__.py

__all__ = ['a', 'b', 'c']
__version__ = '0.1'
__author__ = 'Neil Walton'


from .approximate_rl.cross_entropy_method import *
from .bandits.bandits import *
#from .continuous_control import continuous_control
from .optimal_control import *
from .plotting.plotting import *
#from .stochasitc_approximation import *
from .tabular_rl import *

