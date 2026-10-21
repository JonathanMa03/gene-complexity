"""Dictionary fitting, sparse encoding, and interpretation."""
from .training import run as fit_dictionary
from .grid import run_grid
from .solver import encode, fit_dictionary as fit_arrays, kkt_residual, match_atoms
