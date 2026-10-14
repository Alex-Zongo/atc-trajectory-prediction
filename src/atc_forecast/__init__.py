"""Two-stage aircraft trajectory forecasting under ATC clearance uncertainty.

    stage 1   q(delta | u, kappa, s)   noisy transcript -> distribution over the clearance
    stage 2   p(tau | delta, c)        clearance + context -> distribution over trajectories

See README.md for the pipeline and the module layout.
"""

__version__ = "0.1.0"
