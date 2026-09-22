"""Fast, dependency-light EvoTac validation environments.

The module exercises the observable policy boundary and fixed-step accounting
while an Isaac GPU is occupied. It is not a replacement for Isaac/TacEx
evidence used for final physical claims.
"""

from evotac.simulation.fast_insert_hole import (
    FastEpisode,
    FastInsertHoleEnv,
    FastSimulationConfig,
    run_episode,
)

__all__ = ["FastEpisode", "FastInsertHoleEnv", "FastSimulationConfig", "run_episode"]
