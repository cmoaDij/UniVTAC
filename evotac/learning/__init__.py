"""Simulator-independent learning components for EvoTac.

The modules in this package consume the observation/action contracts and do not
import Isaac Sim.  They can therefore be tested and smoke-tested while a GPU
is occupied by another job.
"""

from evotac.learning.recovery_buffer import ReplayBuffer, Transition
from evotac.learning.recovery_sac import MultiSkillRecoverySAC, RecoverySAC
from evotac.learning.recovery_state_machine import RecoveryStateMachine
from evotac.learning.recovery_training import RecoveryTrainingDriver, TrainingEpisodeResult
from evotac.learning.recovery_trainer import RecoverySACTrainer, SACTrainingEpisode
from evotac.learning.continual_adaptation import ContinualAdaptationController
from evotac.learning.recovery_observation import recovery_summary, recovery_vector
from evotac.learning.effect_training import EffectTrainer, EffectTrainingStep
from evotac.learning.effect_model import EffectHistoryEncoder
from evotac.learning.history_encoder import HistoryEncoder
from evotac.learning.recovery_observation import RecoveryHistory
from evotac.learning.recovery_warmstart import RecoveryWarmStart
from evotac.learning.skill_selector import SkillSelector

__all__ = ["ReplayBuffer", "RecoverySAC", "RecoverySACTrainer", "RecoveryStateMachine",
           "RecoveryTrainingDriver", "SACTrainingEpisode", "ContinualAdaptationController", "SkillSelector",
           "TrainingEpisodeResult", "Transition", "recovery_summary", "recovery_vector", "MultiSkillRecoverySAC",
           "EffectTrainer", "EffectTrainingStep", "EffectHistoryEncoder", "HistoryEncoder", "RecoveryHistory",
           "RecoveryWarmStart"]
