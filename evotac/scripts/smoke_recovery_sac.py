"""CPU smoke test: collect real-shaped transitions and perform one SAC update."""
import json
import numpy as np

from evotac.learning.recovery_buffer import ReplayBuffer, Transition
from evotac.learning.recovery_sac import RecoverySAC


def main():
    observation_dim, action_dim, batch_size = 128, 7, 8
    buffer = ReplayBuffer(batch_size, seed=7)
    rng = np.random.default_rng(7)
    for _ in range(batch_size):
        observation = rng.normal(size=observation_dim).astype(np.float32)
        next_observation = rng.normal(size=observation_dim).astype(np.float32)
        buffer.add(Transition(observation, np.tanh(rng.normal(size=action_dim)), 0.0,
                              next_observation, False, False))
    learner = RecoverySAC(observation_dim, action_dim, hidden_dim=32, seed=7)
    update = learner.update(buffer.sample(batch_size))
    print(json.dumps({"status": "ok", "buffer_size": len(buffer),
                      "sample_batch": update.batch_size,
                      "critic_loss": update.critic_loss,
                      "actor_loss": update.actor_loss,
                      "alpha": update.alpha}, indent=2))


if __name__ == "__main__":
    main()
