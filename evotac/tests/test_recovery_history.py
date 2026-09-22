import json
import numpy as np
import pytest
import torch

from evotac.learning.history_encoder import HistoryEncoder
from evotac.learning.recovery_warmstart import RecoveryWarmStart
from evotac.scripts.train_recovery_warmstart import train as train_warmstart
from evotac.learning.recovery_observation import RECOVERY_OBSERVATION_DIM, RecoveryHistory, recovery_vector
from evotac.policy.tactile_features import CachedTactileEncoder


def frame(step=6):
    return {
        "physics_step": step,
        "tactile_embeddings": np.zeros((3, 2, 512), np.float32) + step,
        "difference_cues": np.zeros((2, 2, 9), np.float32),
        "proprioception": np.zeros(23, np.float32),
        "previous_accepted_target": np.zeros(9, np.float32),
        "previous_action_valid": False,
        "tactile_age_steps": np.zeros(2, np.float32),
        "camera_age_steps": np.zeros(2, np.float32),
        "camera_cues": np.zeros((2, 6), np.float32),
        "reference_physics_steps": np.asarray([1, 3]),
        "remaining_task_steps": 100,
        "remaining_recovery_steps": 100,
    }


def test_recovery_frame_and_history_are_causal_and_fixed_size():
    assert recovery_vector(frame()).shape == (RECOVERY_OBSERVATION_DIM,)
    history = RecoveryHistory(length=8)
    first, first_mask = history.append(frame(6))
    assert first.shape == (8, RECOVERY_OBSERVATION_DIM)
    assert first_mask.tolist() == [False] * 7 + [True]
    second, second_mask = history.append(frame(12))
    assert second_mask.tolist() == [False] * 6 + [True, True]
    np.testing.assert_allclose(second[-1], recovery_vector(frame(12), previous_physics_step=6))
    with pytest.raises(ValueError):
        history.append(frame(12))
    history.reset()
    assert history.append(frame(12))[1].tolist() == [False] * 7 + [True]


def test_history_encoder_freeze_and_checkpoint():
    encoder = HistoryEncoder(input_dim=RECOVERY_OBSERVATION_DIM, hidden_dim=8, output_dim=5)
    frames = np.zeros((2, 8, RECOVERY_OBSERVATION_DIM), np.float32)
    mask = np.zeros((2, 8), bool)
    mask[:, -2:] = True
    output = encoder(frames, mask)
    assert output.shape == (2, 5)
    restored = HistoryEncoder.from_checkpoint(encoder.checkpoint())
    np.testing.assert_allclose(output.detach().numpy(), restored(frames, mask).detach().numpy())
    encoder.freeze()
    encoder.train(True)
    assert not encoder.training and not any(parameter.requires_grad for parameter in encoder.parameters())


def test_cached_encoder_batches_only_new_images():
    class Probe:
        version = {"test": True}

        def __init__(self):
            self.calls = 0

        def __call__(self, images):
            self.calls += 1
            return np.repeat(images.mean((1, 2, 3))[:, None], 2, axis=1)

    probe = Probe()
    cache = CachedTactileEncoder(probe, capacity=6)
    images = np.zeros((2, 3, 4, 3), np.uint8)
    cache(images)
    cache(images.copy())
    assert probe.calls == 1 and cache.encoded_images == 1 and cache.cache_hits == 3


def test_warmstart_trains_each_named_skill_and_exports_training_provenance():
    model = RecoveryWarmStart(("lift", "align"), encoder=HistoryEncoder(input_dim=16, hidden_dim=8,
                                                                          output_dim=8, history_length=2),
                             actor_hidden_dim=8)
    loss = model.train_batch({"history": np.zeros((2, 2, 16), np.float32),
                              "mask": np.ones((2, 2), bool),
                              "action": np.zeros((2, 7), np.float32),
                              "skill_indices": np.asarray([0, 1], np.int64)})
    assert np.isfinite(loss) and model.steps == 1
    state = model.checkpoint([{"split": "train", "skill": "lift"},
                              {"split": "train", "skill": "align"}])
    restored = RecoveryWarmStart.from_checkpoint(state)
    assert restored.steps == 1 and restored.skill_names == ("lift", "align")


def test_warmstart_script_requires_train_split_and_writes_checkpoint(tmp_path):
    metadata = {"schema": "evotac.recovery_history_dataset.v1", "split": "train",
                "skill_names": ["lift", "align"],
                "provenance": [{"split": "train", "skill": "lift", "parent_scene_id": "p0"},
                               {"split": "train", "skill": "align", "parent_scene_id": "p1"}]}
    path = tmp_path / "history.npz"
    np.savez_compressed(path, history=np.zeros((2, 8, RECOVERY_OBSERVATION_DIM), np.float32),
                        mask=np.ones((2, 8), bool), action=np.zeros((2, 7), np.float32),
                        skill_indices=np.asarray([0, 1]), metadata=json.dumps(metadata))
    output = tmp_path / "warmstart.pt"
    report = train_warmstart(path, output, epochs=1, batch_size=2)
    assert report["status"] == "written"
    restored = RecoveryWarmStart.from_checkpoint(torch.load(output, weights_only=False))
    assert restored.skill_names == ("lift", "align")
    # The online training entry must consume this exact writer output and
    # retain the trained actor, rather than silently starting a random one.
    from evotac.scripts.train_recovery import initialize_models
    config = {"action_dim": 7, "hidden_dim": 128, "device": "cpu"}
    with pytest.raises(ValueError, match="skill-name"):
        initialize_models(config, checkpoint=output)
    encoder, learner, info = initialize_models(config, checkpoint=output, skill_name="align")
    frames = np.zeros((1, 8, RECOVERY_OBSERVATION_DIM), np.float32)
    mask = np.ones((1, 8), bool)
    with torch.inference_mode():
        expected_state = restored.encoder(frames, mask)
        actual_state = encoder(frames, mask)
        expected_action, _ = restored.actors["align"].sample(expected_state, deterministic=True)
    np.testing.assert_array_equal(actual_state.numpy(), expected_state.numpy())
    np.testing.assert_array_equal(learner.act(actual_state.numpy()[0], deterministic=True),
                                  expected_action.numpy()[0])
    assert info["status"] == "frozen_warmstart_checkpoint"
    assert not encoder.training and not any(p.requires_grad for p in encoder.parameters())
    state = torch.load(output, weights_only=False)
    state["provenance"][0]["split"] = "dev"
    torch.save(state, output)
    with pytest.raises(ValueError, match="training parents"):
        initialize_models(config, checkpoint=output, skill_name="align")
