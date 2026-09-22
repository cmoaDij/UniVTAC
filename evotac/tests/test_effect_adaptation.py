import json
import numpy as np
import pytest
import torch

from evotac.learning.continual_adaptation import (AcceptanceGate, AdaptationRound,
                                                   CandidateEvaluation, ContinualAdaptationController,
                                                   adaptation_mix)
from evotac.learning.effect_model import EffectHistoryEncoder, EffectLabel, EffectPredictor, collate_effect_labels
from evotac.learning.effect_training import EffectTrainer
from evotac.learning.recovery_state_machine import RecoveryState, RecoveryStateMachine
from evotac.learning.skill_selector import EffectMemory, EffectRecord, SkillSelector
from evotac.scripts.accept_adaptation import accept


def test_state_machine_requires_three_stable_cycles():
    machine = RecoveryStateMachine(stable_cycles=3)
    trigger = machine.observe({"object_lost_risk": 0.9, "remaining_physics_steps": 30})
    assert trigger.current is RecoveryState.SELECT
    assert machine.observe({"remaining_physics_steps": 30}, selected_skill="skill").current is RecoveryState.RECOVERY
    assert machine.observe({"stable": True}) is None
    assert machine.observe({"stable": True}) is None
    assert machine.observe({"stable": True}).current is RecoveryState.HANDOFF
    assert machine.observe({}).current is RecoveryState.NORMAL


def test_effect_predictor_masked_loss_and_memory_selection(tmp_path):
    predictor = EffectPredictor(4, 2, ("a", "b"), hidden_dim=8)
    values = predictor(np.zeros((2, 4)), np.zeros((2, 2)), np.ones(2) * 10)
    labels = {key: np.zeros((2, 2), dtype=np.float32) for key in ("success", "cost", "violation")}
    loss = predictor.masked_loss(values, labels, {"valid": np.ones((2, 2), dtype=np.float32)})
    assert np.isfinite(float(loss))
    metric = predictor.metric_loss(np.ones((2, 4)), np.zeros((2, 4)), np.zeros(2))
    assert np.isfinite(float(metric))
    memory = EffectMemory()
    memory.add(EffectRecord(np.zeros(4), "a", 1, 1, 0, "v1"))
    assert memory.retrieve(np.ones(4), version="v2") == []
    path = tmp_path / "effect_memory.json"
    memory.save(path)
    restored = EffectMemory.load(path)
    assert len(restored.retrieve(np.ones(4), version="v1")) == 1
    selection = SkillSelector(("a", "b"), min_success=0.5).select({"a": {"success_probability": 0.4, "cost": 0, "violation_probability": 0}, "b": {"success_probability": 0.7, "cost": 0, "violation_probability": 0}})
    assert selection.skill_name == "b"


def test_acceptance_gate_requires_paired_evidence_and_mix():
    gate = AcceptanceGate()
    base = CandidateEvaluation("base", (0.2, 0.2), (0.8, 0.8))
    candidate = CandidateEvaluation("new", (0.8, 0.8), (0.8, 0.8))
    assert gate.evaluate(base, candidate).accepted
    assert gate.evaluate(base, CandidateEvaluation("new", (0.8,), (0.8, 0.8))).reason == "insufficient_paired_evidence"
    assert len(adaptation_mix([1, 2], [3, 4], [5, 6], seed=3)) >= 3
    with pytest.raises(ValueError):
        EffectLabel(np.zeros(2), 0, 0, 0, "", "v1")


def test_continual_controller_accepts_and_persists_only_paired_candidate(tmp_path):
    controller = ContinualAdaptationController(current_version="v1", skill_count=1, seed=4)
    baseline = CandidateEvaluation("v1", (0.2, 0.2), (0.8, 0.8), (0.1, 0.1))
    candidate = CandidateEvaluation("v2", (0.8, 0.8), (0.8, 0.8), (0.1, 0.1))
    result = controller.propose(condition="lateral_offset", candidate_version="v2",
                                 candidate_skill_count=2, current_samples=[1, 2],
                                 retrieved_samples=[3], uniform_old_samples=[4],
                                 baseline=baseline, candidate=candidate)
    assert isinstance(result, AdaptationRound) and result.accepted
    assert controller.current_version == "v2" and controller.skill_count == 2
    path = tmp_path / "controller.json"
    controller.save(path)
    restored = ContinualAdaptationController.load(path)
    assert restored.current_version == "v2" and len(restored.history) == 1


def test_accept_adaptation_script_pairs_parent_seed_evidence(tmp_path):
    baseline_path, candidate_path = tmp_path / "baseline.jsonl", tmp_path / "candidate.jsonl"
    rows = []
    for index in range(3):
        rows.append({"parent_scene_id": f"p{index}", "seed": index, "split": "acceptance",
                     "new_condition_success": 0.1, "old_condition_success": 0.8,
                     "new_condition_violation": 0.1, "old_condition_violation": 0.1})
    baseline_path.write_text("\n".join(json.dumps(row) for row in rows))
    candidate_path.write_text("\n".join(json.dumps({**row, "new_condition_success": 0.8}) for row in rows))
    result = accept(baseline_path, candidate_path, current_version="v1", candidate_version="v2",
                    candidate_skill_count=2, condition="C1")
    assert result["status"] == "accepted" and result["controller_version"] == "v2"


def test_collate_effect_labels_preserves_missing_skill_mask():
    first = EffectLabel(np.ones(3), 1, 2, 0, "a", "v1")
    second = EffectLabel(np.ones(3) * 2, 0, 4, 1, "b", "v1")
    embeddings, labels, masks = collate_effect_labels([first, second], ("a", "b"), embedding_dim=3)
    assert embeddings.shape == (2, 3)
    assert masks["valid"].tolist() == [[1.0, 0.0], [0.0, 1.0]]
    row = {"embedding": np.ones(3), "skills": {"a": vars(first)}}
    _, _, row_masks = collate_effect_labels([row], ("a",), embedding_dim=3)
    assert row_masks["valid"].tolist() == [[1.0]]


def test_effect_trainer_masks_missing_labels_and_round_trips(tmp_path):
    predictor = EffectPredictor(3, 2, ("a", "b"), hidden_dim=8)
    trainer = EffectTrainer(predictor, learning_rate=1e-3)
    first = EffectLabel(np.ones(3), 1, 2, 0, "a", "v1")
    second = EffectLabel(np.ones(3) * 2, 0, 4, 1, "b", "v1")
    embeddings, labels, masks = collate_effect_labels([first, second], ("a", "b"), embedding_dim=3)
    step = trainer.train_batch({"embedding": embeddings, "goal": np.zeros((2, 2), np.float32),
                                "remaining_budget": np.ones(2, np.float32),
                                "labels": labels, "masks": masks})
    assert step.valid_labels == 2 and np.isfinite(step.loss)
    path = tmp_path / "effect_trainer.pt"
    trainer.save(path)
    trainer.load(path)


def test_metric_weight_updates_trainable_effect_projection():
    predictor = EffectPredictor(3, 2, ("a",), hidden_dim=8)
    trainer = EffectTrainer(predictor, learning_rate=1e-3, metric_weight=1.0)
    label = EffectLabel(np.ones(3), 1, 1, 0, "a", "v1")
    embeddings, labels, masks = collate_effect_labels([label], ("a",), embedding_dim=3)
    before = [value.detach().clone() for value in predictor.metric_projection.parameters()]
    trainer.train_batch({"embedding": embeddings, "goal": np.zeros((1, 2), np.float32),
                         "remaining_budget": np.ones(1, np.float32), "labels": labels,
                         "masks": masks, "metric_pair": {
                             "left_embedding": np.asarray([[1., 0., 0.]], np.float32),
                             "right_embedding": np.asarray([[0., 1., 0.]], np.float32),
                             "target_distance": np.asarray([0.25], np.float32)}})
    assert any(not torch.equal(old, new) for old, new in zip(before, predictor.metric_projection.parameters()))


def test_effect_history_branch_is_independent_and_64d():
    encoder = EffectHistoryEncoder(input_dim=16, hidden_dim=8, embedding_dim=4, history_length=3)
    frames = np.zeros((2, 3, 16), np.float32)
    mask = np.asarray([[False, True, True], [True, True, True]])
    value = encoder(frames, mask)
    assert value.shape == (2, 4)
    restored = EffectHistoryEncoder.from_checkpoint(encoder.checkpoint())
    np.testing.assert_allclose(value.detach().numpy(), restored(frames, mask).detach().numpy())
    with pytest.raises(ValueError, match="right-aligned"):
        encoder(frames, np.asarray([[True, False, True], [True, True, True]]))


def test_effect_trainer_can_backpropagate_from_history_branch():
    encoder = EffectHistoryEncoder(input_dim=16, hidden_dim=8, embedding_dim=4, history_length=3)
    predictor = EffectPredictor(4, 2, ("a",), hidden_dim=8)
    trainer = EffectTrainer(predictor, learning_rate=1e-3, metric_weight=1.0,
                            effect_encoder=encoder)
    label = EffectLabel(np.ones(4), 1, 1, 0, "a", "v1")
    _, labels, masks = collate_effect_labels([label], ("a",), embedding_dim=4)
    history = {"frames": np.zeros((1, 3, 16), np.float32),
               "mask": np.asarray([[False, True, True]])}
    before = [value.detach().clone() for value in encoder.parameters()]
    trainer.train_batch({"effect_history": history, "goal": np.zeros((1, 2), np.float32),
                         "remaining_budget": np.ones(1, np.float32), "labels": labels,
                         "masks": masks, "metric_pair": {
                             "left_history": history, "right_history": history,
                             "target_distance": np.asarray([0.0], np.float32)}})
    assert any(not torch.equal(old, new) for old, new in zip(before, encoder.parameters()))
