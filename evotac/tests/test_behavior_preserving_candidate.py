import pytest
import torch

from evotac.scripts.build_behavior_preserving_candidate import blend_actor_state


def test_blend_does_not_mutate_inputs_and_preserves_interpolation():
    trained = {"weight": torch.tensor([3.0, 1.0]), "counter": torch.tensor([2], dtype=torch.int64)}
    warm = {"weight": torch.tensor([1.0, 5.0]), "counter": torch.tensor([2], dtype=torch.int64)}
    result = blend_actor_state(trained, warm, 0.25)
    assert torch.equal(result["weight"], torch.tensor([1.5, 4.0]))
    assert torch.equal(result["counter"], trained["counter"])
    assert torch.equal(trained["weight"], torch.tensor([3.0, 1.0]))


def test_blend_rejects_different_nonfloating_buffers():
    with pytest.raises(ValueError, match="non-floating"):
        blend_actor_state({"x": torch.tensor([1], dtype=torch.int64)},
                          {"x": torch.tensor([2], dtype=torch.int64)}, 0.25)
