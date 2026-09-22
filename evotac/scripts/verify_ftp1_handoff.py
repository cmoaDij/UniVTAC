"""Real cached FTP-1 chunk -> scripted recovery -> fresh inference -> resumed controls."""
import json
import hashlib
from pathlib import Path
import traceback
import yaml

from evotac.config import ROOT
from evotac.data.schemas import json_value
from evotac.policy.ftp1_adapter import FTP1Policy
from evotac.policy.ftp1_client import FTP1Client
from evotac.policy.handoff import scripted_handoff
from evotac.envs.state_replay import digest
from evotac.scripts.runtime import parser_for, launch, build_wrapper


def main():
    parser = parser_for(__doc__)
    parser.add_argument("--model-gpu", default="9")
    args = parser.parse_args()
    policy_config = yaml.safe_load((ROOT/"configs/ftp1_insert_hole.yaml").read_text())
    policy_config.update(model_gpu=args.model_gpu, inference_seed=args.seed, execute_chunk_steps=16)
    config, paths, versions, launcher = launch(args)
    wrapper = backend = None
    checks = {"status": "starting", "seed": args.seed,
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (Path(__file__), ROOT/"policy/handoff.py")}}
    try:
        backend = FTP1Client(policy_config, paths["run_root"])
        policy = FTP1Policy(backend, policy_config)
        wrapper = build_wrapper(config, args.run_id, versions, args.device)
        obs, _ = wrapper.reset(args.seed)
        for _ in range(4):
            obs, _, _, _, _ = wrapper.step(policy.next_action(obs, policy_config["prompt"]))
            assert not wrapper.done
        assert len(policy.pending) == 12
        action, checks["handoff"] = scripted_handoff(wrapper, policy,
            [[0, 0, .5, 0, 0, 0, -.5]]*4, policy_config["prompt"])
        assert action is not None
        import numpy as np
        request = policy.last_inference["request_id"]
        with np.load(paths["run_root"]/f"policy/input_{request:05d}.npz") as saved:
            assert digest(dict(saved)) == checks["handoff"]["fresh_input_sha256"]
        obs, _, _, _, info = wrapper.step(action)
        checks["resumed_execution"] = info["execution_info"]
        assert info["execution_info"]["physics_start"] == checks["handoff"]["after_physics"]
        assert info["execution_info"]["physics_steps"] == 6 and not wrapper.done
        for _ in range(3):
            obs, _, _, _, _ = wrapper.step(policy.next_action(obs, policy_config["prompt"]))
            assert not wrapper.done
        checks.update(status="passed", episode_path=str(wrapper.logger.path),
                      task_elapsed=wrapper.task_elapsed, recovery_elapsed=wrapper.recovery_elapsed,
                      model=backend.metadata)
        assert wrapper.task_elapsed == 72 and wrapper.recovery_elapsed == 24
        wrapper.stop("handoff_validation_complete")
    except BaseException as exc:
        checks.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    finally:
        (paths["run_root"]/"checks.json").write_text(json.dumps(json_value(checks), indent=2))
        if backend is not None:
            backend.close()
        if wrapper is not None:
            wrapper.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
