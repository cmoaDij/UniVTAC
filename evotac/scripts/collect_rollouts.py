"""Run every explicit scene attempt, retaining failures and their denominator."""
import json
import traceback

from evotac.data.schemas import Action
from evotac.scripts.runtime import parser_for, launch, build_wrapper


def main():
    parser = parser_for(__doc__)
    parser.add_argument("--scene-manifest", required=True)
    args = parser.parse_args()
    # Validate the entire scene list before starting expensive simulation.
    with open(args.scene_manifest) as handle:
        scenes = [json.loads(line) for line in handle if line.strip()]
    splits = {}
    for scene in scenes:
        parent, split = scene["parent_scene_id"], scene["split"]
        if splits.setdefault(parent, split) != split:
            raise ValueError("Parent scene split conflict")
        for action in scene["actions"]:
            Action(**action)
    config, paths, versions, launcher = launch(args)
    wrapper = None
    outcomes = []
    try:
        wrapper = build_wrapper(config, args.run_id, versions, args.device)
        for scene in scenes:
            try:
                wrapper.reset(scene["seed"], scene.get("condition"), parent_scene_id=scene["parent_scene_id"], split=scene["split"], branch_id=scene.get("branch_id", "base"))
                for action in scene["actions"]:
                    wrapper.step(Action(**action))
                    if wrapper.done:
                        break
                if not wrapper.done:
                    wrapper.stop("script_exhausted")
                outcomes.append({"parent_scene_id": scene["parent_scene_id"], "episode_id": wrapper.scene_metadata["episode_id"], "outcome": json.loads(wrapper.logger.handle.attrs["outcome"])})
            except BaseException as exc:
                outcomes.append({"parent_scene_id": scene["parent_scene_id"], "error": f"{type(exc).__name__}: {exc}"})
                traceback.print_exc()
                if not isinstance(exc, Exception):
                    raise
                if config["budgets"]["stop_on_exception"]:
                    break
    finally:
        (paths["run_root"] / "checks.json").write_text(json.dumps({"attempted": len(outcomes), "listed": len(scenes), "outcomes": outcomes}, indent=2))
        if wrapper:
            wrapper.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
