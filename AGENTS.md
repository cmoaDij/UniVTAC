# Repository Guidelines

## Project Structure & Module Organization

UniVTAC provides visuo-tactile robotics simulation, demonstration collection, and policy evaluation.

- `envs/`: task modules, shared task logic, robot planning, sensors, and utilities. New tasks expose `TaskCfg` and `Task`; see `docs/TaskCreation.md`.
- `policy/`: policy implementations with preprocessing, training, and deployment scripts; shared interfaces live in `_base_policy.py` and `_base_data_preprocessor.py`.
- `encoder/`: tactile encoder training and data loading.
- `scripts/` and root shell scripts: installation, collection, replay, and evaluation entry points.
- `task_config/`: YAML configurations; `assets/`: robot, object, scene, and sensor resources.
- `third_party/TacEx/`: modified simulation dependencies. `docs/` contains workflow guides; `data/` and `eval_result/` hold generated outputs.

## Build, Test, and Development Commands

Run commands from the repository root. Follow `docs/Installation.md` for the Isaac Sim 5.1, Isaac Lab 2.3, Python 3.11, and CUDA 12.6 environment.

- `bash scripts/install.sh`: create/update the Conda environment and build native dependencies.
- `conda activate UniVTAC`: activate the simulation environment.
- `bash scripts/install.sh --check`: validate dependencies without reinstalling.
- `python scripts/smoke_isaac51.py --backend taxim --headless`: check cameras, tactile outputs, actor poses, and render synchronization.
- `bash collect_data.sh lift_can demo 0`: collect demonstrations using `task_config/demo.yml` on GPU 0.
- `bash eval_policy.sh <task> <task_config> <policy_config> <gpu_id>`: evaluate a configured policy; see `docs/Deploy.md`.

## Coding Style & Naming Conventions

Use four-space Python indentation, snake_case functions and variables, and PascalCase classes. Match surrounding code and preserve task module names used by launchers. Keep runtime settings in YAML and reuse shared interfaces. No root-wide formatter is configured; SmolVLA declares Ruff in `policy/smolvla/pyproject.toml`.

## Testing Guidelines

The main integration check is the standalone smoke script; no root-wide coverage threshold is configured. Test both `taxim` and `pix2pix` when changing tactile backends. For task changes, run collection with `--max_seed 0 --config-overrides collect_settings.episode_num=1`. Record the command, seed, backend, and outcome. Simulation checks require the configured NVIDIA GPU environment.

## Commit & Pull Request Guidelines

History uses short imperative subjects such as `update readme` and `Restore TacEx assets as non-LFS files`; no mandatory prefix is evident. Use a specific action and component. PRs should explain behavior changes, link relevant issues, list validation commands/results, and include videos or screenshots for visual simulation changes. Keep generated datasets, logs, and checkpoints out of commits. Preserve vendored TacEx modifications rather than replacing them with public packages.
