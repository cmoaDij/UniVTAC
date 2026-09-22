"""Import this module freely; create_task must run after AppLauncher."""
from copy import deepcopy
from evotac.config import run_paths, validate_config


def create_task(config, run_id, device="cuda:0"):
    config = validate_config(config)
    # Deliberately local imports: loading YAML/contracts must not import Isaac.
    from evotac.envs.tasks.insert_hole import EvoTacInsertHoleTask, TaskCfg
    cfg = TaskCfg()
    cfg.seed = 0  # construction RNG; each attempt reseeds explicitly before reset
    paths = run_paths(config, run_id)
    cfg.save_dir = str(paths["run_root"] / "simulator")
    cfg.scene.num_envs = 1
    cfg.sim.device = device
    cfg.sim.dt = 1 / config["simulation"]["physical_hz"]
    cfg.uipc_sim.dt = cfg.sim.dt
    cfg.reset_time_limit = config["budgets"]["reset_wall_seconds"]
    decimation = int(config["simulation"]["physical_hz"]) // int(config["simulation"]["control_hz"])
    cfg.decimation = decimation
    cfg.sim.render_interval = decimation
    cfg.save_frequency = cfg.video_frequency = cfg.render_frequency = 0
    cfg.arm_stiffness = config["controller"]["arm_stiffness"]
    cfg.arm_damping = config["controller"]["arm_damping"]
    cfg.tactile_sensor_type = "gsmini"
    cfg.tactile_optical_backend = "taxim"
    cfg.camera_names = list(config["observation"]["cameras"])
    cfg.obs_data_type = {"camera": ["rgb"], "tactile": ["rgb", "rgb_marker"],
                         "embodiment": ["joint", "ee"], "actor": True}
    return EvoTacInsertHoleTask(cfg, evotac_config=deepcopy(config), mode="eval")
