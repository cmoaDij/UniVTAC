from evotac.perf.runtime_probe import gpu_preflight, summarize_rates


def test_wall_rates_distinguish_simulated_and_wall_time():
    report = summarize_rates([
        {"status": "executed", "physics_steps": 6, "wall_seconds": 0.5},
        {"status": "executed", "physics_steps": 6, "wall_seconds": 0.5},
        {"status": "rejected", "physics_steps": 0, "wall_seconds": 0.1},
    ])
    assert report["physics_steps"] == 12
    assert report["wall_physics_hz"] == 12.0
    assert report["wall_control_hz"] == 2.0


def test_gpu_preflight_refuses_busy_device_without_touching_cuda():
    snapshot = {"available": True, "gpus": [{"index": 2, "memory_used_mib": 900,
                                                "memory_total_mib": 1000,
                                                "utilization_gpu_percent": 99.0}]}
    result = gpu_preflight(snapshot, device="cuda:2", min_free_mib=200)
    assert not result["ok"] and result["free_mib"] == 100
    assert gpu_preflight(snapshot, device="cpu")["ok"]


def test_gpu_preflight_maps_cuda_visible_devices():
    snapshot = {"available": True, "gpus": [{"index": 8, "memory_used_mib": 100,
                                                "memory_total_mib": 1000,
                                                "utilization_gpu_percent": 1.0}]}
    result = gpu_preflight(snapshot, device="cuda:0", visible_devices="8", min_free_mib=800)
    assert result["ok"] and result["device"] == 8
