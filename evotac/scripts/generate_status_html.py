"""Generate the consolidated EvoTac status page from checked-in evidence."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE_DOCS = (
    ("P0", "P0_Baseline.md", "版本、配置与速度基线"),
    ("P1", "P1_Contracts.md", "控制、观测与数据契约"),
    ("P2", "P2_FTP1_and_Handoff.md", "FTP-1 基线、起点重建与恢复交接"),
    ("P3", "P3_Recovery_SAC.md", "单恢复技能 SAC"),
    ("P4", "P4_Effect_Selection.md", "恢复效果表征、技能选择与经验复用"),
    ("P5", "P5_Continual_Adaptation.md", "持续适应与有限技能扩展"),
    ("P6", "P6_Ablations.md", "最小充分消融实验"),
)


def load_json(relative):
    path = ROOT / relative
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def latest_profile(launch_mode=None):
    candidates = sorted((ROOT / "runs").glob("phase1/**/phase1_profile_*/checks.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            profile = json.loads(path.read_text())
            if profile.get("status") != "profile_complete" or not profile.get("executor_rates"):
                continue
            if launch_mode and profile.get("launch_mode") != launch_mode:
                continue
            return profile, path.parent.name
        except (OSError, json.JSONDecodeError):
            continue
    return None, None


def page(output):
    comparison = load_json("runs/phase2/phase2_comparison_final_v1/comparison.json") or {}
    diagnosis = load_json("runs/phase2/phase2_chunk16_replay_diagnosis_v2/phase2_chunk16_replay_phys_seed0_diagnosis/report.json") or {}
    repeat_checks = load_json("runs/phase1/phase2_chunk16_replay_samegpu_seed47_v3/checks.json") or {}
    repeat_items = repeat_checks.get("current_repeats") or []
    repeat_item = repeat_items[0] if repeat_items else {}
    repeat_match = (repeat_item.get("same_code_as_reference") or {}).get("exact_match")
    markerfix = load_json("runs/phase2/phase2_markerfix_seed47_repeat_report.json") or {}
    noiseoff = load_json("runs/phase1/phase2_markerfix_noiseoff_replay_seed47_v1/checks.json") or {}
    noiseoff_reconstruction = noiseoff.get("reconstruction") or {}
    noiseoff_repeat = (noiseoff.get("current_repeats") or [{}])[0]
    noiseoff_repeat_match = (noiseoff_repeat.get("same_code_as_reference") or {}).get("exact_match")
    candidate = load_json("runs/phase1/phase2_markerfix_noiseoff_replay_seed47_headless_candidate_v3/independent_tolerance_validation.json") or {}
    candidate_source = load_json("runs/phase1/phase2_headless_candidate_source_seed23_v2/checks.json") or {}
    counterfactual = load_json("runs/phase1/phase2_headless_counterfactual_seed23_v3/checks.json") or {}
    counterfactual_full = load_json("runs/phase1/phase2_headless_counterfactual_seed23_full_v2/checks.json") or {}
    current_replay = load_json("runs/phase1/phase2_current_standalone_replay_seed23_20260921/checks.json") or {}
    current_reconstruction = current_replay.get("reconstruction") or {}
    current_paired = load_json("runs/phase1/phase3_paired_current_seed23_full_20260921/checks.json") or {}
    current_branches = current_paired.get("branches") or []
    current_calibration = load_json("runs/phase1/calibration_headless_current_v2_20260921/checks.json") or {}
    actual_transition = load_json("runs/phase1/phase3_actual_transition_seed23_20260921/checks.json") or {}
    actual_episodes = actual_transition.get("episodes") or []
    actual_episode = actual_episodes[0] if actual_episodes else {}
    train_transition = load_json("runs/phase1/phase3_train_transition_seed1_20260921/checks.json") or {}
    train_episodes = train_transition.get("episodes") or []
    train_episode = train_episodes[0] if train_episodes else {}
    current_replay_text = (
        f"seed23 standalone strict={current_reconstruction.get('valid_match', '暂无')}"
        if current_reconstruction else "seed23 standalone strict=暂无"
    )
    current_pair_text = f"paired_valid={current_paired.get('paired_valid', '暂无')}"
    calibration_text = "4 parents/8 repeats；seed18/24 漂移，诊断容差不冻结"
    scene_benchmark = load_json("runs/phase2/scene_build_benchmark.json") or {}
    scene_build_mean = scene_benchmark.get("mean_seconds")
    rates = {"wall_physics_hz": 21.72, "wall_control_hz": 3.62, "wall_seconds": 442.07,
             "physics_steps": 9600, "controls": 1600}
    profile, profile_id = latest_profile("headless_profile")
    livestream_profile, livestream_id = latest_profile("livestream")
    if profile and profile.get("executor_rates"):
        rates.update({"wall_physics_hz": profile["executor_rates"].get("wall_physics_hz"),
                      "wall_control_hz": profile["executor_rates"].get("wall_control_hz"),
                      "wall_seconds": profile["executor_rates"].get("wall_seconds"),
                      "physics_steps": profile["executor_rates"].get("physics_steps"),
                      "controls": profile["executor_rates"].get("controls")})
    livestream_rates = (livestream_profile or {}).get("executor_rates", {})
    headless_e2e = (profile or {}).get("end_to_end_control_hz")
    livestream_text = (f"livestream {livestream_rates.get('wall_physics_hz', 0):.3f} physics Hz / "
                       f"{livestream_rates.get('wall_control_hz', 0):.3f} control Hz"
                       if livestream_profile else "livestream profile 尚未完成")
    replay_match = diagnosis.get("valid_match", diagnosis.get("reconstruction_flags", {}).get("valid_match", False))
    rows = [
        ("基础策略", "FTP-1 chunk16", "6/10 成功；3 丢失；1 预算耗尽"),
        ("重放起点", "seed23 当前代码 strict 通过；多 seed cohort 未冻结", f"{current_replay_text}；observable={current_reconstruction.get('observable_match', '暂无')}、versions={current_reconstruction.get('versions_match', '暂无')}；旧 v3 仍冻结；{calibration_text}"),
        ("恢复交接", "seed23 完整 paired terminal audit 通过", f"{current_pair_text}；no-recovery/scripted-recovery 均 strict prefix match，分别 143/161 continuation controls；两 branch 都 object_lost，恢复收益尚未证明"),
        ("P3 单技能 SAC", "真实 transition + SAC plumbing 更新通过，效果未证明", f"GPU8 controlled-trigger seed23: recovery_actions={actual_episode.get('recovery_actions', '暂无')}、admitted={actual_episode.get('admitted_transitions', '暂无')}、updates={len(actual_episode.get('updates', []))}；train-split seed1: recovery_actions={train_episode.get('recovery_actions', '暂无')}、updates={len(train_episode.get('updates', []))}；使用显式触发和 batch1，不能外推恢复收益"),
        ("P4 效果选择", "CPU 初始骨架", "版本化效果标签、mask、训练器、检索和选择器已实现；没有真实效果标签"),
        ("P5 持续适应", "CPU 初始骨架", "候选 controller、接纳 gate 和 50/25/25 采样已实现；没有真实候选仿真结果"),
        ("P6 最小实验", "协议 + surrogate smoke", "每个 condition 每方法 100 个父场景；三组完成 4,800/6,000/4,800 个 surrogate episode，独立 audit 通过；真实物理实验等待 P2–P5 验收"),
        ("性能", "两种启动模式初步测速", f"headless {rates['wall_physics_hz']:.3f}/{rates['wall_control_hz']:.3f} Hz；{livestream_text}；各 4 controls/24 physics steps；scene binary hash 平均 {scene_build_mean:.2f} s，reset 状态未严格配对" if scene_build_mean else f"headless {rates['wall_physics_hz']:.3f}/{rates['wall_control_hz']:.3f} Hz；{livestream_text}；各 4 controls/24 physics steps，reset 状态未严格配对"),
        ("快速 surrogate", "接口 smoke", "24 个 episode、约 285 episode/s；只验证控制/消融管线，不替代 Isaac/TacEx 物理"),
    ]
    table = "\n".join(
        f"<tr><th>{html.escape(a)}</th><td>{html.escape(b)}</td><td>{html.escape(c)}</td></tr>"
        for a, b, c in rows
    )
    comparison_text = html.escape(json.dumps(comparison, ensure_ascii=False)[:800])
    stage_links = " ".join(f'<a href="stages/{html.escape(filename)}">{html.escape(code)} {html.escape(title)}</a>'
                           for code, filename, title in STAGE_DOCS)
    stage_sections = "".join(
        f'<details><summary>{html.escape(code)}：{html.escape(title)}</summary>'
        f'<pre class="stage-doc">{html.escape((ROOT / "docs/stages" / filename).read_text(encoding="utf-8"))}</pre></details>'
        for code, filename, title in STAGE_DOCS
        if (ROOT / "docs/stages" / filename).exists()
    )
    profile_note = (f"最近的真实 profile：<code>{html.escape(profile_id)}</code>，"
                    f"执行段 {rates['wall_physics_hz']:.3f} physics Hz / {rates['wall_control_hz']:.3f} control Hz；结果仍需按 GPU 占用和启动模式解释。"
                    if profile else "尚未找到新的真实 profile；页面沿用已保存的 chunk16 执行段统计。")
    body = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EvoTac / UniVTAC 5.1 实现汇总</title>
<style>
:root {{ color-scheme: light; --ink:#17212b; --muted:#566573; --line:#d8e0e7; --blue:#1464a5; --bg:#f5f7fa; --card:#fff; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--ink); font:16px/1.65 system-ui,-apple-system,"Noto Sans SC",sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:34px 22px 60px; }} h1 {{ margin:0 0 5px; font-size:2rem; }} h2 {{ margin-top:34px; font-size:1.35rem; }} p {{ max-width:900px; }} .muted {{ color:var(--muted); }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin:24px 0; }} .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; }} .value {{ display:block; color:var(--blue); font-size:1.65rem; font-weight:700; }}
table {{ width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line); }} th,td {{ text-align:left; padding:11px 12px; border-bottom:1px solid var(--line); vertical-align:top; }} th {{ width:22%; }}
code {{ background:#e9eff4; padding:2px 5px; border-radius:4px; }} a {{ color:var(--blue); }} .note {{ border-left:4px solid var(--blue); padding:10px 14px; background:#eaf3fb; }}
.stage-doc {{ white-space:pre-wrap; background:#fbfcfd; border:1px solid var(--line); border-radius:8px; padding:14px; max-height:520px; overflow:auto; }} details {{ margin:10px 0; background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 14px; }}
</style></head><body><main>
<h1>EvoTac / UniVTAC 5.1 实现汇总</h1>
<p class="muted">更新时间：2026-09-21 · 仓库分支：<code>isaac51</code> · Isaac Sim 5.1.0 · Isaac Lab 2.3.0</p>
<div class="cards">
<div class="card"><span class="value">6/10</span>FTP-1 chunk16 成功</div>
<div class="card"><span class="value">3</span>真实 recovery transition</div>
<div class="card"><span class="value">{rates['wall_physics_hz']:.2f}</span>headless 墙钟物理步/秒</div>
<div class="card"><span class="value">{livestream_rates.get('wall_physics_hz', 0):.2f}</span>livestream 墙钟物理步/秒</div>
</div>
<h2>实现状态</h2><table><thead><tr><th>模块</th><th>状态</th><th>证据与边界</th></tr></thead><tbody>{table}</tbody></table>
<h2>为什么当前 FPS 低</h2>
<p>历史 chunk16 测量为 21.72 wall physics Hz；同一物理 GPU、seed 和传感器配置下，headless 执行段为 {rates['wall_physics_hz']:.3f} physics Hz / {rates['wall_control_hz']:.3f} control Hz，livestream 执行段为 {livestream_rates.get('wall_physics_hz', 0):.3f} / {livestream_rates.get('wall_control_hz', 0):.3f}。headless 控制循环（含观测和落盘，不含 reset）约 {(headless_e2e or 0):.3f} control Hz；这些都不是策略 FPS，也不是配置的仿真频率。配置仍是 120 Hz 物理、20 Hz 控制，每次控制推进 6 个物理步；两条 profile 各覆盖 {rates['controls']} 个控制周期。</p>
<p>即使 GPU 空闲，本工作负载仍包含两路相机、双 GelSight Mini 的 RGB/marker 和 taxim/TacEx 接触更新。headless 可以去掉交互视口和 livestream 开销，但相机传感器与触觉计算仍会执行；关闭传感器只能得到性能上界，不能拿来比较策略效果。</p>
<p>重放路径还对大触觉数组使用二进制规范化哈希；最近 seed 47 chunk16 场景构建平均 {f"{scene_build_mean:.2f}" if scene_build_mean else "未测得"} 秒，避免 JSON 展开成为诊断瓶颈。</p>
<p>作为负载量级参考，TacEx 论文在单环境触觉实验中报告 height-map、GPU Taxim、CPU FOTS 约 1.37、5.90、4.49 ms/frame；GIPC 软体物理则从 24.95 ms/frame（1,029 vertices）到 221.61 ms/frame（12,509 vertices）。这些是论文环境的测量，不是本机 5.1 的 FPS。</p>
<div class="note">两种启动模式均完成初步测速。两条 profile 使用 <code>CUDA_VISIBLE_DEVICES=8</code>、seed 0，保持相机/触觉开启；reset 终点分别为 526 和 515，未严格配对隐藏状态。吞吐差异不能全部归因于 livestream，后续需固定重放起点并增加控制周期。</div>
<p>{profile_note}</p>
<p>最近的 livestream profile：<code>{html.escape(livestream_id or '暂无')}</code>，{html.escape(livestream_text)}。直接使用 <code>cuda:8</code> 的启动失败仍保留在 <code>phase1_profile_gpu8_real_20260920</code>；映射为逻辑 <code>cuda:0</code> 后两种启动模式均完成 4 个真实控制周期，全部动作提交和观测记录有效。该短 profile 不能替代策略效果验证。</p>
<h2>下一步顺序</h2>
<ol><li>隔离 seed18/24 的 UIPC/FEM/触觉 reset 漂移，并保持 v3 容差冻结。</li><li>在稳定 strict cohort 后扩大多 seed paired continuation；当前 seed23 full run 只作为 wiring/terminal evidence。</li><li>把 plumbing 触发替换为校准监测器，用冻结 HistoryEncoder 和标准 batch 扩大真实 SAC 训练。</li><li>冻结 P3 技能后生成 P4 效果标签；P5 接纳实验通过后做 P6 三组消融。</li></ol>
<h2>资料与证据</h2>
<ul><li><a href="https://github.com/univtac/UniVTAC">UniVTAC 官方仓库</a>：isaac51 分支、5.1 数据集和相对 4.5 的吞吐说明。</li>
<li><a href="https://isaac-sim.github.io/IsaacLab/develop/source/overview/core-concepts/sensors/camera.html">Isaac Lab Camera</a>：相机由渲染器驱动，图像带宽和分辨率/相机数量会影响性能。</li>
<li><a href="https://docs.isaacsim.omniverse.nvidia.com/5.1.0/reference_material/sim_performance_optimization_handbook.html">Isaac Sim 5.1 性能手册</a>：headless 仍需显式处理 viewport 更新。</li>
<li><a href="https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html">Isaac Sim livestream 文档</a>：WebRTC 是 headless 实例的远程交互方式。</li>
<li><a href="https://arxiv.org/pdf/2411.04776">TacEx 论文</a>：公开的 tactile/FEM 分项耗时参考。</li></ul>
<h2>各阶段完整记录</h2>
{stage_sections}
<details><summary>已读取的对照汇总 JSON 片段</summary><pre>{comparison_text}</pre></details>
<p class="muted">完整阶段计划：<a href="EvoTac_Phase_Implementation_Plan.md">EvoTac_Phase_Implementation_Plan.md</a>；实现状态：<a href="EvoTac_Implementation_Status.md">EvoTac_Implementation_Status.md</a>。</p>
<p class="muted">阶段记录：{stage_links}。</p>
<p class="muted">性能拆解与 headless A/B：<a href="Performance_Analysis_and_Optimization.md">Performance_Analysis_and_Optimization.md</a>。</p>
</main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/EvoTac_Summary.html")
    args = parser.parse_args()
    page(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
