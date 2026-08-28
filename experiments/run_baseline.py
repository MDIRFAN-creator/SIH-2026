"""
Phase 2 Experiment Runner: Open-Loop Baseline Benchmark (SIH26055).

Runs:
1. Single baseline episode on default multi-threat RF scenario.
2. Dwell sensitivity analysis across fixed dwell durations [1, 2, 3, 5].
3. 10-seed statistical evaluation (reporting mean +/- std).
4. Generates visual baseline scanning plot (baseline_sweep_demo.png).
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List
import numpy as np

from environment import EnvironmentConfig, RFEnvironment, load_config
from evaluation import (
    BaselineMetrics,
    aggregate_metrics_across_seeds,
    calculate_baseline_metrics,
)
from runners import EpisodeResult, EpisodeRunner
from schedulers import OpenLoopScheduler
from visualization import plot_baseline_scan_pattern


def run_baseline_benchmark() -> None:
    print("=" * 80)
    print("SIH26055 — Phase 2: Open-Loop Baseline Benchmark Execution")
    print("=" * 80)

    config_path = Path("configs/default.yaml")
    config = load_config(config_path)
    runner = EpisodeRunner()

    # -------------------------------------------------------------
    # 1. Single Episode Benchmark (Dwell = 1)
    # -------------------------------------------------------------
    print("\n[1] Running Single Episode Baseline (Dwell = 1 slot, Seed = 42)...")
    env = RFEnvironment(config)
    scheduler_d1 = OpenLoopScheduler(num_bands=config.num_bands, dwell_time=1)
    
    result_d1 = runner.run_episode(env=env, scheduler=scheduler_d1, seed=42)
    metrics_d1 = calculate_baseline_metrics(result_d1, env.emitter_registry)

    print(f"    - Simulation Horizon: {metrics_d1.total_simulation_slots} slots")
    print(f"    - Scheduling Decisions: {metrics_d1.total_decisions}")
    print(f"    - Ground-Truth Interception Opportunities: {metrics_d1.interception_opportunities}")
    print(f"    - Successful Interceptions: {metrics_d1.successful_interceptions}")
    print(f"    - Scheduler Interception Rate: {metrics_d1.interception_rate * 100:.2f}%")
    print(f"    - Receiver Empirical Pd: {metrics_d1.empirical_pd:.4f} (Target: 0.9000)")
    print(f"    - Receiver Empirical Pfa: {metrics_d1.empirical_pfa:.4f} (Target: 0.0200)")
    print(f"    - Dwell Efficiency: {metrics_d1.dwell_efficiency * 100:.2f}%")
    if metrics_d1.average_intercept_time is not None:
        print(f"    - PRD Average Intercept Time: {metrics_d1.average_intercept_time:.2f} slots")
    print(f"    - PRD Scenario TTFD: {metrics_d1.scenario_ttfd} slots")
    print(f"    - Per-Emitter TTFD:")
    for eid, ttfd in metrics_d1.emitter_ttfd.items():
        print(f"      * {eid}: {ttfd} slots")

    # -------------------------------------------------------------
    # 2. Dwell Comparison [1, 2, 3, 5]
    # -------------------------------------------------------------
    print("\n[2] Running Dwell Comparison Across [1, 2, 3, 5] slots...")
    dwell_options = [1, 2, 3, 5]
    dwell_metrics = []

    print(f"    {'Dwell':<8} | {'Decisions':<10} | {'Interceptions':<15} | {'Intercept Rate':<16} | {'Avg Delay':<12} | {'Efficiency':<12}")
    print("    " + "-" * 78)

    for d in dwell_options:
        env_d = RFEnvironment(config)
        sched = OpenLoopScheduler(num_bands=config.num_bands, dwell_time=d)
        res = runner.run_episode(env=env_d, scheduler=sched, seed=42)
        m = calculate_baseline_metrics(res, env_d.emitter_registry)
        dwell_metrics.append(m)
        delay_str = f"{m.average_intercept_delay:.2f}" if m.average_intercept_delay is not None else "N/A"
        print(
            f"    {d:<8} | {m.total_decisions:<10} | {f'{m.successful_interceptions}/{m.interception_opportunities}':<15} | "
            f"{m.interception_rate * 100:>6.2f}%         | {delay_str:<12} | {m.dwell_efficiency * 100:>6.2f}%"
        )

    # -------------------------------------------------------------
    # 3. 10-Seed Statistical Evaluation (Dwell = 1)
    # -------------------------------------------------------------
    print("\n[3] Running 10-Seed Statistical Evaluation (Seeds 0..9)...")
    seeds = list(range(10))
    multi_seed_metrics: List[BaselineMetrics] = []

    for s in seeds:
        env_s = RFEnvironment(config)
        sched_s = OpenLoopScheduler(num_bands=config.num_bands, dwell_time=1)
        res_s = runner.run_episode(env=env_s, scheduler=sched_s, seed=s)
        m_s = calculate_baseline_metrics(res_s, env_s.emitter_registry)
        multi_seed_metrics.append(m_s)

    aggregates = aggregate_metrics_across_seeds(multi_seed_metrics)

    print(f"    Aggregated Metrics (N = 10 seeds):")
    print(f"    - Interception Rate:      {aggregates['interception_rate'][0] * 100:.2f}% ± {aggregates['interception_rate'][1] * 100:.2f}%")
    print(f"    - Empirical Pd:           {aggregates['empirical_pd'][0]:.4f} ± {aggregates['empirical_pd'][1]:.4f}")
    print(f"    - Empirical Pfa:          {aggregates['empirical_pfa'][0]:.4f} ± {aggregates['empirical_pfa'][1]:.4f}")
    print(f"    - Dwell Efficiency:       {aggregates['dwell_efficiency'][0] * 100:.2f}% ± {aggregates['dwell_efficiency'][1] * 100:.2f}%")
    print(f"    - Avg Intercept Delay:    {aggregates['average_intercept_delay'][0]:.2f} ± {aggregates['average_intercept_delay'][1]:.2f} slots")
    print(f"    - Total Action Hits:      {aggregates['action_hits'][0]:.1f} ± {aggregates['action_hits'][1]:.1f}")
    print(f"    - Total Action False Alm: {aggregates['action_false_alarms'][0]:.1f} ± {aggregates['action_false_alarms'][1]:.1f}")

    # -------------------------------------------------------------
    # 4. Generate Baseline Visualization Plot
    # -------------------------------------------------------------
    print("\n[4] Generating Baseline Sweep Visualization Plot...")
    plot_path = Path("baseline_sweep_demo.png")
    plot_baseline_scan_pattern(
        episode_result=result_d1,
        time_range=(0, 200),
        save_path=str(plot_path),
        title="SIH26055: Open-Loop Baseline Frequency Sweep Pattern (t=0..200, Dwell=1)",
    )
    print(f"    -> Plot saved successfully to: {plot_path.resolve()}")

    print("\n" + "=" * 80)
    print("Phase 2 Open-Loop Baseline Benchmark Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    run_baseline_benchmark()
