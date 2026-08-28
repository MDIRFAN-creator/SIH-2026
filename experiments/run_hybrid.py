"""
SIH26055 — Phase 6: Isolated Hybrid Adaptive RF Scheduler Benchmark Pipeline.

Evaluates 6 scheduling paradigms head-to-head under identical RF conditions:
1. Open-Loop Baseline (Conventional periodic round-robin scanning)
2. XGBoost + Optimization (Phase 3: Supervised active band prediction)
3. Hardened LinUCB Contextual Bandit (Phase 4: Discounted UCB with anti-camping)
4. Original PPO Baseline (Phase 5: Baseline Reinforcement Learning)
5. Hardened PPO Policy (Pre-Phase 6A: Anti-camping action-masked Reinforcement Learning)
6. Hybrid Adaptive Scheduler (Phase 6: Multi-paradigm arbitration architecture)

Strict Non-Leakage Guarantee:
- Evaluated strictly on identical unseen test seeds 0..9 (Horizon = 10,000 slots).
- Zero ground truth provided to any scheduler during action selection.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment import EnvironmentConfig, RFEnvironment, load_config
from environment.types import Action, DetectionResult
from evaluation import (
    BaselineMetrics,
    aggregate_metrics_across_seeds,
    calculate_baseline_metrics,
    extract_emitter_opportunities,
)
from hybrid import HybridConfig
from models import XGBoostBandPredictor
from optimizers import ActionOptimizer
from rl import ActionEncoder, PPOAgent, RLStateExtractor
from runners import EpisodeResult, EpisodeRunner
from schedulers import (
    HybridAdaptiveScheduler,
    LinUCBScheduler,
    OpenLoopScheduler,
    PPOScheduler,
    XGBoostScheduler,
)
from training import train_xgboost_pipeline
from visualization import (
    plot_6way_benchmark_comparison,
    plot_6way_frequency_hopping_adaptation,
    plot_6way_trajectories,
    plot_hybrid_arbitration_diagnostics,
    plot_hybrid_exploration_exploitation,
)


def run_6way_frequency_hopping_adaptation_experiment(
    trained_xgb_model: XGBoostBandPredictor,
    orig_ppo_agent: PPOAgent,
    hardened_ppo_agent: PPOAgent,
    num_bands: int = 20,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """
    Evaluate rapid discovery of sudden frequency hops across all 6 schedulers.
    """
    print("\n" + "=" * 110)
    print("PHASE 6: 6-WAY DYNAMIC FREQUENCY-HOPPING ADAPTATION EXPERIMENT")
    print("=" * 110)

    hop_scenarios = [
        {"name": "Scenario 1: Hop to Band 14", "hop_time": 1000, "orig_band": 3, "dest_band": 14, "duration": 2500},
        {"name": "Scenario 2: Hop to Band 7", "hop_time": 2000, "orig_band": 12, "dest_band": 7, "duration": 3500},
        {"name": "Scenario 3: Hop to Band 18", "hop_time": 3000, "orig_band": 5, "dest_band": 18, "duration": 4500},
    ]

    schedulers_factory = {
        "Open-Loop Baseline": lambda: OpenLoopScheduler(num_bands=num_bands, dwell_time=1),
        "XGBoost Adaptive": lambda: XGBoostScheduler(
            model=trained_xgb_model,
            num_bands=num_bands,
            optimizer=ActionOptimizer(num_bands=num_bands, max_consecutive_scans=3),
        ),
        "Hardened LinUCB": lambda: LinUCBScheduler(
            num_bands=num_bands,
            alpha=1.0,
            gamma=0.99,
            max_consecutive_scans=3,
            min_initial_pulls=1,
            seed=seed,
        ),
        "Original PPO Baseline": lambda: PPOScheduler(
            agent=orig_ppo_agent,
            num_bands=num_bands,
            max_consecutive_scans=5000,
            deterministic=True,
            scheduler_name="OriginalPPOBaseline",
        ),
        "Hardened PPO": lambda: PPOScheduler(
            agent=hardened_ppo_agent,
            num_bands=num_bands,
            max_consecutive_scans=3,
            deterministic=True,
            scheduler_name="HardenedPPO",
        ),
        "Hybrid Adaptive Scheduler": lambda: HybridAdaptiveScheduler(
            config=HybridConfig(num_bands=num_bands, max_consecutive_scans=3, seed=seed),
            xgb_model=trained_xgb_model,
            ppo_agent=hardened_ppo_agent,
        ),
    }

    runner = EpisodeRunner()
    results: Dict[str, List[float]] = {name: [] for name in schedulers_factory}

    for scen in hop_scenarios:
        hop_t = scen["hop_time"]
        orig_b = scen["orig_band"]
        dest_b = scen["dest_band"]
        dur = scen["duration"]

        print(f"\n[*] Evaluating {scen['name']}: Emitter hops from Band {orig_b} -> Band {dest_b} at t = {hop_t} slots (Horizon = {dur})...")

        env_config = EnvironmentConfig(
            num_bands=num_bands,
            simulation_duration=dur,
            emitters=[
                {
                    "emitter_id": "hopping_threat",
                    "emitter_type": "AGILE_PREDICTABLE",
                    "band_sequence": [orig_b, dest_b],
                    "hop_period": hop_t,
                },
                {
                    "emitter_id": "background_periodic_1",
                    "emitter_type": "PERIODIC",
                    "frequency_band": 0,
                    "period": 25,
                    "active_duration": 4,
                },
            ],
        )

        for s_name, s_init in schedulers_factory.items():
            env = RFEnvironment(config=env_config)
            scheduler = s_init()
            res = runner.run_episode(env=env, scheduler=scheduler, seed=seed)

            # Find first true positive on dest_b after hop_t
            first_detection_delay = 9999.0
            for record in res.step_records:
                if record.start_time >= hop_t and record.action.frequency_band == dest_b:
                    if record.observation.result == DetectionResult.HIT:
                        first_detection_delay = float(record.start_time - hop_t)
                        break

            results[s_name].append(first_detection_delay)
            delay_str = f"{first_detection_delay:.0f} slots" if first_detection_delay < 9999.0 else "NOT ACQUIRED (>1000 slots)"
            print(f"    - {s_name:<30}: First Intercept Delay = {delay_str}")

    return results


def run_6way_benchmark(
    config: EnvironmentConfig,
    trained_xgb_model: XGBoostBandPredictor,
    orig_ppo_agent: PPOAgent,
    hardened_ppo_agent: PPOAgent,
    test_seeds: List[int],
) -> Tuple[Dict[str, Dict[str, Tuple[float, float]]], Dict[str, EpisodeResult], List[Any]]:
    """
    Execute 6-way head-to-head benchmark over 10 unseen test seeds (0..9).
    """
    print("\n" + "=" * 110)
    print("PHASE 6: 6-WAY HEAD-TO-HEAD BENCHMARK (10 UNSEEN SEEDS 0..9, HORIZON = 10,000 SLOTS)")
    print("=" * 110)

    num_bands = config.num_bands
    schedulers_factory = {
        "Open-Loop Baseline": lambda s: OpenLoopScheduler(num_bands=num_bands, dwell_time=1),
        "XGBoost Adaptive": lambda s: XGBoostScheduler(
            model=trained_xgb_model,
            num_bands=num_bands,
            optimizer=ActionOptimizer(num_bands=num_bands, max_consecutive_scans=3),
        ),
        "Hardened LinUCB": lambda s: LinUCBScheduler(
            num_bands=num_bands,
            alpha=1.0,
            gamma=0.99,
            max_consecutive_scans=3,
            min_initial_pulls=1,
            seed=s,
        ),
        "Original PPO Baseline": lambda s: PPOScheduler(
            agent=orig_ppo_agent,
            num_bands=num_bands,
            max_consecutive_scans=5000,
            deterministic=True,
            scheduler_name="OriginalPPOBaseline",
        ),
        "Hardened PPO": lambda s: PPOScheduler(
            agent=hardened_ppo_agent,
            num_bands=num_bands,
            max_consecutive_scans=3,
            deterministic=True,
            scheduler_name="HardenedPPO",
        ),
        "Hybrid Adaptive Scheduler": lambda s: HybridAdaptiveScheduler(
            config=HybridConfig(num_bands=num_bands, max_consecutive_scans=3, seed=s),
            xgb_model=trained_xgb_model,
            ppo_agent=hardened_ppo_agent,
        ),
    }

    runner = EpisodeRunner()
    all_metrics: Dict[str, List[BaselineMetrics]] = {name: [] for name in schedulers_factory}
    all_telemetry: Dict[str, List[Dict[str, Any]]] = {name: [] for name in schedulers_factory}
    sample_episodes: Dict[str, EpisodeResult] = {}
    sample_hybrid_logs: List[Any] = []

    for s_idx, s in enumerate(test_seeds):
        print(f"\n[*] Executing Test Seed {s} ({s_idx + 1}/{len(test_seeds)})...")
        for s_name, s_fn in schedulers_factory.items():
            env = RFEnvironment(config=config)
            scheduler = s_fn(s)

            res = runner.run_episode(env=env, scheduler=scheduler, seed=s)
            metrics = calculate_baseline_metrics(res, emitter_registry=env.emitter_registry)
            all_metrics[s_name].append(metrics)

            # Telemetry diagnostics
            scanned_bands = set()
            max_consec = 1
            curr_consec = 1
            last_b = None
            consec_runs = []
            scans_hit_bands = 0
            scans_unhit_bands = 0
            hit_bands_seen = set()

            band_counts = np.zeros(num_bands, dtype=np.int64)
            for r in res.step_records:
                b = r.action.frequency_band
                band_counts[b] += 1
                scanned_bands.add(b)

                if r.observation.result == DetectionResult.HIT:
                    hit_bands_seen.add(b)

                if b in hit_bands_seen:
                    scans_hit_bands += 1
                else:
                    scans_unhit_bands += 1

                if b == last_b:
                    curr_consec += 1
                else:
                    if last_b is not None:
                        consec_runs.append(curr_consec)
                    curr_consec = 1
                    last_b = b
                if curr_consec > max_consec:
                    max_consec = curr_consec

            if last_b is not None:
                consec_runs.append(curr_consec)

            mean_run = float(np.mean(consec_runs)) if consec_runs else 0.0

            # Shannon Entropy
            probs = band_counts / max(1, np.sum(band_counts))
            non_zero = probs[probs > 0]
            entropy = float(-np.sum(non_zero * np.log(non_zero)))

            tot_scans = max(1, len(res.step_records))
            all_telemetry[s_name].append({
                "unique_bands": len(scanned_bands),
                "max_consecutive": max_consec,
                "mean_run_length": mean_run,
                "entropy": entropy,
                "scans_hit_pct": (scans_hit_bands / tot_scans) * 100.0,
                "scans_unhit_pct": (scans_unhit_bands / tot_scans) * 100.0,
            })

            if s == 0:
                sample_episodes[s_name] = res
                if isinstance(scheduler, HybridAdaptiveScheduler):
                    sample_hybrid_logs = list(scheduler.diagnostics.step_logs)

    # Aggregate summaries
    benchmark_summary: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for s_name in schedulers_factory:
        ms = all_metrics[s_name]
        ts = all_telemetry[s_name]

        ir = [m.interception_rate for m in ms]
        opp = [m.successful_interceptions for m in ms]
        delay = [m.average_intercept_time for m in ms if m.average_intercept_time is not None]
        ttfd = [m.scenario_ttfd for m in ms if m.scenario_ttfd is not None]
        pd = [m.empirical_pd for m in ms]
        pfa = [m.empirical_pfa for m in ms]
        eff = [m.dwell_efficiency for m in ms]
        tp = [m.tp_count for m in ms]
        tp_per_opp = [m.tp_count / max(1, m.successful_interceptions) for m in ms]
        ub = [t["unique_bands"] for t in ts]
        mc = [t["max_consecutive"] for t in ts]
        mrl = [t["mean_run_length"] for t in ts]
        ent = [t["entropy"] for t in ts]
        s_hit = [t["scans_hit_pct"] for t in ts]
        s_unhit = [t["scans_unhit_pct"] for t in ts]

        benchmark_summary[s_name] = {
            "interception_rate": (float(np.mean(ir)), float(np.std(ir, ddof=1))),
            "successful_interceptions": (float(np.mean(opp)), float(np.std(opp, ddof=1))),
            "average_intercept_time": (float(np.mean(delay)), float(np.std(delay, ddof=1))),
            "scenario_ttfd": (float(np.mean(ttfd)), float(np.std(ttfd, ddof=1))) if ttfd else (0.0, 0.0),
            "empirical_pd": (float(np.mean(pd)), float(np.std(pd, ddof=1))),
            "empirical_pfa": (float(np.mean(pfa)), float(np.std(pfa, ddof=1))),
            "dwell_efficiency": (float(np.mean(eff)), float(np.std(eff, ddof=1))),
            "tp_count": (float(np.mean(tp)), float(np.std(tp, ddof=1))),
            "tp_per_opp": (float(np.mean(tp_per_opp)), float(np.std(tp_per_opp, ddof=1))),
            "unique_bands": (float(np.mean(ub)), float(np.std(ub, ddof=1))),
            "max_consecutive": (float(np.mean(mc)), float(np.std(mc, ddof=1))),
            "mean_run_length": (float(np.mean(mrl)), float(np.std(mrl, ddof=1))),
            "entropy": (float(np.mean(ent)), float(np.std(ent, ddof=1))),
            "scans_hit_pct": (float(np.mean(s_hit)), float(np.std(s_hit, ddof=1))),
            "scans_unhit_pct": (float(np.mean(s_unhit)), float(np.std(s_unhit, ddof=1))),
        }

    return benchmark_summary, sample_episodes, sample_hybrid_logs


def print_6way_benchmark_table(summary: Dict[str, Dict[str, Tuple[float, float]]]) -> None:
    """Print ASCII comparison table across all 6 schedulers."""
    print("\n" + "=" * 145)
    print(f"{'Metric':<36} | {'Open-Loop':<16} | {'XGBoost':<16} | {'Hardened LinUCB':<17} | {'Original PPO':<16} | {'Hardened PPO':<16} | {'Phase 6 Hybrid':<16}")
    print("=" * 145)

    metric_rows = [
        ("Interception Rate", "interception_rate", "%", 100.0),
        ("Unique Opps Intercepted", "successful_interceptions", "", 1.0),
        ("Average Intercept Delay", "average_intercept_time", " slots", 1.0),
        ("Scenario TTFD", "scenario_ttfd", " slots", 1.0),
        ("Receiver Empirical Pd", "empirical_pd", "", 1.0),
        ("Receiver Empirical Pfa", "empirical_pfa", "", 1.0),
        ("Dwell Efficiency", "dwell_efficiency", "%", 100.0),
        ("Total TP Detections (slots)", "tp_count", "", 1.0),
        ("TP / Intercepted Opp", "tp_per_opp", "", 1.0),
        ("Unique Bands Scanned", "unique_bands", " / 20", 1.0),
        ("Max Consecutive Scans", "max_consecutive", " slots", 1.0),
        ("Mean Run Length", "mean_run_length", " slots", 1.0),
        ("Shannon Entropy", "entropy", " nats", 1.0),
        ("Scans on Previously Hit Bands", "scans_hit_pct", "%", 1.0),
        ("Scans on Unsuccessful Bands", "scans_unhit_pct", "%", 1.0),
    ]

    sched_keys = [
        "Open-Loop Baseline",
        "XGBoost Adaptive",
        "Hardened LinUCB",
        "Original PPO Baseline",
        "Hardened PPO",
        "Hybrid Adaptive Scheduler",
    ]

    for label, key, unit, scale in metric_rows:
        row_str = f"{label:<36} |"
        for s_name in sched_keys:
            mean_val, std_val = summary[s_name].get(key, (0.0, 0.0))
            if unit == "%":
                entry = f"{mean_val * scale:.2f}% ± {std_val * scale:.2f}%"
            elif unit == " / 20":
                entry = f"{mean_val:.1f} ± {std_val:.1f}"
            elif unit in (" slots", " nats", ""):
                entry = f"{mean_val * scale:.2f} ± {std_val * scale:.2f}"
            else:
                entry = f"{mean_val * scale:.2f}"
            row_str += f" {entry:<16} |"
        print(row_str)

    print("=" * 145)


def main() -> None:
    print("=" * 125)
    print("SIH26055 — Phase 6: Isolated Hybrid Adaptive RF Scheduler Benchmark Pipeline")
    print("=" * 125)

    base_yaml = Path("configs/default.yaml")
    env_config = load_config(base_yaml) if base_yaml.exists() else EnvironmentConfig()

    train_seeds = list(range(100, 120))
    val_seeds = list(range(120, 125))
    test_seeds = list(range(0, 10))

    # 1. Prepare XGBoost Baseline Model
    print("\n[*] Preparing Phase 3 XGBoost Adaptive Model...")
    xgb_model, _ = train_xgboost_pipeline(
        config=env_config,
        train_seeds=train_seeds,
        val_seeds=val_seeds,
    )

    # 2. Load Original PPO Model (Phase 5 Baseline)
    orig_ppo_agent = PPOAgent(state_dim=227, action_dim=60)
    orig_model_path = Path("artifacts/ppo/best_ppo_model.pt")
    if orig_model_path.exists():
        orig_ppo_agent.load(str(orig_model_path))
        print(f"[*] Loaded Original PPO Baseline model from {orig_model_path}")
    else:
        print("[!] Original PPO checkpoint not found!")

    # 3. Load Hardened PPO Agent (Pre-Phase 6A)
    hardened_ppo_agent = PPOAgent(state_dim=227, action_dim=60)
    hardened_model_path = Path("artifacts/ppo/best_hardened_ppo_model.pt")
    if hardened_model_path.exists():
        hardened_ppo_agent.load(str(hardened_model_path))
        print(f"[*] Loaded trained Hardened PPO checkpoint from {hardened_model_path}")
    else:
        print("[!] Hardened PPO checkpoint not found!")

    # 4. Run 6-Way Dynamic Frequency Hopping Adaptation Experiment
    hop_results = run_6way_frequency_hopping_adaptation_experiment(
        trained_xgb_model=xgb_model,
        orig_ppo_agent=orig_ppo_agent,
        hardened_ppo_agent=hardened_ppo_agent,
    )

    # 5. Run 6-Way Head-to-Head Benchmark on 10 Unseen Test Seeds (0..9)
    benchmark_summary, sample_episodes, sample_hybrid_logs = run_6way_benchmark(
        config=env_config,
        trained_xgb_model=xgb_model,
        orig_ppo_agent=orig_ppo_agent,
        hardened_ppo_agent=hardened_ppo_agent,
        test_seeds=test_seeds,
    )

    # 6. Print 6-Way Comparison Table
    print_6way_benchmark_table(benchmark_summary)

    # 7. Generate Presentation-Quality Visual Artifacts
    print("\n[*] Generating Phase 6 Visual Artifacts...")
    plot_6way_benchmark_comparison(benchmark_summary, output_path="phase6_sixway_comparison.png")
    print("    - Saved phase6_sixway_comparison.png")

    plot_6way_trajectories({k: v.step_records for k, v in sample_episodes.items()}, output_path="phase6_hybrid_trajectory_comparison.png")
    print("    - Saved phase6_hybrid_trajectory_comparison.png")

    plot_6way_frequency_hopping_adaptation(hop_results, output_path="phase6_hybrid_frequency_hopping.png")
    print("    - Saved phase6_hybrid_frequency_hopping.png")

    if sample_hybrid_logs:
        plot_hybrid_exploration_exploitation(sample_hybrid_logs, output_path="phase6_hybrid_exploration_exploitation.png")
        print("    - Saved phase6_hybrid_exploration_exploitation.png")

        plot_hybrid_arbitration_diagnostics(sample_hybrid_logs, output_path="phase6_hybrid_arbitration_diagnostics.png")
        print("    - Saved phase6_hybrid_arbitration_diagnostics.png")

    print("\n" + "=" * 125)
    print("PHASE 6: HYBRID ADAPTIVE RF SCHEDULER PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 125)


if __name__ == "__main__":
    main()
