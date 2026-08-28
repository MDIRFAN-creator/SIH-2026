"""
Phase 4 LinUCB Contextual Bandit Experiment & Tri-Scheduler Benchmark (SIH26055 Hardened).

Performs:
1. Alpha & Discount Factor Gamma Sensitivity Sweep (validation seeds 120..124).
2. Multi-Time & Multi-Band Controlled Frequency-Change Adaptation Experiment.
3. Cold-Start and Anti-Camping Validation.
4. Tri-Scheduler Head-to-Head Benchmark on 10 Unseen Test Seeds (0..9):
   - Phase 2: Open-Loop Baseline
   - Phase 3: XGBoost Adaptive Scheduler
   - Phase 4: Hardened LinUCB Contextual Bandit
   - Reference: Pre-Hardening LinUCB
5. Visualization Artifacts Generation.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import numpy as np

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
from models import XGBoostBandPredictor
from optimizers import ActionOptimizer
from runners import EpisodeResult, EpisodeRunner
from schedulers import LinUCBScheduler, OpenLoopScheduler, XGBoostScheduler
from training import train_xgboost_pipeline
from visualization import (
    plot_before_after_hardening,
    plot_frequency_adaptation_timeline,
    plot_linucb_diagnostics,
    plot_tri_benchmark_summary,
    plot_tri_scheduler_trajectories,
)


def run_parameter_sensitivity(
    config: EnvironmentConfig,
    val_seeds: List[int],
    runner: EpisodeRunner,
) -> Tuple[Dict[float, Dict[str, Any]], Dict[float, Dict[str, Any]]]:
    """
    Evaluate LinUCB sensitivity over exploration alpha and discount gamma on validation seeds.
    """
    alpha_candidates = [0.25, 0.5, 1.0, 2.0]
    gamma_candidates = [0.95, 0.98, 0.99, 1.00]

    alpha_results: Dict[float, Dict[str, Any]] = {}
    gamma_results: Dict[float, Dict[str, Any]] = {}

    # 1. Alpha sweep (fixed gamma=0.99)
    for alpha in alpha_candidates:
        ir_list, delay_list, eff_list, rew_list, max_c_list, ent_list = [], [], [], [], [], []
        for s in val_seeds:
            env = RFEnvironment(config)
            scheduler = LinUCBScheduler(
                num_bands=config.num_bands,
                alpha=alpha,
                gamma=0.99,
                max_consecutive_scans=3,
                min_initial_pulls=1,
                seed=s,
            )
            res = runner.run_episode(env, scheduler, seed=s)
            m = calculate_baseline_metrics(res, env.emitter_registry)

            ir_list.append(m.interception_rate)
            delay_list.append(m.average_intercept_time)
            eff_list.append(m.dwell_efficiency)
            rew_list.append(scheduler.cumulative_reward)
            ent_list.append(scheduler.compute_band_selection_entropy())

            # Max consecutive scans
            max_c, curr_c, prev_b = 0, 0, -1
            for r in res.step_records:
                b = r.action.frequency_band
                if b == prev_b:
                    curr_c += 1
                else:
                    prev_b = b
                    curr_c = 1
                max_c = max(max_c, curr_c)
            max_c_list.append(max_c)

        alpha_results[alpha] = {
            "interception_rate": (float(np.mean(ir_list)), float(np.std(ir_list, ddof=1))),
            "average_intercept_time": (float(np.mean(delay_list)), float(np.std(delay_list, ddof=1))),
            "dwell_efficiency": (float(np.mean(eff_list)), float(np.std(eff_list, ddof=1))),
            "cumulative_reward": (float(np.mean(rew_list)), float(np.std(rew_list, ddof=1))),
            "max_consecutive": (float(np.mean(max_c_list)), float(np.std(max_c_list, ddof=1))),
            "entropy": (float(np.mean(ent_list)), float(np.std(ent_list, ddof=1))),
        }

    # 2. Gamma sweep (fixed alpha=1.0)
    for gamma in gamma_candidates:
        ir_list, delay_list, eff_list, rew_list, max_c_list, ent_list = [], [], [], [], [], []
        for s in val_seeds:
            env = RFEnvironment(config)
            scheduler = LinUCBScheduler(
                num_bands=config.num_bands,
                alpha=1.0,
                gamma=gamma,
                max_consecutive_scans=3,
                min_initial_pulls=1,
                seed=s,
            )
            res = runner.run_episode(env, scheduler, seed=s)
            m = calculate_baseline_metrics(res, env.emitter_registry)

            ir_list.append(m.interception_rate)
            delay_list.append(m.average_intercept_time)
            eff_list.append(m.dwell_efficiency)
            rew_list.append(scheduler.cumulative_reward)
            ent_list.append(scheduler.compute_band_selection_entropy())

            max_c, curr_c, prev_b = 0, 0, -1
            for r in res.step_records:
                b = r.action.frequency_band
                if b == prev_b:
                    curr_c += 1
                else:
                    prev_b = b
                    curr_c = 1
                max_c = max(max_c, curr_c)
            max_c_list.append(max_c)

        gamma_results[gamma] = {
            "interception_rate": (float(np.mean(ir_list)), float(np.std(ir_list, ddof=1))),
            "average_intercept_time": (float(np.mean(delay_list)), float(np.std(delay_list, ddof=1))),
            "dwell_efficiency": (float(np.mean(eff_list)), float(np.std(eff_list, ddof=1))),
            "cumulative_reward": (float(np.mean(rew_list)), float(np.std(rew_list, ddof=1))),
            "max_consecutive": (float(np.mean(max_c_list)), float(np.std(max_c_list, ddof=1))),
            "entropy": (float(np.mean(ent_list)), float(np.std(ent_list, ddof=1))),
        }

    return alpha_results, gamma_results


def run_multi_frequency_adaptation_test(
    xgb_model: XGBoostBandPredictor,
    runner: EpisodeRunner,
) -> List[Dict[str, Any]]:
    """
    Compare how rapidly Open-Loop, XGBoost, and Hardened LinUCB detect frequency changes
    across multiple destination bands and multiple change times.
    """
    scenarios = [
        {"dest_band": 14, "change_time": 1000},
        {"dest_band": 7, "change_time": 2000},
        {"dest_band": 18, "change_time": 3000},
    ]

    adaptation_results = []

    for sc in scenarios:
        dest_b = sc["dest_band"]
        t_ch = sc["change_time"]

        config_dyn = EnvironmentConfig(
            num_bands=20,
            simulation_duration=5000,
            seed=2026,
            emitters=[
                {"emitter_id": "bg_p_b2", "emitter_type": "PERIODIC", "frequency_band": 2, "period": 20, "active_duration": 4},
                {"emitter_id": f"hopping_target_b{dest_b}", "emitter_type": "PERIODIC", "frequency_band": dest_b, "period": 15, "active_duration": 6, "start_time": t_ch},
            ],
        )

        # 1. Open Loop
        env_ol = RFEnvironment(config_dyn)
        sched_ol = OpenLoopScheduler(num_bands=20, dwell_time=1)
        res_ol = runner.run_episode(env_ol, sched_ol, seed=2026)

        # 2. XGBoost
        env_xgb = RFEnvironment(config_dyn)
        opt = ActionOptimizer(num_bands=20, allowed_dwells=[1, 2, 3], repeat_penalty_weight=0.15, dwell_penalty_weight=0.05, max_consecutive_scans=3)
        sched_xgb = XGBoostScheduler(model=xgb_model, num_bands=20, optimizer=opt)
        res_xgb = runner.run_episode(env_xgb, sched_xgb, seed=2026)

        # 3. Hardened LinUCB
        env_lin = RFEnvironment(config_dyn)
        sched_lin = LinUCBScheduler(num_bands=20, alpha=1.0, gamma=0.99, max_consecutive_scans=3, min_initial_pulls=1, seed=2026)
        res_lin = runner.run_episode(env_lin, sched_lin, seed=2026)

        entry: Dict[str, Any] = {
            "dest_band": dest_b,
            "change_time": t_ch,
        }

        for name, res in [("Open-Loop", res_ol), ("XGBoost", res_xgb), ("LinUCB", res_lin)]:
            first_scan = None
            first_hit = None
            for rec in res.step_records:
                if rec.start_time >= t_ch:
                    if rec.action.frequency_band == dest_b:
                        if first_scan is None:
                            first_scan = rec.start_time
                        if rec.observation.result == DetectionResult.HIT:
                            if first_hit is None:
                                first_hit = rec.start_time
                                break
            scan_lat = (first_scan - t_ch) if first_scan is not None else 9999
            hit_lat = (first_hit - t_ch) if first_hit is not None else 9999
            entry[name] = {
                "scan_latency": scan_lat,
                "detection_latency": hit_lat,
            }

        adaptation_results.append(entry)

    return adaptation_results


def run_detailed_profiling(
    results_dict: Dict[str, List[EpisodeResult]],
    schedulers_dict: Dict[str, List[Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Extract comprehensive behavioral, allocation, entropy, and run-length statistics.
    """
    profiling = {}

    for name, res_list in results_dict.items():
        pct_hit_band = []
        pct_unsucc_band = []
        pct_never_band = []
        unique_bands = []
        max_consec = []
        mean_consec = []
        dwell_1 = []
        dwell_2 = []
        dwell_3 = []
        entropies = []

        for idx, res in enumerate(res_list):
            hit_bands = set()
            scanned_bands = set()
            c_hit, c_unsucc, c_never = 0, 0, 0
            d1, d2, d3 = 0, 0, 0

            max_c = 0
            curr_c = 0
            last_b = -1
            run_lengths = []

            for r in res.step_records:
                b = r.action.frequency_band
                d = r.action.dwell_time

                if d == 1:
                    d1 += 1
                elif d == 2:
                    d2 += 1
                elif d == 3:
                    d3 += 1

                if b == last_b:
                    curr_c += 1
                else:
                    if curr_c > 0:
                        run_lengths.append(curr_c)
                    last_b = b
                    curr_c = 1
                max_c = max(max_c, curr_c)

                if b not in scanned_bands:
                    c_never += 1
                elif b in hit_bands:
                    c_hit += 1
                else:
                    c_unsucc += 1

                scanned_bands.add(b)
                if r.observation.result == DetectionResult.HIT:
                    hit_bands.add(b)

            if curr_c > 0:
                run_lengths.append(curr_c)

            total_steps = len(res.step_records)
            pct_hit_band.append(c_hit / total_steps)
            pct_unsucc_band.append(c_unsucc / total_steps)
            pct_never_band.append(c_never / total_steps)
            unique_bands.append(len(scanned_bands))
            max_consec.append(max_c)
            mean_consec.append(float(np.mean(run_lengths)) if run_lengths else 1.0)
            dwell_1.append(d1 / total_steps)
            dwell_2.append(d2 / total_steps)
            dwell_3.append(d3 / total_steps)

            # Shannon Entropy
            pull_counts = np.zeros(20, dtype=np.float64)
            for r in res.step_records:
                pull_counts[r.action.frequency_band] += 1.0
            probs = pull_counts / np.sum(pull_counts)
            probs = probs[probs > 0]
            ent = -float(np.sum(probs * np.log(probs)))
            entropies.append(ent)

        profiling[name] = {
            "pct_hit_band": (float(np.mean(pct_hit_band)), float(np.std(pct_hit_band, ddof=1))),
            "pct_unsucc_band": (float(np.mean(pct_unsucc_band)), float(np.std(pct_unsucc_band, ddof=1))),
            "pct_never_band": (float(np.mean(pct_never_band)), float(np.std(pct_never_band, ddof=1))),
            "unique_bands": (float(np.mean(unique_bands)), float(np.std(unique_bands, ddof=1))),
            "max_consecutive": (float(np.mean(max_consec)), float(np.std(max_consec, ddof=1))),
            "mean_consecutive": (float(np.mean(mean_consec)), float(np.std(mean_consec, ddof=1))),
            "entropy": (float(np.mean(entropies)), float(np.std(entropies, ddof=1))),
            "dwell_distribution": {
                "dwell_1": float(np.mean(dwell_1)),
                "dwell_2": float(np.mean(dwell_2)),
                "dwell_3": float(np.mean(dwell_3)),
            },
        }

    return profiling


def run_phase4_benchmark():
    print("=" * 105)
    print("SIH26055 — Phase 4 Hardened LinUCB Benchmark & Tri-Scheduler Comparison")
    print("=" * 105)

    config_path = Path("configs/default.yaml")
    config = load_config(config_path)
    runner = EpisodeRunner()

    train_seeds = list(range(100, 120))
    val_seeds = list(range(120, 125))
    test_seeds = list(range(0, 10))

    # 1. XGBoost Predictor Preparation
    print("\n[1] Preparing XGBoost Adaptive Predictor...")
    xgb_model_path = Path("models/xgboost_model.json")
    if xgb_model_path.exists():
        xgb_model = XGBoostBandPredictor.load(xgb_model_path)
    else:
        xgb_model, _ = train_xgboost_pipeline(config, train_seeds, val_seeds, save_path=xgb_model_path)

    # 2. Alpha & Gamma Sensitivity Sweeps
    print("\n[2] Executing Parameter Sensitivity Sweeps on Validation Seeds (120..124)...")
    alpha_res, gamma_res = run_parameter_sensitivity(config, val_seeds, runner)

    print("\n  --- Alpha Exploration Sweep (fixed gamma=0.99, max_consec=3) ---")
    print(f"  {'Alpha':<8} | {'Interception Rate':<20} | {'Avg Delay':<18} | {'Dwell Eff':<16} | {'Max Run':<10} | {'Entropy':<12}")
    print("  " + "-" * 95)
    for alpha, st in alpha_res.items():
        ir_str = f"{st['interception_rate'][0]*100:.2f}% ± {st['interception_rate'][1]*100:.2f}%"
        del_str = f"{st['average_intercept_time'][0]:.2f} ± {st['average_intercept_time'][1]:.2f}s"
        eff_str = f"{st['dwell_efficiency'][0]*100:.2f}%"
        mc_str = f"{st['max_consecutive'][0]:.1f}"
        ent_str = f"{st['entropy'][0]:.3f}"
        print(f"  {alpha:<8} | {ir_str:<20} | {del_str:<18} | {eff_str:<16} | {mc_str:<10} | {ent_str:<12}")

    print("\n  --- Discount Gamma Sweep (fixed alpha=1.0, max_consec=3) ---")
    print(f"  {'Gamma':<8} | {'Interception Rate':<20} | {'Avg Delay':<18} | {'Dwell Eff':<16} | {'Max Run':<10} | {'Reward':<14}")
    print("  " + "-" * 95)
    for gamma, st in gamma_res.items():
        ir_str = f"{st['interception_rate'][0]*100:.2f}% ± {st['interception_rate'][1]*100:.2f}%"
        del_str = f"{st['average_intercept_time'][0]:.2f} ± {st['average_intercept_time'][1]:.2f}s"
        eff_str = f"{st['dwell_efficiency'][0]*100:.2f}%"
        mc_str = f"{st['max_consecutive'][0]:.1f}"
        rew_str = f"{st['cumulative_reward'][0]:.1f}"
        print(f"  {gamma:<8} | {ir_str:<20} | {del_str:<18} | {eff_str:<16} | {mc_str:<10} | {rew_str:<14}")

    selected_alpha = 1.0
    selected_gamma = 0.99

    # 3. Multi-Frequency Adaptation Response
    print("\n[3] Executing Multi-Scenario Controlled Frequency Adaptation Experiment...")
    adapt_results = run_multi_frequency_adaptation_test(xgb_model, runner)
    print(f"{'Scenario':<22} | {'Open-Loop (Scan/Hit Latency)':<32} | {'XGBoost (Scan/Hit Latency)':<30} | {'Hardened LinUCB (Scan/Hit Latency)':<32}")
    print("-" * 122)
    for entry in adapt_results:
        sc_lbl = f"t={entry['change_time']}, Band {entry['dest_band']}"
        ol_str = f"{entry['Open-Loop']['scan_latency']} slots / {entry['Open-Loop']['detection_latency']} slots"
        xgb_str = f"{entry['XGBoost']['scan_latency']} slots / {entry['XGBoost']['detection_latency']} slots"
        lin_str = f"{entry['LinUCB']['scan_latency']} slots / {entry['LinUCB']['detection_latency']} slots"
        print(f"{sc_lbl:<22} | {ol_str:<32} | {xgb_str:<30} | {lin_str:<32}")

    # 4. Main Head-to-Head Benchmark on 10 Unseen Test Seeds (0..9)
    print("\n[4] Running Benchmark Across Schedulers on 10 Unseen Test Seeds (0..9)...")
    results_dict: Dict[str, List[EpisodeResult]] = {
        "Open-Loop": [],
        "XGBoost": [],
        "Pre-Hardening LinUCB": [],
        "Hardened LinUCB": [],
    }
    metrics_dict: Dict[str, List[BaselineMetrics]] = {
        "Open-Loop": [],
        "XGBoost": [],
        "Pre-Hardening LinUCB": [],
        "Hardened LinUCB": [],
    }
    schedulers_dict: Dict[str, List[Any]] = {
        "Open-Loop": [],
        "XGBoost": [],
        "Pre-Hardening LinUCB": [],
        "Hardened LinUCB": [],
    }

    for seed in test_seeds:
        # A. Open-Loop Baseline
        env_ol = RFEnvironment(config)
        sched_ol = OpenLoopScheduler(num_bands=config.num_bands, dwell_time=1)
        res_ol = runner.run_episode(env_ol, sched_ol, seed=seed)
        m_ol = calculate_baseline_metrics(res_ol, env_ol.emitter_registry)
        results_dict["Open-Loop"].append(res_ol)
        metrics_dict["Open-Loop"].append(m_ol)
        schedulers_dict["Open-Loop"].append(sched_ol)

        # B. XGBoost Adaptive
        env_xgb = RFEnvironment(config)
        opt = ActionOptimizer(num_bands=config.num_bands, allowed_dwells=[1, 2, 3], repeat_penalty_weight=0.15, dwell_penalty_weight=0.05, max_consecutive_scans=3)
        sched_xgb = XGBoostScheduler(model=xgb_model, num_bands=config.num_bands, optimizer=opt)
        res_xgb = runner.run_episode(env_xgb, sched_xgb, seed=seed)
        m_xgb = calculate_baseline_metrics(res_xgb, env_xgb.emitter_registry)
        results_dict["XGBoost"].append(res_xgb)
        metrics_dict["XGBoost"].append(m_xgb)
        schedulers_dict["XGBoost"].append(sched_xgb)

        # C. Pre-Hardening LinUCB (for exact before-after delta)
        env_pre = RFEnvironment(config)
        sched_pre = LinUCBScheduler(num_bands=config.num_bands, alpha=1.0, gamma=1.0, max_consecutive_scans=9999, min_initial_pulls=0, seed=seed)
        res_pre = runner.run_episode(env_pre, sched_pre, seed=seed)
        m_pre = calculate_baseline_metrics(res_pre, env_pre.emitter_registry)
        results_dict["Pre-Hardening LinUCB"].append(res_pre)
        metrics_dict["Pre-Hardening LinUCB"].append(m_pre)
        schedulers_dict["Pre-Hardening LinUCB"].append(sched_pre)

        # D. Hardened LinUCB
        env_hard = RFEnvironment(config)
        sched_hard = LinUCBScheduler(
            num_bands=config.num_bands,
            alpha=selected_alpha,
            gamma=selected_gamma,
            max_consecutive_scans=3,
            min_initial_pulls=1,
            seed=seed,
        )
        res_hard = runner.run_episode(env_hard, sched_hard, seed=seed)
        m_hard = calculate_baseline_metrics(res_hard, env_hard.emitter_registry)
        results_dict["Hardened LinUCB"].append(res_hard)
        metrics_dict["Hardened LinUCB"].append(m_hard)
        schedulers_dict["Hardened LinUCB"].append(sched_hard)

    # Aggregate metrics
    agg_ol = aggregate_metrics_across_seeds(metrics_dict["Open-Loop"])
    agg_xgb = aggregate_metrics_across_seeds(metrics_dict["XGBoost"])
    agg_pre = aggregate_metrics_across_seeds(metrics_dict["Pre-Hardening LinUCB"])
    agg_hard = aggregate_metrics_across_seeds(metrics_dict["Hardened LinUCB"])

    def get_opp_stats(m_list: List[BaselineMetrics]):
        unique_opps = [m.successful_interceptions for m in m_list]
        tps = [m.tp_count for m in m_list]
        tp_per_opp = [(m.tp_count / m.successful_interceptions) if m.successful_interceptions > 0 else 0.0 for m in m_list]
        return (
            (float(np.mean(unique_opps)), float(np.std(unique_opps, ddof=1))),
            (float(np.mean(tps)), float(np.std(tps, ddof=1))),
            (float(np.mean(tp_per_opp)), float(np.std(tp_per_opp, ddof=1))),
        )

    opp_ol, tp_ol, tpopp_ol = get_opp_stats(metrics_dict["Open-Loop"])
    opp_xgb, tp_xgb, tpopp_xgb = get_opp_stats(metrics_dict["XGBoost"])
    opp_pre, tp_pre, tpopp_pre = get_opp_stats(metrics_dict["Pre-Hardening LinUCB"])
    opp_hard, tp_hard, tpopp_hard = get_opp_stats(metrics_dict["Hardened LinUCB"])

    # Detailed profiling
    profiling = run_detailed_profiling(results_dict, schedulers_dict)

    # 5. Display Benchmark Table
    print("\n" + "=" * 115)
    print("TRI-SCHEDULER BENCHMARK COMPARISON: Open Loop vs XGBoost vs Hardened LinUCB (N = 10 Unseen Test Seeds)")
    print("=" * 115)
    print(f"{'Metric':<38} | {'Open-Loop Baseline':<22} | {'XGBoost Adaptive':<22} | {'Hardened LinUCB':<22}")
    print("-" * 115)

    def fmt_pct(stat):
        return f"{stat[0]*100:.2f}% ± {stat[1]*100:.2f}%"

    def fmt_num(stat, unit=""):
        u = f" {unit}" if unit else ""
        return f"{stat[0]:.2f} ± {stat[1]:.2f}{u}"

    print(f"{'Interception Rate':<38} | {fmt_pct(agg_ol['interception_rate']):<22} | {fmt_pct(agg_xgb['interception_rate']):<22} | {fmt_pct(agg_hard['interception_rate']):<22}")
    print(f"{'Unique Opportunities Intercepted':<38} | {fmt_num(opp_ol):<22} | {fmt_num(opp_xgb):<22} | {fmt_num(opp_hard):<22}")
    print(f"{'Average Intercept Time':<38} | {fmt_num(agg_ol['average_intercept_time'], 'slots'):<22} | {fmt_num(agg_xgb['average_intercept_time'], 'slots'):<22} | {fmt_num(agg_hard['average_intercept_time'], 'slots'):<22}")
    print(f"{'PRD Scenario TTFD':<38} | {fmt_num(agg_ol['scenario_ttfd'], 'slots'):<22} | {fmt_num(agg_xgb['scenario_ttfd'], 'slots'):<22} | {fmt_num(agg_hard['scenario_ttfd'], 'slots'):<22}")
    print(f"{'Receiver Empirical Pd':<38} | {fmt_num(agg_ol['empirical_pd']):<22} | {fmt_num(agg_xgb['empirical_pd']):<22} | {fmt_num(agg_hard['empirical_pd']):<22}")
    print(f"{'Receiver Empirical Pfa':<38} | {fmt_num(agg_ol['empirical_pfa']):<22} | {fmt_num(agg_xgb['empirical_pfa']):<22} | {fmt_num(agg_hard['empirical_pfa']):<22}")
    print(f"{'Dwell Efficiency':<38} | {fmt_pct(agg_ol['dwell_efficiency']):<22} | {fmt_pct(agg_xgb['dwell_efficiency']):<22} | {fmt_pct(agg_hard['dwell_efficiency']):<22}")
    print(f"{'Total TP Detections (Slots)':<38} | {fmt_num(tp_ol):<22} | {fmt_num(tp_xgb):<22} | {fmt_num(tp_hard):<22}")
    print(f"{'TP Detections / Intercepted Opp':<38} | {fmt_num(tpopp_ol):<22} | {fmt_num(tpopp_xgb):<22} | {fmt_num(tpopp_hard):<22}")
    print(f"{'Unique Frequency Bands Scanned':<38} | {fmt_num(profiling['Open-Loop']['unique_bands']):<22} | {fmt_num(profiling['XGBoost']['unique_bands']):<22} | {fmt_num(profiling['Hardened LinUCB']['unique_bands']):<22}")
    print(f"{'Max Consecutive Band Scans':<38} | {fmt_num(profiling['Open-Loop']['max_consecutive']):<22} | {fmt_num(profiling['XGBoost']['max_consecutive']):<22} | {fmt_num(profiling['Hardened LinUCB']['max_consecutive']):<22}")
    print(f"{'Mean Consecutive Run Length':<38} | {fmt_num(profiling['Open-Loop']['mean_consecutive']):<22} | {fmt_num(profiling['XGBoost']['mean_consecutive']):<22} | {fmt_num(profiling['Hardened LinUCB']['mean_consecutive']):<22}")
    print(f"{'Band-Selection Shannon Entropy':<38} | {fmt_num(profiling['Open-Loop']['entropy']):<22} | {fmt_num(profiling['XGBoost']['entropy']):<22} | {fmt_num(profiling['Hardened LinUCB']['entropy']):<22}")
    print(f"{'Scans on Previously Hit Bands':<38} | {fmt_pct(profiling['Open-Loop']['pct_hit_band']):<22} | {fmt_pct(profiling['XGBoost']['pct_hit_band']):<22} | {fmt_pct(profiling['Hardened LinUCB']['pct_hit_band']):<22}")
    print(f"{'Scans on Unsuccessful Bands':<38} | {fmt_pct(profiling['Open-Loop']['pct_unsucc_band']):<22} | {fmt_pct(profiling['XGBoost']['pct_unsucc_band']):<22} | {fmt_pct(profiling['Hardened LinUCB']['pct_unsucc_band']):<22}")

    hard_rew = (
        float(np.mean([s.cumulative_reward for s in schedulers_dict["Hardened LinUCB"]])),
        float(np.std([s.cumulative_reward for s in schedulers_dict["Hardened LinUCB"]], ddof=1)),
    )
    print(f"{'Cumulative Online Reward':<38} | {'N/A':<22} | {'N/A':<22} | {fmt_num(hard_rew):<22}")
    print("=" * 115)

    # 6. Before vs After Hardening Comparison
    print("\n" + "=" * 90)
    print("PHASE 4 HARDENING DELTA: Pre-Hardening LinUCB vs Hardened LinUCB (N = 10 Seeds)")
    print("=" * 90)
    print(f"{'Metric':<35} | {'Pre-Hardening LinUCB':<25} | {'Hardened LinUCB':<25}")
    print("-" * 90)
    print(f"{'Maximum Consecutive Scans':<35} | {fmt_num(profiling['Pre-Hardening LinUCB']['max_consecutive']):<25} | {fmt_num(profiling['Hardened LinUCB']['max_consecutive']):<25}")
    print(f"{'Mean Consecutive Run Length':<35} | {fmt_num(profiling['Pre-Hardening LinUCB']['mean_consecutive']):<25} | {fmt_num(profiling['Hardened LinUCB']['mean_consecutive']):<25}")
    print(f"{'Band-Selection Shannon Entropy':<35} | {fmt_num(profiling['Pre-Hardening LinUCB']['entropy']):<25} | {fmt_num(profiling['Hardened LinUCB']['entropy']):<25}")
    print(f"{'Interception Rate':<35} | {fmt_pct(agg_pre['interception_rate']):<25} | {fmt_pct(agg_hard['interception_rate']):<25}")
    print(f"{'Unique Opportunities Intercepted':<35} | {fmt_num(opp_pre):<25} | {fmt_num(opp_hard):<25}")
    print(f"{'Average Intercept Time':<35} | {fmt_num(agg_pre['average_intercept_time'], 'slots'):<25} | {fmt_num(agg_hard['average_intercept_time'], 'slots'):<25}")
    print(f"{'Dwell Efficiency':<35} | {fmt_pct(agg_pre['dwell_efficiency']):<25} | {fmt_pct(agg_hard['dwell_efficiency']):<25}")
    print(f"{'Total TP Detections (Slots)':<35} | {fmt_num(tp_pre):<25} | {fmt_num(tp_hard):<25}")
    print("=" * 90)

    # 7. Generate Visualizations
    print("\n[5] Generating Visualizations...")
    traj_path = Path("phase4_trajectory_comparison.png")
    plot_tri_scheduler_trajectories(
        open_loop_result=results_dict["Open-Loop"][-1],
        xgboost_result=results_dict["XGBoost"][-1],
        linucb_result=results_dict["Hardened LinUCB"][-1],
        max_time=200,
        save_path=traj_path,
    )
    print(f"    -> Saved {traj_path}")

    diag_path = Path("phase4_linucb_diagnostics.png")
    plot_linucb_diagnostics(
        scheduler=schedulers_dict["Hardened LinUCB"][-1],
        save_path=diag_path,
    )
    print(f"    -> Saved {diag_path}")

    bench_path = Path("phase4_benchmark_comparison.png")
    summary_data = {
        "Open-Loop": {
            "interception_rate": agg_ol["interception_rate"],
            "average_intercept_time": agg_ol["average_intercept_time"],
            "dwell_efficiency": agg_ol["dwell_efficiency"],
            "unique_opportunities": opp_ol,
        },
        "XGBoost": {
            "interception_rate": agg_xgb["interception_rate"],
            "average_intercept_time": agg_xgb["average_intercept_time"],
            "dwell_efficiency": agg_xgb["dwell_efficiency"],
            "unique_opportunities": opp_xgb,
        },
        "LinUCB": {
            "interception_rate": agg_hard["interception_rate"],
            "average_intercept_time": agg_hard["average_intercept_time"],
            "dwell_efficiency": agg_hard["dwell_efficiency"],
            "unique_opportunities": opp_hard,
        },
    }
    plot_tri_benchmark_summary(summary_data, save_path=bench_path)
    print(f"    -> Saved {bench_path}")

    before_after_path = Path("phase4_before_after_hardening.png")
    before_stats = {
        "max_consecutive": profiling["Pre-Hardening LinUCB"]["max_consecutive"],
        "interception_rate": agg_pre["interception_rate"],
        "unique_opportunities": opp_pre,
        "entropy": profiling["Pre-Hardening LinUCB"]["entropy"],
    }
    after_stats = {
        "max_consecutive": profiling["Hardened LinUCB"]["max_consecutive"],
        "interception_rate": agg_hard["interception_rate"],
        "unique_opportunities": opp_hard,
        "entropy": profiling["Hardened LinUCB"]["entropy"],
    }
    plot_before_after_hardening(before_stats, after_stats, save_path=before_after_path)
    print(f"    -> Saved {before_after_path}")

    timeline_path = Path("phase4_frequency_adaptation_timeline.png")
    plot_frequency_adaptation_timeline(adapt_results, save_path=timeline_path)
    print(f"    -> Saved {timeline_path}")

    print("\n" + "=" * 95)
    print("Phase 4 Hardening Benchmark Pipeline Completed Successfully!")
    print("=" * 95)


if __name__ == "__main__":
    run_phase4_benchmark()
