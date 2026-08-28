"""
Phase 3 Hardening & Adaptive Validation Analysis Script (SIH26055).

Performs:
1. Scanned dwell-slot decomposition (TP, FN, FP, TN, signal vs idle slots).
2. Repeated-detection & hit inflation analysis.
3. Frequency-change adaptation response experiment.
4. Exploitation vs Exploration scan distribution analysis.
5. Time-feature ablation experiment (Full vs No-Time vs Time-Only).
6. Extended metrics calculation across 10 test seeds.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment import EnvironmentConfig, RFEnvironment, load_config
from environment.types import Action, DetectionResult, Observation
from evaluation import (
    BaselineMetrics,
    aggregate_metrics_across_seeds,
    calculate_baseline_metrics,
    extract_emitter_opportunities,
)
from features.rf_features import FEATURE_NAMES, RFFeatureExtractor
from models import XGBoostBandPredictor
from optimizers import ActionOptimizer
from runners import EpisodeResult, EpisodeRunner
from schedulers import OpenLoopScheduler, XGBoostScheduler
from training import XGBoostDatasetGenerator, train_xgboost_pipeline


def analyze_dwell_and_hit_inflation(
    open_loop_results: List[EpisodeResult],
    xgboost_results: List[EpisodeResult],
    config: EnvironmentConfig,
) -> Dict[str, Any]:
    """
    Detailed slot-level and opportunity-level decomposition.
    """
    analysis: Dict[str, Any] = {
        "open_loop": {},
        "xgboost": {},
    }

    for name, res_list in [("open_loop", open_loop_results), ("xgboost", xgboost_results)]:
        total_scanned_slots = []
        observable_signal_slots = []
        idle_noise_slots = []
        tp_slots = []
        fn_slots = []
        fp_slots = []
        tn_slots = []

        total_decisions_list = []
        dwell_choices = {1: 0, 2: 0, 3: 0, 5: 0}

        unique_opps_intercepted = []
        total_tp_per_ep = []
        tps_per_intercepted_opp = []
        repeated_detections_per_opp = []
        unique_bands_scanned = []

        # Exploration/Exploitation metrics
        scans_on_prev_hit_band = []
        scans_on_prev_unsuccessful_band = []
        scans_on_never_scanned_band = []
        max_consecutive_scans_list = []
        band_visit_distributions = []

        for res in res_list:
            # Dwell slots decomposition
            tp = sum(sum(1 for s in d.slot_outcomes if s.is_true_positive) for d in res.dwell_history)
            fn = sum(sum(1 for s in d.slot_outcomes if s.is_false_negative) for d in res.dwell_history)
            fp = sum(sum(1 for s in d.slot_outcomes if s.is_false_positive) for d in res.dwell_history)
            tn = sum(sum(1 for s in d.slot_outcomes if s.is_true_negative) for d in res.dwell_history)

            total_slots = tp + fn + fp + tn
            signal_slots = tp + fn
            noise_slots = fp + tn

            total_scanned_slots.append(total_slots)
            observable_signal_slots.append(signal_slots)
            idle_noise_slots.append(noise_slots)
            tp_slots.append(tp)
            fn_slots.append(fn)
            fp_slots.append(fp)
            tn_slots.append(tn)

            total_decisions_list.append(res.total_decisions)
            for r in res.step_records:
                dwell_choices[r.action.dwell_time] = dwell_choices.get(r.action.dwell_time, 0) + 1

            # Unique bands
            bands_visited = set(r.action.frequency_band for r in res.step_records)
            unique_bands_scanned.append(len(bands_visited))

            # Band scan distribution
            band_counts = np.zeros(20, dtype=int)
            for r in res.step_records:
                band_counts[r.action.frequency_band] += 1
            band_visit_distributions.append(band_counts / max(1, res.total_decisions))

            # Consecutive scans
            max_consec = 0
            curr_consec = 0
            last_b = -1
            for r in res.step_records:
                b = r.action.frequency_band
                if b == last_b:
                    curr_consec += 1
                else:
                    last_b = b
                    curr_consec = 1
                max_consec = max(max_consec, curr_consec)
            max_consecutive_scans_list.append(max_consec)

            # Exploration vs Exploitation tracking
            hit_bands = set()
            scanned_bands = set()
            c_prev_hit = 0
            c_prev_unsucc = 0
            c_never = 0

            for r in res.step_records:
                b = r.action.frequency_band
                if b not in scanned_bands:
                    c_never += 1
                elif b in hit_bands:
                    c_prev_hit += 1
                else:
                    c_prev_unsucc += 1

                scanned_bands.add(b)
                if r.observation.result == DetectionResult.HIT:
                    hit_bands.add(b)

            total_steps = len(res.step_records)
            scans_on_prev_hit_band.append(c_prev_hit / total_steps)
            scans_on_prev_unsuccessful_band.append(c_prev_unsucc / total_steps)
            scans_on_never_scanned_band.append(c_never / total_steps)

            # Opportunity hit inflation analysis
            env_tmp = RFEnvironment(config)
            opps = extract_emitter_opportunities(
                env_tmp.emitter_registry,
                config.simulation_duration,
                config.num_bands,
            )

            # Count detections per opportunity
            opp_tp_counts = [0] * len(opps)
            for dwell in res.dwell_history:
                for opp_idx, opp in enumerate(opps):
                    if dwell.scanned_band == opp.frequency_band:
                        overlap_start = max(dwell.start_time, opp.start_time)
                        overlap_end = min(dwell.end_time, opp.end_time)
                        if overlap_start <= overlap_end:
                            for s in dwell.slot_outcomes:
                                if overlap_start <= s.time_slot <= overlap_end and s.is_true_positive:
                                    opp_tp_counts[opp_idx] += 1

            intercepted_opp_count = sum(1 for c in opp_tp_counts if c > 0)
            unique_opps_intercepted.append(intercepted_opp_count)
            total_tp_per_ep.append(sum(opp_tp_counts))

            if intercepted_opp_count > 0:
                tps_per_intercepted = [c for c in opp_tp_counts if c > 0]
                tps_per_intercepted_opp.append(float(np.mean(tps_per_intercepted)))
                repeated = sum(max(0, c - 1) for c in opp_tp_counts if c > 0)
                repeated_detections_per_opp.append(repeated)
            else:
                tps_per_intercepted_opp.append(0.0)
                repeated_detections_per_opp.append(0)

        analysis[name] = {
            "total_scanned_slots": (float(np.mean(total_scanned_slots)), float(np.std(total_scanned_slots, ddof=1))),
            "observable_signal_slots": (float(np.mean(observable_signal_slots)), float(np.std(observable_signal_slots, ddof=1))),
            "idle_noise_slots": (float(np.mean(idle_noise_slots)), float(np.std(idle_noise_slots, ddof=1))),
            "tp_slots": (float(np.mean(tp_slots)), float(np.std(tp_slots, ddof=1))),
            "fn_slots": (float(np.mean(fn_slots)), float(np.std(fn_slots, ddof=1))),
            "fp_slots": (float(np.mean(fp_slots)), float(np.std(fp_slots, ddof=1))),
            "tn_slots": (float(np.mean(tn_slots)), float(np.std(tn_slots, ddof=1))),
            "total_decisions": (float(np.mean(total_decisions_list)), float(np.std(total_decisions_list, ddof=1))),
            "dwell_choices_distribution": {k: v / sum(dwell_choices.values()) for k, v in dwell_choices.items()},
            "unique_opps_intercepted": (float(np.mean(unique_opps_intercepted)), float(np.std(unique_opps_intercepted, ddof=1))),
            "total_tp_detections": (float(np.mean(total_tp_per_ep)), float(np.std(total_tp_per_ep, ddof=1))),
            "tps_per_intercepted_opp": (float(np.mean(tps_per_intercepted_opp)), float(np.std(tps_per_intercepted_opp, ddof=1))),
            "repeated_detections_per_opp": (float(np.mean(repeated_detections_per_opp)), float(np.std(repeated_detections_per_opp, ddof=1))),
            "unique_bands_scanned": (float(np.mean(unique_bands_scanned)), float(np.std(unique_bands_scanned, ddof=1))),
            "max_consecutive_scans": (float(np.mean(max_consecutive_scans_list)), float(np.std(max_consecutive_scans_list, ddof=1))),
            "pct_scans_prev_hit_band": (float(np.mean(scans_on_prev_hit_band)), float(np.std(scans_on_prev_hit_band, ddof=1))),
            "pct_scans_prev_unsucc_band": (float(np.mean(scans_on_prev_unsuccessful_band)), float(np.std(scans_on_prev_unsuccessful_band, ddof=1))),
            "pct_scans_never_scanned_band": (float(np.mean(scans_on_never_scanned_band)), float(np.std(scans_on_never_scanned_band, ddof=1))),
            "mean_band_distribution": np.mean(band_visit_distributions, axis=0),
        }

    return analysis


def run_frequency_change_adaptation_experiment(
    model: XGBoostBandPredictor,
) -> Dict[str, Any]:
    """
    Controlled frequency-adaptation experiment:
    Scenario with an emitter that activates/changes to a new band at t=2000.
    Measures time-to-first-scan on the new band and time-to-first-detection after change.
    """
    config_adaptation = EnvironmentConfig(
        num_bands=20,
        simulation_duration=5000,
        seed=2026,
        emitters=[
            # Baseline background emitter on B2
            {
                "emitter_id": "bg_p_b2",
                "emitter_type": "PERIODIC",
                "frequency_band": 2,
                "period": 20,
                "active_duration": 4,
            },
            # Dynamic hopping emitter: appears on Band 14 at t=2000
            {
                "emitter_id": "hopping_target_b14",
                "emitter_type": "PERIODIC",
                "frequency_band": 14,
                "period": 30,
                "active_duration": 6,
                "start_time": 2000,
            },
        ],
    )

    runner = EpisodeRunner()

    # 1. Run Open Loop
    env_ol = RFEnvironment(config_adaptation)
    sched_ol = OpenLoopScheduler(num_bands=20, dwell_time=1)
    res_ol = runner.run_episode(env_ol, sched_ol, seed=2026)

    # 2. Run XGBoost Scheduler
    env_xgb = RFEnvironment(config_adaptation)
    optimizer = ActionOptimizer(num_bands=20, allowed_dwells=[1, 2, 3], repeat_penalty_weight=0.15, dwell_penalty_weight=0.05, max_consecutive_scans=3)
    sched_xgb = XGBoostScheduler(model=model, num_bands=20, optimizer=optimizer)
    res_xgb = runner.run_episode(env_xgb, sched_xgb, seed=2026)

    results = {}
    for name, res in [("Open-Loop", res_ol), ("XGBoost", res_xgb)]:
        first_scan_after_2000 = None
        first_hit_after_2000 = None

        for rec in res.step_records:
            if rec.start_time >= 2000:
                if rec.action.frequency_band == 14:
                    if first_scan_after_2000 is None:
                        first_scan_after_2000 = rec.start_time
                    if rec.observation.result == DetectionResult.HIT:
                        if first_hit_after_2000 is None:
                            first_hit_after_2000 = rec.start_time
                            break

        scan_delay = (first_scan_after_2000 - 2000) if first_scan_after_2000 is not None else None
        hit_delay = (first_hit_after_2000 - 2000) if first_hit_after_2000 is not None else None

        results[name] = {
            "first_scan_time": first_scan_after_2000,
            "first_hit_time": first_hit_after_2000,
            "scan_latency_slots": scan_delay,
            "detection_latency_slots": hit_delay,
        }

    return results


def run_time_feature_ablation(
    config: EnvironmentConfig,
    train_seeds: List[int],
    val_seeds: List[int],
    test_seeds: List[int],
) -> Dict[str, Any]:
    """
    Run ablation comparing:
    1. Full Feature Set (12 features)
    2. No-Time Feature Set (10 features: omitting time_sin, time_cos)
    3. History-Only Feature Set (cumulative_hit_rate, windowed_hit_rate, false_alarm_rate, scan_fraction, is_last_scanned, consecutive_scans)
    """
    runner = EpisodeRunner()

    ablation_configs = {
        "Full Feature Set (12 feats)": FEATURE_NAMES,
        "No-Time Features (10 feats)": [f for f in FEATURE_NAMES if f not in ["time_sin", "time_cos"]],
        "History-Only (6 feats)": [
            "cumulative_hit_rate",
            "windowed_hit_rate",
            "false_alarm_rate",
            "scan_fraction",
            "is_last_scanned",
            "consecutive_scans",
        ],
    }

    ablation_results = {}

    for name, feat_subset in ablation_configs.items():
        # Custom extractor wrapper
        class CustomExtractor(RFFeatureExtractor):
            def extract_feature_single_band(self, current_time: int, band: int) -> np.ndarray:
                full_vec = super().extract_feature_single_band(current_time, band)
                indices = [FEATURE_NAMES.index(f) for f in feat_subset]
                return full_vec[indices]

        # Generator using custom extractor
        generator = XGBoostDatasetGenerator(config)

        # Collect train and val sets
        all_train_X, all_train_y = [], []
        for s in train_seeds:
            env = RFEnvironment(config)
            obs = env.reset(seed=s)
            ext = CustomExtractor(num_bands=config.num_bands)
            rng = np.random.default_rng(s)
            step_idx = 0
            while not env.is_terminated:
                ext.update(obs)
                band = int(rng.integers(0, config.num_bands)) if rng.random() < 0.5 else (step_idx % config.num_bands)
                dwell = int(rng.choice([1, 2, 3]))
                action = Action(band, dwell)
                feat = ext.extract_feature_single_band(obs.current_time, band)
                obs, reward, terminated, info = env.step(action)
                dwell_sum = info.get("dwell_summary")
                has_tp = any(slot.is_true_positive for slot in dwell_sum.slot_outcomes) if dwell_sum else (obs.result == DetectionResult.HIT)
                all_train_X.append(feat)
                all_train_y.append(1 if has_tp else 0)
                step_idx += 1

        all_val_X, all_val_y = [], []
        for s in val_seeds:
            env = RFEnvironment(config)
            obs = env.reset(seed=s)
            ext = CustomExtractor(num_bands=config.num_bands)
            rng = np.random.default_rng(s)
            step_idx = 0
            while not env.is_terminated:
                ext.update(obs)
                band = int(rng.integers(0, config.num_bands)) if rng.random() < 0.5 else (step_idx % config.num_bands)
                dwell = int(rng.choice([1, 2, 3]))
                action = Action(band, dwell)
                feat = ext.extract_feature_single_band(obs.current_time, band)
                obs, reward, terminated, info = env.step(action)
                dwell_sum = info.get("dwell_summary")
                has_tp = any(slot.is_true_positive for slot in dwell_sum.slot_outcomes) if dwell_sum else (obs.result == DetectionResult.HIT)
                all_val_X.append(feat)
                all_val_y.append(1 if has_tp else 0)
                step_idx += 1

        X_train = np.array(all_train_X, dtype=np.float32)
        y_train = np.array(all_train_y, dtype=int)
        X_val = np.array(all_val_X, dtype=np.float32)
        y_val = np.array(all_val_y, dtype=int)

        model = XGBoostBandPredictor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, feature_names=feat_subset
        )
        val_metrics = model.fit(X_train, y_train, X_val, y_val)

        # Test on 10 test seeds
        test_metrics_list = []
        for s in test_seeds:
            env_test = RFEnvironment(config)
            opt = ActionOptimizer(num_bands=config.num_bands, allowed_dwells=[1, 2, 3], repeat_penalty_weight=0.15, dwell_penalty_weight=0.05, max_consecutive_scans=3)
            ext_test = CustomExtractor(num_bands=config.num_bands)
            sched = XGBoostScheduler(model=model, num_bands=config.num_bands, optimizer=opt, feature_extractor=ext_test)
            res = runner.run_episode(env_test, sched, seed=s)
            m = calculate_baseline_metrics(res, env_test.emitter_registry)
            test_metrics_list.append(m)

        agg = aggregate_metrics_across_seeds(test_metrics_list)

        ablation_results[name] = {
            "val_roc_auc": val_metrics.get("roc_auc", 0.0),
            "val_pr_auc": val_metrics.get("pr_auc", 0.0),
            "interception_rate": agg["interception_rate"],
            "average_intercept_time": agg["average_intercept_time"],
            "dwell_efficiency": agg["dwell_efficiency"],
        }

    return ablation_results


def run_full_validation():
    print("=" * 85)
    print("SIH26055 — Phase 3 Hardening & Adaptive Validation Execution")
    print("=" * 85)

    config_path = Path("configs/default.yaml")
    config = load_config(config_path)
    runner = EpisodeRunner()

    train_seeds = list(range(100, 120))
    val_seeds = list(range(120, 125))
    test_seeds = list(range(0, 10))

    # 1. Train standard model
    print("\n[Step 1] Loading / Training Standard XGBoost Model...")
    model_path = Path("models/xgboost_model.json")
    if model_path.exists():
        model = XGBoostBandPredictor.load(model_path)
    else:
        model, report = train_xgboost_pipeline(config, train_seeds, val_seeds, save_path=model_path)

    # 2. Run episodes across test seeds
    print("\n[Step 2] Executing 10 Test Episodes for Open-Loop and XGBoost...")
    ol_results = []
    xgb_results = []
    for s in test_seeds:
        # Open Loop
        env_ol = RFEnvironment(config)
        sched_ol = OpenLoopScheduler(num_bands=config.num_bands, dwell_time=1)
        ol_results.append(runner.run_episode(env_ol, sched_ol, seed=s))

        # XGBoost
        env_xgb = RFEnvironment(config)
        opt = ActionOptimizer(num_bands=config.num_bands, allowed_dwells=[1, 2, 3], repeat_penalty_weight=0.15, dwell_penalty_weight=0.05, max_consecutive_scans=3)
        sched_xgb = XGBoostScheduler(model=model, num_bands=config.num_bands, optimizer=opt)
        xgb_results.append(runner.run_episode(env_xgb, sched_xgb, seed=s))

    # 3. Analyze Dwell & Hit Inflation
    print("\n[Step 3] Performing Dwell Decomposition & Hit Inflation Analysis...")
    analysis = analyze_dwell_and_hit_inflation(ol_results, xgb_results, config)

    print("\n--- 1. SCANNED DWELL-SLOT DECOMPOSITION ---")
    print(f"{'Slot Category':<32} | {'Open-Loop Baseline':<24} | {'XGBoost Adaptive':<24}")
    print("-" * 85)
    ol = analysis["open_loop"]
    xb = analysis["xgboost"]
    print(f"{'Total Scanned Dwell Slots':<32} | {ol['total_scanned_slots'][0]:.1f} ± {ol['total_scanned_slots'][1]:.1f}        | {xb['total_scanned_slots'][0]:.1f} ± {xb['total_scanned_slots'][1]:.1f}")
    print(f"{'Total Scheduling Decisions':<32} | {ol['total_decisions'][0]:.1f} ± {ol['total_decisions'][1]:.1f}        | {xb['total_decisions'][0]:.1f} ± {xb['total_decisions'][1]:.1f}")
    print(f"{'Observable Signal Dwell Slots':<32} | {ol['observable_signal_slots'][0]:.1f} ± {ol['observable_signal_slots'][1]:.1f}        | {xb['observable_signal_slots'][0]:.1f} ± {xb['observable_signal_slots'][1]:.1f}")
    print(f"{'Idle / Noise Dwell Slots':<32} | {ol['idle_noise_slots'][0]:.1f} ± {ol['idle_noise_slots'][1]:.1f}        | {xb['idle_noise_slots'][0]:.1f} ± {xb['idle_noise_slots'][1]:.1f}")
    print(f"{'True Positive (TP) Slots':<32} | {ol['tp_slots'][0]:.1f} ± {ol['tp_slots'][1]:.1f}        | {xb['tp_slots'][0]:.1f} ± {xb['tp_slots'][1]:.1f}")
    print(f"{'False Negative (FN) Slots':<32} | {ol['fn_slots'][0]:.1f} ± {ol['fn_slots'][1]:.1f}        | {xb['fn_slots'][0]:.1f} ± {xb['fn_slots'][1]:.1f}")
    print(f"{'False Positive (FP) Slots':<32} | {ol['fp_slots'][0]:.1f} ± {ol['fp_slots'][1]:.1f}        | {xb['fp_slots'][0]:.1f} ± {xb['fp_slots'][1]:.1f}")
    print(f"{'True Negative (TN) Slots':<32} | {ol['tn_slots'][0]:.1f} ± {ol['tn_slots'][1]:.1f}        | {xb['tn_slots'][0]:.1f} ± {xb['tn_slots'][1]:.1f}")

    print("\n--- 2. DWELL DURATION DISTRIBUTION (%) ---")
    print(f"    Open-Loop Dwells: {ol['dwell_choices_distribution']}")
    print(f"    XGBoost Dwells:   {xb['dwell_choices_distribution']}")

    print("\n--- 3. REPEATED DETECTION & HIT INFLATION ANALYSIS ---")
    print(f"{'Metric':<40} | {'Open-Loop':<20} | {'XGBoost':<20}")
    print("-" * 85)
    print(f"{'Unique Opportunities Intercepted':<40} | {ol['unique_opps_intercepted'][0]:.1f} ± {ol['unique_opps_intercepted'][1]:.1f}      | {xb['unique_opps_intercepted'][0]:.1f} ± {xb['unique_opps_intercepted'][1]:.1f}")
    print(f"{'Total True Positive Detections':<40} | {ol['total_tp_detections'][0]:.1f} ± {ol['total_tp_detections'][1]:.1f}      | {xb['total_tp_detections'][0]:.1f} ± {xb['total_tp_detections'][1]:.1f}")
    print(f"{'TP Detections / Intercepted Opp':<40} | {ol['tps_per_intercepted_opp'][0]:.2f} ± {ol['tps_per_intercepted_opp'][1]:.2f}      | {xb['tps_per_intercepted_opp'][0]:.2f} ± {xb['tps_per_intercepted_opp'][1]:.2f}")
    print(f"{'Repeated Detections on Same Opp':<40} | {ol['repeated_detections_per_opp'][0]:.1f} ± {ol['repeated_detections_per_opp'][1]:.1f}      | {xb['repeated_detections_per_opp'][0]:.1f} ± {xb['repeated_detections_per_opp'][1]:.1f}")

    print("\n--- 4. EXPLOITATION VS EXPLORATION PROFILE ---")
    print(f"{'Scanning Behavior':<40} | {'Open-Loop':<20} | {'XGBoost':<20}")
    print("-" * 85)
    print(f"{'Scans on Previously HIT Bands':<40} | {ol['pct_scans_prev_hit_band'][0]*100:.1f}% ± {ol['pct_scans_prev_hit_band'][1]*100:.1f}%   | {xb['pct_scans_prev_hit_band'][0]*100:.1f}% ± {xb['pct_scans_prev_hit_band'][1]*100:.1f}%")
    print(f"{'Scans on Previously Unsuccessful Bands':<40} | {ol['pct_scans_prev_unsucc_band'][0]*100:.1f}% ± {ol['pct_scans_prev_unsucc_band'][1]*100:.1f}%   | {xb['pct_scans_prev_unsucc_band'][0]*100:.1f}% ± {xb['pct_scans_prev_unsucc_band'][1]*100:.1f}%")
    print(f"{'Scans on Never-Before-Scanned Bands':<40} | {ol['pct_scans_never_scanned_band'][0]*100:.1f}% ± {ol['pct_scans_never_scanned_band'][1]*100:.1f}%   | {xb['pct_scans_never_scanned_band'][0]*100:.1f}% ± {xb['pct_scans_never_scanned_band'][1]*100:.1f}%")
    print(f"{'Unique Frequency Bands Visited':<40} | {ol['unique_bands_scanned'][0]:.1f} / 20            | {xb['unique_bands_scanned'][0]:.1f} / 20")
    print(f"{'Max Consecutive Scans on Same Band':<40} | {ol['max_consecutive_scans'][0]:.1f}                | {xb['max_consecutive_scans'][0]:.1f}")

    # 4. Run Frequency Change Adaptation Experiment
    print("\n[Step 4] Running Frequency-Change Adaptation Experiment...")
    adapt_results = run_frequency_change_adaptation_experiment(model)
    print(f"{'Scheduler':<15} | {'First Scan After Change':<25} | {'First Detection Latency':<25}")
    print("-" * 70)
    for name, res in adapt_results.items():
        print(f"{name:<15} | {res['scan_latency_slots']} slots after t=2000     | {res['detection_latency_slots']} slots after t=2000")

    # 5. Run Time Feature Ablation
    print("\n[Step 5] Running Time-Feature Ablation Experiment...")
    ablation = run_time_feature_ablation(config, train_seeds, val_seeds, test_seeds)
    print(f"{'Feature Configuration':<30} | {'Val ROC-AUC':<12} | {'Interception Rate':<20} | {'Avg Delay':<18} | {'Dwell Efficiency':<18}")
    print("-" * 105)
    for name, res in ablation.items():
        ir_str = f"{res['interception_rate'][0]*100:.2f}% ± {res['interception_rate'][1]*100:.2f}%"
        delay_str = f"{res['average_intercept_time'][0]:.2f} ± {res['average_intercept_time'][1]:.2f} slots"
        eff_str = f"{res['dwell_efficiency'][0]*100:.2f}% ± {res['dwell_efficiency'][1]*100:.2f}%"
        print(f"{name:<30} | {res['val_roc_auc']:<12.4f} | {ir_str:<20} | {delay_str:<18} | {eff_str:<18}")

    print("\n" + "=" * 85)
    print("Phase 3 Validation & Hardening Analysis Completed!")
    print("=" * 85)


if __name__ == "__main__":
    run_full_validation()
