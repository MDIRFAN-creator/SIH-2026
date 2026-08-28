"""
Phase 3 Experiment Runner: XGBoost + Optimization Adaptive Scheduler Benchmark (SIH26055).

Workflows:
1. Supervised dataset generation from exploratory RF simulation episodes.
2. XGBoostBandPredictor model training and validation evaluation.
3. Model serialization to models/xgboost_model.json.
4. Head-to-head fair benchmark comparison: OpenLoopScheduler vs XGBoostScheduler across 10 identical test seeds.
5. Visualizations: feature importance ranking and scanning trajectory comparison.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Tuple
import numpy as np

from environment import EnvironmentConfig, RFEnvironment, load_config
from evaluation import (
    BaselineMetrics,
    aggregate_metrics_across_seeds,
    calculate_baseline_metrics,
    extract_emitter_opportunities,
)
from models import XGBoostBandPredictor
from optimizers import ActionOptimizer
from runners import EpisodeResult, EpisodeRunner
from schedulers import OpenLoopScheduler, XGBoostScheduler
from training import train_xgboost_pipeline
from visualization import plot_feature_importances, plot_scheduler_comparison


def run_xgboost_benchmark() -> None:
    print("=" * 85)
    print("SIH26055 — Phase 3: XGBoost + Optimization Adaptive Scheduler Benchmark")
    print("=" * 85)

    config_path = Path("configs/default.yaml")
    config = load_config(config_path)
    runner = EpisodeRunner()

    # Seed splits ensuring 0 temporal or data contamination
    train_seeds = list(range(100, 120))  # 20 episodes
    val_seeds = list(range(120, 125))    # 5 episodes
    test_seeds = list(range(0, 10))      # 10 test episodes (identical to Phase 2 test seeds)

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "xgboost_model.json"

    # -------------------------------------------------------------
    # 1. Train / Load XGBoost Model
    # -------------------------------------------------------------
    print(f"\n[1] Training XGBoost Band Predictor on {len(train_seeds)} exploration seeds...")
    print(f"    - Training Seeds: {train_seeds[0]}..{train_seeds[-1]}")
    print(f"    - Validation Seeds: {val_seeds[0]}..{val_seeds[-1]}")

    model, report = train_xgboost_pipeline(
        config=config,
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        save_path=model_path,
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )

    val_metrics = report["validation_metrics"]
    print("\n[2] XGBoost Model Offline Classification Performance:")
    print(f"    - Training Samples:        {report['train_samples']} (Positive Ratio: {report['train_positive_ratio']*100:.1f}%)")
    print(f"    - Validation Samples:      {report['val_samples']} (Positive Ratio: {report['val_positive_ratio']*100:.1f}%)")
    print(f"    - ROC-AUC Score:           {val_metrics.get('roc_auc', 0.0):.4f}")
    print(f"    - PR-AUC Score:            {val_metrics.get('pr_auc', 0.0):.4f}")
    print(f"    - Accuracy:                {val_metrics.get('accuracy', 0.0)*100:.2f}%")
    print(f"    - Precision:               {val_metrics.get('precision', 0.0):.4f}")
    print(f"    - Recall:                  {val_metrics.get('recall', 0.0):.4f}")
    print(f"    - F1 Score:                {val_metrics.get('f1', 0.0):.4f}")
    print(f"    - Model Saved To:          {model_path.resolve()}")

    # -------------------------------------------------------------
    # 2. Feature Importance Report & Plot
    # -------------------------------------------------------------
    print("\n[3] Model Feature Importance Ranking:")
    importances = report["feature_importances"]
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    for rank, (feat, score) in enumerate(sorted_importances, 1):
        print(f"    {rank:2d}. {feat:<22}: {score:.4f}")

    feat_plot_path = Path("xgboost_feature_importance.png")
    plot_feature_importances(importances, save_path=feat_plot_path)
    print(f"    -> Feature importance plot saved to: {feat_plot_path.resolve()}")

    # -------------------------------------------------------------
    # 3. Head-to-Head Fair Benchmark: Open Loop vs XGBoost Scheduler
    # -------------------------------------------------------------
    print(f"\n[4] Running Fair Head-to-Head Comparison on {len(test_seeds)} Unseen Test Seeds (0..9)...")
    print("    Both schedulers evaluate on the EXACT same scenario configuration and seeds.")

    open_loop_metrics_list: List[BaselineMetrics] = []
    xgboost_metrics_list: List[BaselineMetrics] = []

    open_loop_results: List[EpisodeResult] = []
    xgboost_results: List[EpisodeResult] = []

    for seed in test_seeds:
        # A. Open-Loop Baseline Run
        env_ol = RFEnvironment(config)
        sched_ol = OpenLoopScheduler(num_bands=config.num_bands, dwell_time=1)
        res_ol = runner.run_episode(env=env_ol, scheduler=sched_ol, seed=seed)
        m_ol = calculate_baseline_metrics(res_ol, env_ol.emitter_registry)
        open_loop_metrics_list.append(m_ol)
        open_loop_results.append(res_ol)

        # B. XGBoost Adaptive Scheduler Run
        env_xgb = RFEnvironment(config)
        optimizer = ActionOptimizer(
            num_bands=config.num_bands,
            allowed_dwells=[1, 2, 3],
            repeat_penalty_weight=0.15,
            dwell_penalty_weight=0.05,
            max_consecutive_scans=3,
        )
        sched_xgb = XGBoostScheduler(
            model=model,
            num_bands=config.num_bands,
            optimizer=optimizer,
        )
        res_xgb = runner.run_episode(env=env_xgb, scheduler=sched_xgb, seed=seed)
        m_xgb = calculate_baseline_metrics(res_xgb, env_xgb.emitter_registry)
        xgboost_metrics_list.append(m_xgb)
        xgboost_results.append(res_xgb)

    last_res_ol = open_loop_results[-1] if open_loop_results else None
    last_res_xgb = xgboost_results[-1] if xgboost_results else None

    agg_ol = aggregate_metrics_across_seeds(open_loop_metrics_list)
    agg_xgb = aggregate_metrics_across_seeds(xgboost_metrics_list)

    # -------------------------------------------------------------
    # 4. Print Comparative Results Table
    # -------------------------------------------------------------
    # Compute extended opportunity and band coverage metrics
    def compute_extended_stats(results_list: List[EpisodeResult], metrics_list: List[BaselineMetrics]):
        tp_totals = [m.tp_count for m in metrics_list]
        unique_opps = [m.successful_interceptions for m in metrics_list]
        tp_per_opp = [(m.tp_count / m.successful_interceptions) if m.successful_interceptions > 0 else 0.0 for m in metrics_list]
        unique_bands = [len(set(r.action.frequency_band for r in res.step_records)) for res in results_list]
        return (
            (float(np.mean(tp_totals)), float(np.std(tp_totals, ddof=1))),
            (float(np.mean(unique_opps)), float(np.std(unique_opps, ddof=1))),
            (float(np.mean(tp_per_opp)), float(np.std(tp_per_opp, ddof=1))),
            (float(np.mean(unique_bands)), float(np.std(unique_bands, ddof=1))),
        )

    ext_tp_ol, ext_opp_ol, ext_tpopp_ol, ext_bands_ol = compute_extended_stats(
        open_loop_results,
        open_loop_metrics_list,
    )
    ext_tp_xgb, ext_opp_xgb, ext_tpopp_xgb, ext_bands_xgb = compute_extended_stats(
        xgboost_results,
        xgboost_metrics_list,
    )

    print("\n" + "=" * 95)
    print("BENCHMARK COMPARISON: Open-Loop Baseline vs Phase 3 XGBoost Scheduler (N = 10 Seeds)")
    print("=" * 95)
    print(f"{'Metric':<40} | {'Open-Loop Baseline (Mean ± Std)':<25} | {'XGBoost Adaptive (Mean ± Std)':<25}")
    print("-" * 95)

    ir_ol = f"{agg_ol['interception_rate'][0]*100:.2f}% ± {agg_ol['interception_rate'][1]*100:.2f}%"
    ir_xgb = f"{agg_xgb['interception_rate'][0]*100:.2f}% ± {agg_xgb['interception_rate'][1]*100:.2f}%"
    print(f"{'Interception Rate':<40} | {ir_ol:<25} | {ir_xgb:<25}")

    ttfd_ol = f"{agg_ol['scenario_ttfd'][0]:.2f} ± {agg_ol['scenario_ttfd'][1]:.2f} slots"
    ttfd_xgb = f"{agg_xgb['scenario_ttfd'][0]:.2f} ± {agg_xgb['scenario_ttfd'][1]:.2f} slots"
    print(f"{'PRD Scenario TTFD':<40} | {ttfd_ol:<25} | {ttfd_xgb:<25}")

    delay_ol = f"{agg_ol['average_intercept_time'][0]:.2f} ± {agg_ol['average_intercept_time'][1]:.2f} slots"
    delay_xgb = f"{agg_xgb['average_intercept_time'][0]:.2f} ± {agg_xgb['average_intercept_time'][1]:.2f} slots"
    print(f"{'Avg Intercept Time':<40} | {delay_ol:<25} | {delay_xgb:<25}")

    pd_ol = f"{agg_ol['empirical_pd'][0]:.4f} ± {agg_ol['empirical_pd'][1]:.4f}"
    pd_xgb = f"{agg_xgb['empirical_pd'][0]:.4f} ± {agg_xgb['empirical_pd'][1]:.4f}"
    print(f"{'Receiver Empirical Pd':<40} | {pd_ol:<25} | {pd_xgb:<25}")

    pfa_ol = f"{agg_ol['empirical_pfa'][0]:.4f} ± {agg_ol['empirical_pfa'][1]:.4f}"
    pfa_xgb = f"{agg_xgb['empirical_pfa'][0]:.4f} ± {agg_xgb['empirical_pfa'][1]:.4f}"
    print(f"{'Receiver Empirical Pfa':<40} | {pfa_ol:<25} | {pfa_xgb:<25}")

    eff_ol = f"{agg_ol['dwell_efficiency'][0]*100:.2f}% ± {agg_ol['dwell_efficiency'][1]*100:.2f}%"
    eff_xgb = f"{agg_xgb['dwell_efficiency'][0]*100:.2f}% ± {agg_xgb['dwell_efficiency'][1]*100:.2f}%"
    print(f"{'Dwell Efficiency':<40} | {eff_ol:<25} | {eff_xgb:<25}")

    tp_ol_str = f"{ext_tp_ol[0]:.1f} ± {ext_tp_ol[1]:.1f}"
    tp_xgb_str = f"{ext_tp_xgb[0]:.1f} ± {ext_tp_xgb[1]:.1f}"
    print(f"{'Total TP Detections (Slots)':<40} | {tp_ol_str:<25} | {tp_xgb_str:<25}")

    opp_ol_str = f"{ext_opp_ol[0]:.1f} ± {ext_opp_ol[1]:.1f}"
    opp_xgb_str = f"{ext_opp_xgb[0]:.1f} ± {ext_opp_xgb[1]:.1f}"
    print(f"{'Unique Opportunities Intercepted':<40} | {opp_ol_str:<25} | {opp_xgb_str:<25}")

    tpopp_ol_str = f"{ext_tpopp_ol[0]:.2f} ± {ext_tpopp_ol[1]:.2f}"
    tpopp_xgb_str = f"{ext_tpopp_xgb[0]:.2f} ± {ext_tpopp_xgb[1]:.2f}"
    print(f"{'TP Detections / Intercepted Opp':<40} | {tpopp_ol_str:<25} | {tpopp_xgb_str:<25}")

    bands_ol_str = f"{ext_bands_ol[0]:.1f} / 20"
    bands_xgb_str = f"{ext_bands_xgb[0]:.1f} / 20"
    print(f"{'Unique Bands Scanned':<40} | {bands_ol_str:<25} | {bands_xgb_str:<25}")

    print("=" * 95)
    print("Note: Receiver operating characteristics (Pd, Pfa) remain consistent across both schedulers.")

    # -------------------------------------------------------------
    # 5. Generate Visual Comparison Plot
    # -------------------------------------------------------------
    print("\n[5] Generating Trajectory Comparison Plot...")
    comp_plot_path = Path("xgboost_vs_openloop.png")
    if last_res_ol is not None and last_res_xgb is not None:
        plot_scheduler_comparison(
            open_loop_result=last_res_ol,
            xgboost_result=last_res_xgb,
            time_range=(0, 200),
            save_path=comp_plot_path,
        )
        print(f"    -> Trajectory comparison saved to: {comp_plot_path.resolve()}")

    print("\n" + "=" * 85)
    print("Phase 3 XGBoost Benchmark Completed Successfully!")
    print("=" * 85)


if __name__ == "__main__":
    run_xgboost_benchmark()
