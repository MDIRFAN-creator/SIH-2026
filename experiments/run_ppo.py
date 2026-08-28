"""
Phase 5 Reinforcement Learning (PPO) Training & 4-Way Benchmark Pipeline (SIH26055).

Performs:
1. PPO Policy Training on Training Seeds (100..119) with Periodic Validation (120..124).
2. Dynamic Frequency-Hopping Adaptation Experiment (Open-Loop vs XGBoost vs LinUCB vs PPO).
3. 4-Way Comprehensive Benchmark on 10 Unseen Test Seeds (0..9).
4. Visual Artifact Generation (Training Curves, Exploration Diagnostics, 4-Way Benchmark, Adaptation Timeline, 4-Way Trajectories).
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
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
from models import XGBoostBandPredictor
from optimizers import ActionOptimizer
from rl import ActionEncoder, PPOAgent, PPOConfig, RFRLGymEnv, RLRewardCalculator, RLRewardConfig, RLStateExtractor
from runners import EpisodeResult, EpisodeRunner
from schedulers import LinUCBScheduler, OpenLoopScheduler, PPOScheduler, XGBoostScheduler
from training import train_xgboost_pipeline
from visualization import (
    plot_4way_benchmark_comparison,
    plot_4way_trajectories,
    plot_frequency_hopping_adaptation,
    plot_ppo_exploration_diagnostics,
    plot_ppo_training_curves,
)


def train_ppo_agent(
    env_config: EnvironmentConfig,
    train_seeds: List[int],
    val_seeds: List[int],
    n_episodes: int = 120,
    steps_per_update: int = 512,
    artifacts_dir: str = "artifacts/ppo",
) -> Tuple[PPOAgent, Dict[str, List[float]]]:
    """
    Train PPO Agent on training seeds with validation checkpoints.
    """
    print(f"\n[1] Initializing PPO Agent and Environment Wrapper...")
    encoder = ActionEncoder(num_bands=env_config.num_bands, dwell_values=[1, 2, 3])
    state_extractor = RLStateExtractor(num_bands=env_config.num_bands, max_dwell=3)

    ppo_config = PPOConfig(
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.20,
        ent_coef=0.02,
        vf_coef=0.50,
        max_grad_norm=0.50,
        n_epochs=8,
        batch_size=64,
        hidden_dim=128,
        seed=42,
    )

    agent = PPOAgent(
        state_dim=state_extractor.state_dim,
        action_dim=encoder.num_actions,
        config=ppo_config,
    )

    reward_config = RLRewardConfig(
        hit_reward=1.0,
        miss_penalty=-0.05,
        false_alarm_penalty=-0.50,
        dwell_cost=0.05,
        repetition_penalty=0.10,
        repetition_threshold=2,
    )

    gym_env = RFRLGymEnv(
        env_config=env_config,
        reward_config=reward_config,
        dwell_values=[1, 2, 3],
    )

    training_history: Dict[str, List[float]] = {
        "rewards": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
    }

    best_val_reward = -float("inf")
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    best_model_path = Path(artifacts_dir) / "best_ppo_model.pt"

    buffer_states: List[np.ndarray] = []

    buffer_actions: List[int] = []
    buffer_log_probs: List[float] = []
    buffer_rewards: List[float] = []
    buffer_values: List[float] = []
    buffer_dones: List[bool] = []

    print(f"[2] Training PPO for {n_episodes} episodes on seeds {train_seeds[0]}..{train_seeds[-1]}...")
    total_env_steps = 0

    for ep in range(1, n_episodes + 1):
        seed = train_seeds[(ep - 1) % len(train_seeds)]
        state, _ = gym_env.reset(seed=seed)
        ep_reward = 0.0
        terminated = False

        while not terminated:
            action_id, log_prob, value = agent.select_action(state, deterministic=False)
            next_state, reward, terminated, _, _ = gym_env.step(action_id)

            buffer_states.append(state)
            buffer_actions.append(action_id)
            buffer_log_probs.append(log_prob)
            buffer_rewards.append(reward)
            buffer_values.append(value)
            buffer_dones.append(terminated)

            ep_reward += reward
            total_env_steps += 1
            state = next_state

            # Perform PPO update if buffer is full or episode ends
            if len(buffer_states) >= steps_per_update or (terminated and len(buffer_states) >= 128):
                if terminated:
                    last_val = 0.0
                else:
                    _, _, last_val = agent.select_action(next_state, deterministic=False)

                advs, returns = agent.compute_gae(
                    rewards=buffer_rewards,
                    values=buffer_values,
                    dones=buffer_dones,
                    last_value=last_val,
                )

                update_metrics = agent.update(
                    states=np.array(buffer_states, dtype=np.float32),
                    actions=np.array(buffer_actions, dtype=np.int64),
                    old_log_probs=np.array(buffer_log_probs, dtype=np.float32),
                    returns=returns,
                    advantages=advs,
                )

                # Clear rollout buffer
                buffer_states.clear()
                buffer_actions.clear()
                buffer_log_probs.clear()
                buffer_rewards.clear()
                buffer_values.clear()
                buffer_dones.clear()

        training_history["rewards"].append(ep_reward)
        if "update_metrics" in locals():
            training_history["policy_loss"].append(update_metrics["policy_loss"])
            training_history["value_loss"].append(update_metrics["value_loss"])
            training_history["entropy"].append(update_metrics["entropy"])
            training_history["approx_kl"].append(update_metrics["approx_kl"])
            training_history["clip_fraction"].append(update_metrics["clip_fraction"])
        else:
            training_history["policy_loss"].append(0.0)
            training_history["value_loss"].append(0.0)
            training_history["entropy"].append(np.log(60))
            training_history["approx_kl"].append(0.0)
            training_history["clip_fraction"].append(0.0)

        if ep % 10 == 0 or ep == n_episodes:
            recent_avg = np.mean(training_history["rewards"][-10:])
            print(f"  Episode {ep:3d}/{n_episodes} | Steps: {total_env_steps:5d} | Ep Return: {ep_reward:7.1f} | 10-Ep Avg: {recent_avg:7.1f} | Entropy: {training_history['entropy'][-1]:.3f}")

        # Validation Checkpoint
        if ep % 20 == 0 or ep == n_episodes:
            val_returns = []
            for vs in val_seeds:
                v_state, _ = gym_env.reset(seed=vs)
                v_term = False
                v_ret = 0.0
                while not v_term:
                    v_act, _, _ = agent.select_action(v_state, deterministic=True)
                    v_state, v_r, v_term, _, _ = gym_env.step(v_act)
                    v_ret += v_r
                val_returns.append(v_ret)
            mean_val_ret = float(np.mean(val_returns))
            print(f"    --> [Val Checkpoint Ep {ep}] Validation Return (Seeds {val_seeds[0]}..{val_seeds[-1]}): {mean_val_ret:.1f} ± {np.std(val_returns):.1f}")
            if mean_val_ret > best_val_reward:
                best_val_reward = mean_val_ret
                agent.save(str(best_model_path))

    # Load best checkpoint
    if best_model_path.exists():
        agent.load(str(best_model_path))
        print(f"  -> Successfully loaded best checkpoint with validation return: {best_val_reward:.1f}")

    return agent, training_history


def run_frequency_hopping_adaptation_experiment(
    trained_xgb_model: XGBoostBandPredictor,
    trained_ppo_agent: PPOAgent,
) -> Dict[str, Dict[str, Tuple[int, int]]]:
    """
    Evaluate dynamic frequency hopping detection latency for Open-Loop, XGBoost, LinUCB, and PPO.
    """
    print(f"\n[3] Running Dynamic Frequency-Hopping Adaptation Experiment...")
    runner = EpisodeRunner()

    change_scenarios = [
        {"change_time": 1000, "dest_band": 14, "name": "t=1000, B14"},
        {"change_time": 2000, "dest_band": 7, "name": "t=2000, B7"},
        {"change_time": 3000, "dest_band": 18, "name": "t=3000, B18"},
    ]

    results_table: Dict[str, Dict[str, Tuple[int, int]]] = {}

    for scen in change_scenarios:
        t_change = scen["change_time"]
        dest_b = scen["dest_band"]
        scen_name = scen["name"]
        results_table[scen_name] = {}

        dyn_config = EnvironmentConfig(
            num_bands=20,
            simulation_duration=4000,
            seed=2026,
            emitters=[
                {"emitter_id": "p_static_1", "emitter_type": "PERIODIC", "frequency_band": 3, "period": 15, "active_duration": 4},
                {"emitter_id": "p_static_2", "emitter_type": "PERIODIC", "frequency_band": 11, "period": 23, "active_duration": 6},
                {"emitter_id": f"dyn_hopper_b{dest_b}", "emitter_type": "PERIODIC", "frequency_band": dest_b, "period": 17, "active_duration": 5, "start_time": t_change},
            ],
        )

        schedulers: List[Tuple[str, Any]] = [
            ("Open-Loop Baseline", OpenLoopScheduler(num_bands=20, dwell_time=1)),
            ("XGBoost Adaptive", XGBoostScheduler(model=trained_xgb_model, num_bands=20, optimizer=ActionOptimizer(max_consecutive_scans=3))),
            ("Hardened LinUCB", LinUCBScheduler(num_bands=20, alpha=1.0, gamma=0.99, max_consecutive_scans=3, min_initial_pulls=1, seed=2026)),
            ("PPO Policy", PPOScheduler(agent=trained_ppo_agent, deterministic=True)),
        ]


        for s_name, scheduler in schedulers:
            env = RFEnvironment(dyn_config)
            res = runner.run_episode(env, scheduler, seed=2026)

            first_scan_slot: Optional[int] = None
            first_hit_slot: Optional[int] = None

            for rec in res.step_records:
                if rec.start_time >= t_change:
                    if rec.action.frequency_band == dest_b:
                        if first_scan_slot is None:
                            first_scan_slot = rec.start_time
                        if rec.observation.result == DetectionResult.HIT:
                            first_hit_slot = rec.start_time
                            break

            scan_lat = (first_scan_slot - t_change) if first_scan_slot is not None else 9999
            hit_lat = (first_hit_slot - t_change) if first_hit_slot is not None else 9999
            results_table[scen_name][s_name] = (scan_lat, hit_lat)

        print(f"  {scen_name:15s} | OL: {results_table[scen_name]['Open-Loop Baseline'][1]}s | XGB: {results_table[scen_name]['XGBoost Adaptive'][1]}s | LinUCB: {results_table[scen_name]['Hardened LinUCB'][1]}s | PPO: {results_table[scen_name]['PPO Policy'][1]}s")

    return results_table


def run_4way_benchmark(
    config: EnvironmentConfig,
    trained_xgb_model: XGBoostBandPredictor,
    trained_ppo_agent: PPOAgent,
    test_seeds: List[int],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, EpisodeResult]]:
    """
    Run 4-Way benchmark on unseen test seeds (0..9).
    """
    print(f"\n[4] Running 4-Way Head-to-Head Benchmark on {len(test_seeds)} Unseen Test Seeds ({test_seeds[0]}..{test_seeds[-1]})...")
    runner = EpisodeRunner()

    schedulers_factory = {
        "Open-Loop Baseline": lambda s: OpenLoopScheduler(num_bands=config.num_bands, dwell_time=1),
        "XGBoost Adaptive": lambda s: XGBoostScheduler(model=trained_xgb_model, num_bands=config.num_bands, optimizer=ActionOptimizer(max_consecutive_scans=3)),
        "Hardened LinUCB": lambda s: LinUCBScheduler(num_bands=config.num_bands, alpha=1.0, gamma=0.99, max_consecutive_scans=3, min_initial_pulls=1, seed=s),
        "PPO Policy": lambda s: PPOScheduler(agent=trained_ppo_agent, deterministic=True),
    }

    all_metrics: Dict[str, List[BaselineMetrics]] = {name: [] for name in schedulers_factory}
    all_telemetry: Dict[str, List[Dict[str, Any]]] = {name: [] for name in schedulers_factory}
    sample_episodes: Dict[str, EpisodeResult] = {}

    for s_name, factory in schedulers_factory.items():
        for s in test_seeds:
            env = RFEnvironment(config)
            scheduler = factory(s)
            res = runner.run_episode(env, scheduler, seed=s)
            m = calculate_baseline_metrics(res, env.emitter_registry)
            all_metrics[s_name].append(m)

            # Telemetry collection
            scanned_bands = set()
            max_consec, curr_consec, last_b = 0, 0, None
            consec_runs = []
            scans_hit_bands, scans_unhit_bands = 0, 0
            hit_bands_seen = set()
            band_counts = np.zeros(config.num_bands, dtype=int)

            for rec in res.step_records:
                b = rec.action.frequency_band
                scanned_bands.add(b)
                band_counts[b] += 1

                if b in hit_bands_seen:
                    scans_hit_bands += 1
                else:
                    scans_unhit_bands += 1

                if rec.observation.result == DetectionResult.HIT:
                    hit_bands_seen.add(b)

                if last_b is not None and b == last_b:
                    curr_consec += 1
                else:
                    if curr_consec > 0:
                        consec_runs.append(curr_consec)
                    curr_consec = 1
                last_b = b

            if curr_consec > 0:
                consec_runs.append(curr_consec)
            max_consec = max(consec_runs) if consec_runs else 0
            mean_run = float(np.mean(consec_runs)) if consec_runs else 0.0

            # Shannon Entropy
            probs = band_counts / max(1, np.sum(band_counts))
            non_zero = probs[probs > 0]
            entropy = float(-np.sum(non_zero * np.log(non_zero)))

            # Online reward
            rew_val = 0.0
            if hasattr(scheduler, "cumulative_reward"):
                rew_val = scheduler.cumulative_reward

            tot_scans = max(1, len(res.step_records))
            all_telemetry[s_name].append({
                "unique_bands": len(scanned_bands),
                "max_consecutive": max_consec,
                "mean_run_length": mean_run,
                "entropy": entropy,
                "scans_hit_pct": (scans_hit_bands / tot_scans) * 100.0,
                "scans_unhit_pct": (scans_unhit_bands / tot_scans) * 100.0,
                "reward": rew_val,
            })

            if s == 0:
                sample_episodes[s_name] = res

    # Aggregate summaries
    benchmark_summary: Dict[str, Dict[str, Any]] = {}
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
        rew = [t["reward"] for t in ts]

        benchmark_summary[s_name] = {
            "interception_rate": (float(np.mean(ir)), float(np.std(ir, ddof=1))),
            "unique_opportunities": (float(np.mean(opp)), float(np.std(opp, ddof=1))),
            "avg_intercept_time": (float(np.mean(delay)), float(np.std(delay, ddof=1))),
            "scenario_ttfd": (float(np.mean(ttfd)), float(np.std(ttfd, ddof=1))) if ttfd else (0.0, 0.0),
            "empirical_pd": (float(np.mean(pd)), float(np.std(pd, ddof=1))),
            "empirical_pfa": (float(np.mean(pfa)), float(np.std(pfa, ddof=1))),
            "dwell_efficiency": (float(np.mean(eff)), float(np.std(eff, ddof=1))),
            "total_tp": (float(np.mean(tp)), float(np.std(tp, ddof=1))),
            "tp_per_opp": (float(np.mean(tp_per_opp)), float(np.std(tp_per_opp, ddof=1))),
            "unique_bands": (float(np.mean(ub)), float(np.std(ub, ddof=1))),
            "max_consecutive": (float(np.mean(mc)), float(np.std(mc, ddof=1))),
            "mean_run_length": (float(np.mean(mrl)), float(np.std(mrl, ddof=1))),
            "entropy": (float(np.mean(ent)), float(np.std(ent, ddof=1))),
            "scans_hit_pct": (float(np.mean(s_hit)), float(np.std(s_hit, ddof=1))),
            "scans_unhit_pct": (float(np.mean(s_unhit)), float(np.std(s_unhit, ddof=1))),
            "reward": (float(np.mean(rew)), float(np.std(rew, ddof=1))),
        }

    return benchmark_summary, sample_episodes


def main() -> None:
    print("=" * 115)
    print("SIH26055 — Phase 5 Reinforcement Learning (PPO) Benchmark Pipeline")
    print("=" * 115)

    base_yaml = Path("configs/default.yaml")
    env_config = load_config(base_yaml) if base_yaml.exists() else EnvironmentConfig()

    # 1. Train XGBoost Baseline Model
    print("\n[*] Preparing Phase 3 XGBoost Adaptive Model...")
    xgb_model, _ = train_xgboost_pipeline(
        config=env_config,
        train_seeds=list(range(100, 120)),
        val_seeds=list(range(120, 125)),
    )

    # 2. Train PPO Agent
    train_seeds = list(range(100, 120))
    val_seeds = list(range(120, 125))
    test_seeds = list(range(0, 10))

    ppo_agent, ppo_history = train_ppo_agent(
        env_config=env_config,
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        n_episodes=60,
    )


    # 3. Dynamic Frequency Hopping Experiment
    hop_results = run_frequency_hopping_adaptation_experiment(
        trained_xgb_model=xgb_model,
        trained_ppo_agent=ppo_agent,
    )

    # 4. 4-Way Head-to-Head Benchmark on 10 Unseen Test Seeds (0..9)
    benchmark_summary, sample_episodes = run_4way_benchmark(
        config=env_config,
        trained_xgb_model=xgb_model,
        trained_ppo_agent=ppo_agent,
        test_seeds=test_seeds,
    )

    # 5. Print Comparison Table
    print("\n" + "=" * 125)
    print("FOUR-WAY BENCHMARK COMPARISON: Open Loop vs XGBoost vs Hardened LinUCB vs PPO (N = 10 Unseen Test Seeds)")
    print("=" * 125)
    header = f"{'Metric':<36s} | {'Open-Loop Baseline':<20s} | {'XGBoost Adaptive':<20s} | {'Hardened LinUCB':<20s} | {'PPO Policy':<20s}"
    print(header)
    print("-" * 125)

    rows = [
        ("Interception Rate", "interception_rate", lambda m, s: f"{m*100.0:.2f}% ± {s*100.0:.2f}%"),
        ("Unique Opportunities Intercepted", "unique_opportunities", lambda m, s: f"{m:.2f} ± {s:.2f}"),
        ("Average Intercept Time", "avg_intercept_time", lambda m, s: f"{m:.2f} ± {s:.2f} slots"),
        ("PRD Scenario TTFD", "scenario_ttfd", lambda m, s: f"{m:.2f} ± {s:.2f} slots"),
        ("Receiver Empirical Pd", "empirical_pd", lambda m, s: f"{m:.2f} ± {s:.2f}"),
        ("Receiver Empirical Pfa", "empirical_pfa", lambda m, s: f"{m:.2f} ± {s:.2f}"),
        ("Dwell Efficiency", "dwell_efficiency", lambda m, s: f"{m*100.0:.2f}% ± {s*100.0:.2f}%"),
        ("Total TP Detections (Slots)", "total_tp", lambda m, s: f"{m:.2f} ± {s:.2f}"),
        ("TP Detections / Intercepted Opp", "tp_per_opp", lambda m, s: f"{m:.2f} ± {s:.2f}"),
        ("Unique Frequency Bands Scanned", "unique_bands", lambda m, s: f"{m:.2f} ± {s:.2f}"),
        ("Max Consecutive Band Scans", "max_consecutive", lambda m, s: f"{m:.2f} ± {s:.2f}"),
        ("Mean Consecutive Run Length", "mean_run_length", lambda m, s: f"{m:.2f} ± {s:.2f}"),
        ("Band-Selection Shannon Entropy", "entropy", lambda m, s: f"{m:.2f} ± {s:.2f}"),
        ("Scans on Previously Hit Bands", "scans_hit_pct", lambda m, s: f"{m:.2f}% ± {s:.2f}%"),
        ("Scans on Unsuccessful Bands", "scans_unhit_pct", lambda m, s: f"{m:.2f}% ± {s:.2f}%"),
    ]

    for label, key, fmt in rows:
        ol = fmt(*benchmark_summary["Open-Loop Baseline"][key])
        xgb = fmt(*benchmark_summary["XGBoost Adaptive"][key])
        lin = fmt(*benchmark_summary["Hardened LinUCB"][key])
        ppo = fmt(*benchmark_summary["PPO Policy"][key])
        print(f"{label:<36s} | {ol:<20s} | {xgb:<20s} | {lin:<20s} | {ppo:<20s}")
    print("=" * 125)

    # 6. Generate Publication Visualizations
    print("\n[5] Generating Visualizations...")
    plot_ppo_training_curves(ppo_history, save_path=Path("phase5_ppo_training_curve.png"))
    print("    -> Saved phase5_ppo_training_curve.png")

    ppo_sched = PPOScheduler(agent=ppo_agent, deterministic=True)
    env_eval = RFEnvironment(env_config)
    runner = EpisodeRunner()
    res_ppo = runner.run_episode(env_eval, ppo_sched, seed=0)

    plot_ppo_exploration_diagnostics(
        band_selection_counts=ppo_sched.band_selection_counts,
        entropy_history=ppo_history["entropy"],
        save_path=Path("phase5_ppo_exploration_diagnostics.png"),
    )
    print("    -> Saved phase5_ppo_exploration_diagnostics.png")

    plot_4way_benchmark_comparison(benchmark_summary, save_path=Path("phase5_4way_benchmark_comparison.png"))
    print("    -> Saved phase5_4way_benchmark_comparison.png")

    plot_frequency_hopping_adaptation(hop_results, save_path=Path("phase5_frequency_hopping_adaptation.png"))
    print("    -> Saved phase5_frequency_hopping_adaptation.png")

    plot_4way_trajectories(
        open_loop_res=sample_episodes["Open-Loop Baseline"],
        xgboost_res=sample_episodes["XGBoost Adaptive"],
        linucb_res=sample_episodes["Hardened LinUCB"],
        ppo_res=sample_episodes["PPO Policy"],
        max_time=200,
        save_path=Path("phase5_4way_trajectory_comparison.png"),
    )
    print("    -> Saved phase5_4way_trajectory_comparison.png")

    print("\n" + "=" * 115)
    print("Phase 5 PPO Benchmark Pipeline Completed Successfully!")
    print("=" * 115)


if __name__ == "__main__":
    main()
