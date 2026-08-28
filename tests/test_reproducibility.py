"""
Reproducibility tests for the RF Environment (SIH26055 - Phase 1).
"""

from environment.config import EnvironmentConfig
from environment.rf_environment import RFEnvironment
from environment.types import Action


def test_seeded_episode_determinism():
    """Verify that identical seeds and action sequences produce identical simulation runs."""
    config = EnvironmentConfig(
        num_bands=20,
        simulation_duration=1000,
        seed=42,
        emitters=[
            {
                "emitter_id": "periodic_1",
                "emitter_type": "PERIODIC",
                "frequency_band": 4,
                "period": 20,
                "active_duration": 4,
            },
            {
                "emitter_id": "agile_rand_1",
                "emitter_type": "AGILE_RANDOM",
                "allowed_bands": [2, 6, 10, 14],
                "hop_period": 10,
            },
            {
                "emitter_id": "intermittent_1",
                "emitter_type": "INTERMITTENT",
                "frequency_band": 8,
                "scan_period": 30,
                "observable_duration": 5,
            },
        ],
    )

    action_sequence = [
        Action(frequency_band=(i * 3) % 20, dwell_time=[1, 2, 3, 5][i % 4])
        for i in range(50)
    ]

    # Run Episode 1
    env1 = RFEnvironment(config)
    obs1 = env1.reset(seed=42)
    results1 = []
    rewards1 = []
    for act in action_sequence:
        o, r, t, info = env1.step(act)
        results1.append((o.result, o.current_time, info["overall_result"]))
        rewards1.append(r)

    # Run Episode 2 (identical seed)
    env2 = RFEnvironment(config)
    obs2 = env2.reset(seed=42)
    results2 = []
    rewards2 = []
    for act in action_sequence:
        o, r, t, info = env2.step(act)
        results2.append((o.result, o.current_time, info["overall_result"]))
        rewards2.append(r)

    assert obs1 == obs2
    assert results1 == results2
    assert rewards1 == rewards2

    # Verify internal dwell histories match
    assert len(env1.episode_dwell_history) == len(env2.episode_dwell_history)
    for d1, d2 in zip(env1.episode_dwell_history, env2.episode_dwell_history):
        assert d1.start_time == d2.start_time
        assert d1.end_time == d2.end_time
        assert d1.scanned_band == d2.scanned_band
        assert d1.overall_result == d2.overall_result
        for s1, s2 in zip(d1.slot_outcomes, d2.slot_outcomes):
            assert s1.time_slot == s2.time_slot
            assert s1.detected == s2.detected
            assert s1.result == s2.result


def test_different_seeds_produce_variation():
    """Verify that different seeds produce distinct stochastic trajectories."""
    config = EnvironmentConfig(
        num_bands=20,
        simulation_duration=2000,
        emitters=[
            {
                "emitter_id": "agile_rand",
                "emitter_type": "AGILE_RANDOM",
                "allowed_bands": [0, 1, 2, 3, 4, 5, 6, 7],
                "hop_period": 5,
            }
        ],
    )

    action_sequence = [Action(frequency_band=b % 8, dwell_time=2) for b in range(100)]

    env_a = RFEnvironment(config)
    env_a.reset(seed=42)
    results_a = [env_a.step(a)[0].result for a in action_sequence]

    env_b = RFEnvironment(config)
    env_b.reset(seed=999)
    results_b = [env_b.step(a)[0].result for a in action_sequence]

    # Verify that the two runs with different seeds are not identical
    assert results_a != results_b


def test_scheduler_action_invariance_on_emitter_ground_truth():
    """
    Verify fairness / non-interference guarantee:
    Running completely different scheduler action sequences on the same scenario configuration
    and seed produces the EXACT same underlying emitter ground truth across the entire timeline.
    
    This ensures future scheduler benchmarks (Open Loop vs LinUCB vs XGBoost) are 100% fair.
    """
    config = EnvironmentConfig(
        num_bands=20,
        simulation_duration=1000,
        seed=42,
        emitters=[
            {
                "emitter_id": "periodic_b5",
                "emitter_type": "PERIODIC",
                "frequency_band": 5,
                "period": 20,
                "active_duration": 4,
            },
            {
                "emitter_id": "agile_rand",
                "emitter_type": "AGILE_RANDOM",
                "allowed_bands": [1, 4, 7, 10, 13, 16],
                "hop_period": 10,
                "emitter_seed": 777,
            },
            {
                "emitter_id": "agile_pred",
                "emitter_type": "AGILE_PREDICTABLE",
                "band_sequence": [2, 8, 14, 18],
                "hop_period": 15,
            },
            {
                "emitter_id": "intermittent_b9",
                "emitter_type": "INTERMITTENT",
                "frequency_band": 9,
                "scan_period": 40,
                "observable_duration": 6,
            },
            {
                "emitter_id": "dynamic_b17",
                "emitter_type": "PERIODIC",
                "frequency_band": 17,
                "period": 25,
                "active_duration": 5,
                "start_time": 200,
            },
        ],
    )

    # Scheduler 1: Cyclic sweep with alternating dwells [1, 2, 3, 5]
    env1 = RFEnvironment(config)
    env1.reset(seed=42)
    current_band = 0
    dwells = [1, 2, 3, 5]
    step_idx = 0
    while env1.current_time < 500:
        env1.step(Action(frequency_band=current_band, dwell_time=dwells[step_idx % 4]))
        current_band = (current_band + 1) % 20
        step_idx += 1

    # Scheduler 2: High-dwell camping scanner on Band 12 (dwell 5 always)
    env2 = RFEnvironment(config)
    env2.reset(seed=42)
    while env2.current_time < 500:
        env2.step(Action(frequency_band=12, dwell_time=5))

    # Scheduler 3: Fast 1-slot hopping scanner on even bands
    env3 = RFEnvironment(config)
    env3.reset(seed=42)
    b = 0
    while env3.current_time < 500:
        env3.step(Action(frequency_band=(b * 2) % 20, dwell_time=1))
        b += 1

    # Compare ground truth across all 20 bands and 500 time slots (10,000 bins total)
    for t in range(500):
        for band in range(20):
            gt1 = env1.get_ground_truth_at(t, band)
            gt2 = env2.get_ground_truth_at(t, band)
            gt3 = env3.get_ground_truth_at(t, band)

            assert gt1.is_transmitting == gt2.is_transmitting == gt3.is_transmitting, (
                f"Ground truth transmission mismatch at (t={t}, band={band})"
            )
            assert gt1.is_observable == gt2.is_observable == gt3.is_observable, (
                f"Ground truth observability mismatch at (t={t}, band={band})"
            )
            assert gt1.active_emitter_ids == gt2.active_emitter_ids == gt3.active_emitter_ids, (
                f"Ground truth active emitter IDs mismatch at (t={t}, band={band})"
            )

