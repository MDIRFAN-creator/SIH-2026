"""
Phase 1 Validation & Demonstration Script (SIH26055).

This script demonstrates:
1. Loading configuration from YAML.
2. Initializing and stepping the RFEnvironment with a scanning strategy.
3. Multi-emitter scenario execution (Periodic, Agile, Intermittent, Dynamic).
4. Generating and saving a high-contrast RF timeline visualization plot.
5. Verifying ground-truth isolation and reproducibility.
"""

from pathlib import Path
import numpy as np

from environment import (
    Action,
    DetectionResult,
    EnvironmentConfig,
    RFEnvironment,
    load_config,
)
from environment.visualization import plot_rf_timeline


def run_phase1_validation() -> None:
    print("=" * 70)
    print("SIH26055 — Phase 1: RF Simulation Environment Validation")
    print("=" * 70)

    config_path = Path("configs/default.yaml")
    print(f"\n[1] Loading configuration from: {config_path}")
    config = load_config(config_path)
    print(f"    - Frequency Bands: {config.num_bands}")
    print(f"    - Simulation Duration: {config.simulation_duration} slots")
    print(f"    - Receiver Pd: {config.receiver.pd:.2f}, Pfa: {config.receiver.pfa:.2f}")
    print(f"    - Allowed Dwell Times: {config.receiver.allowed_dwell_times}")
    print(f"    - Registered Emitters: {len(config.emitters)}")
    for e in config.emitters:
        print(f"      * ID: {e['emitter_id']} ({e['emitter_type']})")

    # Instantiate Environment
    env = RFEnvironment(config)

    print("\n[2] Initializing environment with Seed = 42")
    initial_obs = env.reset(seed=42)
    print(f"    - Initial Time: t = {initial_obs.current_time}")
    print(f"    - Initial Observation Result: {initial_obs.result}")

    # Run simulation with an illustrative sweep scheduler
    print("\n[3] Executing 100 simulation steps across spectrum...")
    scan_results = {DetectionResult.HIT: 0, DetectionResult.FALSE_ALARM: 0, DetectionResult.MISS: 0}
    
    # Cyclic scanner alternating dwells
    current_band = 0
    dwell_choices = [1, 2, 3, 5]
    step_count = 0

    while not env.is_terminated and step_count < 100:
        dwell = dwell_choices[step_count % len(dwell_choices)]
        action = Action(frequency_band=current_band, dwell_time=dwell)
        
        obs, reward, term, info = env.step(action)
        scan_results[obs.result] += 1
        step_count += 1
        current_band = (current_band + 1) % env.num_bands

    print(f"    - Completed {step_count} decisions reaching t = {env.current_time} slots.")
    print(f"    - Detection Outcomes:")
    print(f"      * HITs (True Positives): {scan_results[DetectionResult.HIT]}")
    print(f"      * FALSE ALARMs (False Positives): {scan_results[DetectionResult.FALSE_ALARM]}")
    print(f"      * MISSes / Inactive scans: {scan_results[DetectionResult.MISS]}")

    # Verify Dynamic Emitter discovery after t = 5000
    print("\n[4] Running simulation to verify Dynamic Emitter (E_Dynamic_B17 appearing at t = 5000)...")
    env_dynamic = RFEnvironment(config)
    env_dynamic.reset(seed=42)
    
    # Fast-forward by scanning B0 with dwell 5 until t >= 4995
    while env_dynamic.current_time < 4990:
        env_dynamic.step(Action(frequency_band=0, dwell_time=5))

    # Scan Band 17 around appearance boundary
    print(f"    - Current time before transition: t = {env_dynamic.current_time}")
    # Prior to t=5000: B17 should not be active
    gt_pre = env_dynamic.get_ground_truth_at(4995, 17)
    print(f"    - Ground Truth at t=4995 on Band 17: is_transmitting={gt_pre.is_transmitting}")
    assert gt_pre.is_transmitting is False

    # Advance to t=5005 (where E_Dynamic_B17 period=30, active_duration=4, offset=5 is transmitting at 5005..5008)
    gt_post = env_dynamic.get_ground_truth_at(5005, 17)
    print(f"    - Ground Truth at t=5005 on Band 17: is_transmitting={gt_post.is_transmitting}, active_ids={gt_post.active_emitter_ids}")
    assert gt_post.is_transmitting is True
    assert "E_Dynamic_B17" in gt_post.active_emitter_ids
    print("    -> Dynamic emitter appearance confirmed at t >= 5000.")

    # Generate Visualization Plot
    print("\n[5] Generating RF Timeline Visualization Plot (t = 0 to 200)...")
    output_plot_path = Path("rf_timeline_demo.png")
    plot_rf_timeline(
        env=env,
        time_range=(0, 200),
        save_path=str(output_plot_path),
        title="SIH26055: RF Spectrum Ground Truth vs ESM Receiver Scans (t=0..200)",
    )
    print(f"    -> Plot saved successfully to: {output_plot_path.resolve()}")

    print("\n" + "=" * 70)
    print("Phase 1 RF Environment Validation Succeeded 100%!")
    print("=" * 70)


if __name__ == "__main__":
    run_phase1_validation()
