"""
Baseline evaluation metrics calculator for SIH26055 (Phase 2).

This module calculates formal EW evaluation metrics from an EpisodeResult:
- Interception Rate (fraction of burst opportunities intercepted)
- Time to First Detection (TTFD) per emitter
- Average Intercept Delay per opportunity
- Receiver-level Empirical Pd and Pfa
- Confusion matrix counts (TP, FN, FP, TN)
- Dwell Efficiency
- Multi-seed statistical aggregation (mean, std)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from environment.emitters import EmitterRegistry
from environment.types import DetectionResult
from runners.episode_runner import EpisodeResult


@dataclass
class EmitterOpportunity:
    """Represents a single contiguous window where an emitter is transmitting & observable."""
    emitter_id: str
    frequency_band: int
    start_time: int
    end_time: int  # inclusive
    intercepted: bool = False
    first_intercept_time: Optional[int] = None

    @property
    def duration(self) -> int:
        return self.end_time - self.start_time + 1


@dataclass
class BaselineMetrics:
    """
    Comprehensive baseline metric container for a single episode.
    
    Metrics are strictly separated into:
    - Receiver-level detection physics (Empirical Pd, Empirical Pfa, Confusion Matrix)
    - Scheduler-level frequency-time resource allocation (Interception Rate, TTFD, Average Intercept Time, Dwell Efficiency)
    """
    scheduler_name: str
    seed: Optional[int]
    total_simulation_slots: int
    total_decisions: int

    # Action-level scan outcomes
    action_hits: int
    action_misses: int
    action_false_alarms: int

    # Slot-level confusion matrix
    tp_count: int  # True Positive (signal present, observable, detected)
    fn_count: int  # False Negative (signal present, observable, NOT detected)
    fp_count: int  # False Positive (noise on empty/unobservable slot detected)
    tn_count: int  # True Negative (empty/unobservable slot correctly quiet)

    # Receiver statistical metrics (detection behavior on scanned slots)
    empirical_pd: float   # TP / (TP + FN)
    empirical_pfa: float  # FP / (FP + TN)

    # Scheduler EW performance metrics (frequency-time scanning effectiveness)
    interception_opportunities: int       # Total ground-truth burst windows in the scenario horizon
    successful_interceptions: int         # Burst windows intercepted with at least one True Positive
    interception_rate: float              # successful_interceptions / interception_opportunities
    scenario_ttfd: Optional[int]          # PRD TTFD: Earliest detection time across entire scenario - scenario start (0)
    emitter_ttfd: Dict[str, Optional[int]]# Per-emitter TTFD: Earliest detection time - emitter start_time
    average_intercept_time: Optional[float] # PRD Average Intercept Time: Mean (first_detection_in_burst - burst_start)
    dwell_efficiency: float               # (TP + FN) / total_scanned_dwell_slots

    @property
    def time_to_first_detection(self) -> Dict[str, Optional[int]]:
        """Backward-compatible alias for emitter_ttfd."""
        return self.emitter_ttfd

    @property
    def average_intercept_delay(self) -> Optional[float]:
        """Backward-compatible alias for average_intercept_time."""
        return self.average_intercept_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scheduler_name": self.scheduler_name,
            "seed": self.seed,
            "total_simulation_slots": self.total_simulation_slots,
            "total_decisions": self.total_decisions,
            "action_hits": self.action_hits,
            "action_misses": self.action_misses,
            "action_false_alarms": self.action_false_alarms,
            "tp_count": self.tp_count,
            "fn_count": self.fn_count,
            "fp_count": self.fp_count,
            "tn_count": self.tn_count,
            "empirical_pd": self.empirical_pd,
            "empirical_pfa": self.empirical_pfa,
            "interception_opportunities": self.interception_opportunities,
            "successful_interceptions": self.successful_interceptions,
            "interception_rate": self.interception_rate,
            "scenario_ttfd": self.scenario_ttfd,
            "emitter_ttfd": self.emitter_ttfd,
            "average_intercept_time": self.average_intercept_time,
            "dwell_efficiency": self.dwell_efficiency,
        }


def extract_emitter_opportunities(
    emitter_registry: EmitterRegistry,
    simulation_duration: int,
    num_bands: int = 20,
) -> List[EmitterOpportunity]:
    """
    Extract all distinct contiguous (transmitting + observable) opportunity windows
    for every registered emitter across the entire simulation duration.
    """
    opportunities: List[EmitterOpportunity] = []

    for emitter in emitter_registry.get_emitters():
        in_opp = False
        opp_start = -1
        opp_band = -1

        for t in range(simulation_duration):
            if emitter.is_active_in_scenario(t):
                st = emitter.get_state(t)
                is_opp = st.is_transmitting and st.is_observable
                band = st.frequency_band

                if is_opp:
                    if not in_opp:
                        in_opp = True
                        opp_start = t
                        opp_band = band
                    elif band != opp_band:
                        # Frequency hopped while observable -> finish current opportunity and start new
                        opportunities.append(
                            EmitterOpportunity(
                                emitter_id=emitter.emitter_id,
                                frequency_band=opp_band,
                                start_time=opp_start,
                                end_time=t - 1,
                            )
                        )
                        opp_start = t
                        opp_band = band
                else:
                    if in_opp:
                        in_opp = False
                        opportunities.append(
                            EmitterOpportunity(
                                emitter_id=emitter.emitter_id,
                                frequency_band=opp_band,
                                start_time=opp_start,
                                end_time=t - 1,
                            )
                        )
            else:
                if in_opp:
                    in_opp = False
                    opportunities.append(
                        EmitterOpportunity(
                            emitter_id=emitter.emitter_id,
                            frequency_band=opp_band,
                            start_time=opp_start,
                            end_time=t - 1,
                        )
                    )

        if in_opp:
            opportunities.append(
                EmitterOpportunity(
                    emitter_id=emitter.emitter_id,
                    frequency_band=opp_band,
                    start_time=opp_start,
                    end_time=simulation_duration - 1,
                )
            )

    # Sort opportunities chronologically by start_time
    opportunities.sort(key=lambda o: (o.start_time, o.emitter_id))
    return opportunities


def calculate_baseline_metrics(
    episode_result: EpisodeResult,
    emitter_registry: EmitterRegistry,
) -> BaselineMetrics:
    """
    Compute comprehensive EW baseline metrics from an executed EpisodeResult.
    
    Args:
        episode_result: Recorded EpisodeResult from EpisodeRunner.
        emitter_registry: Scenario emitter registry used for ground-truth comparison.
        
    Returns:
        BaselineMetrics: Fully computed metrics dataclass.
    """
    # 1. Action-level counts
    action_hits = sum(1 for r in episode_result.step_records if r.observation.result == DetectionResult.HIT)
    action_fa = sum(1 for r in episode_result.step_records if r.observation.result == DetectionResult.FALSE_ALARM)
    action_misses = sum(1 for r in episode_result.step_records if r.observation.result == DetectionResult.MISS)

    # 2. Slot-level confusion matrix counts
    tp_count = 0
    fn_count = 0
    fp_count = 0
    tn_count = 0

    for dwell in episode_result.dwell_history:
        for slot in dwell.slot_outcomes:
            if slot.is_true_positive:
                tp_count += 1
            elif slot.is_false_negative:
                fn_count += 1
            elif slot.is_false_positive:
                fp_count += 1
            elif slot.is_true_negative:
                tn_count += 1

    # 3. Receiver empirical probabilities
    eligible_signal_slots = tp_count + fn_count
    eligible_noise_slots = fp_count + tn_count

    empirical_pd = (tp_count / eligible_signal_slots) if eligible_signal_slots > 0 else 0.0
    empirical_pfa = (fp_count / eligible_noise_slots) if eligible_noise_slots > 0 else 0.0

    # 4. Interception Opportunities and Interception Rate
    # The scenario horizon is defined strictly by the environment configuration (simulation_duration)
    # to guarantee that the ground-truth opportunity denominator is 100% scheduler-independent.
    scenario_horizon = (
        episode_result.environment_config.simulation_duration
        if episode_result.environment_config
        else episode_result.total_time_slots
    )
    num_bands = (
        episode_result.environment_config.num_bands
        if episode_result.environment_config
        else 20
    )

    opportunities = extract_emitter_opportunities(
        emitter_registry=emitter_registry,
        simulation_duration=scenario_horizon,
        num_bands=num_bands,
    )

    # Check which opportunities were intercepted by receiver scans
    for opp in opportunities:
        for dwell in episode_result.dwell_history:
            # Overlap check between dwell [start_time, end_time] and opportunity [opp.start_time, opp.end_time]
            if dwell.scanned_band == opp.frequency_band:
                overlap_start = max(dwell.start_time, opp.start_time)
                overlap_end = min(dwell.end_time, opp.end_time)

                if overlap_start <= overlap_end:
                    # Check if any slot within the overlap generated a true positive
                    for slot in dwell.slot_outcomes:
                        if overlap_start <= slot.time_slot <= overlap_end and slot.is_true_positive:
                            opp.intercepted = True
                            if opp.first_intercept_time is None or slot.time_slot < opp.first_intercept_time:
                                opp.first_intercept_time = slot.time_slot

    total_opps = len(opportunities)
    successful_opps = sum(1 for o in opportunities if o.intercepted)
    interception_rate = (successful_opps / total_opps) if total_opps > 0 else 0.0

    # 5. TTFD (PRD Section 25.6: TTFD = time of first successful detection - scenario start)
    all_hits = [
        o.first_intercept_time
        for o in opportunities
        if o.intercepted and o.first_intercept_time is not None
    ]
    scenario_ttfd = min(all_hits) if all_hits else None

    # Per-emitter TTFD: Earliest detection of emitter - emitter.start_time
    emitter_ttfd: Dict[str, Optional[int]] = {}
    for emitter in emitter_registry.get_emitters():
        e_id = emitter.emitter_id
        e_hits = [
            o.first_intercept_time
            for o in opportunities
            if o.emitter_id == e_id and o.intercepted and o.first_intercept_time is not None
        ]
        emitter_ttfd[e_id] = (min(e_hits) - emitter.start_time) if e_hits else None

    # 6. Average Intercept Time (PRD Section 24.2: mean of first detection - opportunity start)
    delays = [
        (o.first_intercept_time - o.start_time)
        for o in opportunities
        if o.intercepted and o.first_intercept_time is not None
    ]
    avg_intercept_time = (float(np.mean(delays))) if delays else None

    # 7. Dwell Efficiency (observable signal scanned slots / total dwell slots scanned)
    total_slots_scanned = sum(r.action.dwell_time for r in episode_result.step_records)
    dwell_efficiency = (eligible_signal_slots / total_slots_scanned) if total_slots_scanned > 0 else 0.0

    return BaselineMetrics(
        scheduler_name=episode_result.scheduler_name,
        seed=episode_result.seed,
        total_simulation_slots=episode_result.total_time_slots,
        total_decisions=episode_result.total_decisions,
        action_hits=action_hits,
        action_misses=action_misses,
        action_false_alarms=action_fa,
        tp_count=tp_count,
        fn_count=fn_count,
        fp_count=fp_count,
        tn_count=tn_count,
        empirical_pd=empirical_pd,
        empirical_pfa=empirical_pfa,
        interception_opportunities=total_opps,
        successful_interceptions=successful_opps,
        interception_rate=interception_rate,
        scenario_ttfd=scenario_ttfd,
        emitter_ttfd=emitter_ttfd,
        average_intercept_time=avg_intercept_time,
        dwell_efficiency=dwell_efficiency,
    )


def aggregate_metrics_across_seeds(metrics_list: List[BaselineMetrics]) -> Dict[str, Tuple[float, float]]:
    """
    Compute mean and sample standard deviation across multiple independent seed runs.
    
    Returns:
        Dict mapping metric_name -> (mean, std)
    """
    if not metrics_list:
        raise ValueError("metrics_list cannot be empty")

    numeric_keys = [
        "interception_rate",
        "empirical_pd",
        "empirical_pfa",
        "dwell_efficiency",
        "action_hits",
        "action_false_alarms",
        "action_misses",
    ]

    aggregates: Dict[str, Tuple[float, float]] = {}

    for key in numeric_keys:
        values = [getattr(m, key) for m in metrics_list]
        mean_val = float(np.mean(values))
        std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        aggregates[key] = (mean_val, std_val)

    # Average Intercept Time (ignoring None)
    delays = [m.average_intercept_time for m in metrics_list if m.average_intercept_time is not None]
    if delays:
        aggregates["average_intercept_time"] = (
            float(np.mean(delays)),
            float(np.std(delays, ddof=1)) if len(delays) > 1 else 0.0,
        )
        aggregates["average_intercept_delay"] = aggregates["average_intercept_time"]
    else:
        aggregates["average_intercept_time"] = (0.0, 0.0)
        aggregates["average_intercept_delay"] = (0.0, 0.0)

    # Scenario TTFD (ignoring None)
    scenario_ttfds = [m.scenario_ttfd for m in metrics_list if m.scenario_ttfd is not None]
    if scenario_ttfds:
        aggregates["scenario_ttfd"] = (
            float(np.mean(scenario_ttfds)),
            float(np.std(scenario_ttfds, ddof=1)) if len(scenario_ttfds) > 1 else 0.0,
        )
    else:
        aggregates["scenario_ttfd"] = (0.0, 0.0)

    return aggregates

