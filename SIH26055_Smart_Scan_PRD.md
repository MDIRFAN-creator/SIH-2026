# SIH26055 --- Smart Scan Strategy for Electronic Warfare

## Product Requirements Document (PRD) and Implementation Specification

**Version:** 1.0\
**Date:** 27 August 2026\
**Problem Statement:** SIH26055 --- Smart Scan strategy for Electronic
Warfare\
**Organization:** DRDO\
**Project Type:** Software / simulation / machine-learning scheduler

------------------------------------------------------------------------

## 1. Executive Summary

This project develops a simulation-driven, machine-learning-based
adaptive scheduler for an Electronic Support (ESM) receiver operating
over a wide RF spectrum while having limited instantaneous bandwidth.

The central problem is a scheduling problem:

> The receiver cannot observe the entire spectrum simultaneously, so it
> must continuously decide **which frequency to observe, when to observe
> it, and for how long**.

A conventional open-loop scheduler follows a predetermined scanning
pattern. Our proposed system learns from receiver observations (hits,
misses, false alarms, timing and historical activity) and adaptively
reallocates scanning effort toward promising or uncertain frequency-time
regions.

The project will use a common RF simulation environment and a common
scheduler interface so that multiple approaches can be implemented and
compared fairly:

1.  Open-loop frequency sweep --- baseline
2.  XGBoost + optimization --- ML benchmark
3.  Contextual Bandit (LinUCB) --- primary proposed approach
4.  Reinforcement Learning --- optional/future benchmark

All algorithms will face the same generated RF scenarios and will be
evaluated using the same metrics.

------------------------------------------------------------------------

## 2. Important Source / Scope Note

The uploaded SIH 2026 PDF confirms the official listing:

-   **SIH26055**
-   **DRDO**
-   **Clean & Green Technology**
-   **Smart Scan strategy for Electronic Warfare**

The uploaded PDF is a 9-page list of 172 software problem statements and
does **not** contain the detailed SIH26055 technical description.
Therefore, this document separates:

-   **Official listing:** directly supported by the uploaded SIH PDF.
-   **Project specification:** the agreed technical interpretation and
    engineering design developed by the team for the hackathon.

Do not present the implementation assumptions below as DRDO
specifications unless independently verified from the detailed official
problem statement.

Source: `SIH_2026_Problem_Statements.pdf`.

------------------------------------------------------------------------

# 3. Problem Understanding

## 3.1 Simplified Problem

Imagine an ESM receiver that can listen to only one small portion of a
very large RF spectrum at a time.

The receiver has to search:

``` text
B1 B2 B3 B4 B5 ... B20
```

An emitter may appear in one band, disappear, return later, or change
frequency.

A fixed scanner might do:

``` text
B1 → B2 → B3 → ... → B20 → B1 → ...
```

The problem is that equal scanning effort may be wasted on inactive
bands while an important emitter is active elsewhere.

The proposed system instead learns:

``` text
Where have signals appeared?
When do they tend to appear?
Which bands have recently been missed?
Which bands are uncertain?
Where should the receiver look next?
How long should it stay there?
```

The output of the scheduler is therefore:

``` text
Next frequency + dwell time
```

------------------------------------------------------------------------

# 4. Project Objectives

## Primary Objective

Develop and evaluate an adaptive frequency-time scheduler that improves
receiver resource allocation compared with a conventional open-loop
scanning strategy.

## Secondary Objectives

-   Model a realistic-enough RF search environment for algorithm
    comparison.
-   Simulate periodic emitters.
-   Simulate frequency-agile emitters.
-   Simulate intermittent/spatially scanning emitters.
-   Simulate new emitters appearing during an episode.
-   Include imperfect detection and false alarms.
-   Train/update schedulers using receiver observations rather than
    hidden ground truth.
-   Measure interception performance and scanning efficiency.
-   Provide a reusable environment interface for multiple algorithms.
-   Produce reproducible experiments with controlled random seeds.

------------------------------------------------------------------------

# 5. Non-Goals

The first prototype will NOT attempt to:

-   Build a physical ESM receiver.
-   Implement RF hardware.
-   Perform real-world electromagnetic propagation.
-   Identify real military platforms.
-   Build a classified emitter library.
-   Reproduce DRDO's actual operational receiver implementation.
-   Use real classified RF data.
-   Claim operational deployment readiness.

The prototype is a simulation and algorithm-evaluation framework.

------------------------------------------------------------------------

# 6. High-Level System Architecture

``` text
                  ┌─────────────────────────────┐
                  │     RF ENVIRONMENT          │
                  │                             │
                  │  Emitter Models             │
                  │  Ground Truth               │
                  │  Time/Frequency State       │
                  │  Interceptability           │
                  └─────────────┬───────────────┘
                                │
                                ↓
                     ┌────────────────────┐
                     │  ESM RECEIVER      │
                     │                    │
                     │ limited bandwidth  │
                     │ Pd / Pfa model     │
                     └─────────┬──────────┘
                               │
                         HIT / MISS
                               │
                               ↓
                   ┌────────────────────────┐
                   │ OBSERVATION / MEMORY   │
                   │                        │
                   │ history                │
                   │ hit rate               │
                   │ recency                │
                   │ timing information     │
                   └───────────┬────────────┘
                               │
                               ↓
              ┌─────────────────────────────────┐
              │         SCHEDULER               │
              │                                 │
              │ Open Loop                      │
              │ XGBoost + Optimization         │
              │ Contextual Bandit (LinUCB)     │
              │ RL (optional)                  │
              └──────────────┬──────────────────┘
                             │
                             ↓
                    frequency + dwell
                             │
                             └──────→ Receiver
```

------------------------------------------------------------------------

# 7. Core Design Principle

## The environment owns truth.

The RF environment knows:

-   actual emitter state
-   actual transmission
-   actual frequency
-   actual interceptability
-   scenario configuration

## The scheduler does NOT receive truth.

The scheduler receives only information that a receiver could reasonably
observe:

-   selected frequency
-   receiver observation
-   hit/miss
-   false alarm
-   time
-   historical observations
-   engineered context features

This prevents data leakage.

------------------------------------------------------------------------

# 8. RF Environment Specification

## 8.1 Frequency Representation

Version 1 will represent the spectrum as 20 abstract frequency bands:

``` text
B1, B2, B3, ..., B20
```

These are logical bands rather than real RF frequencies.

The abstraction allows us to study the scheduling problem without
requiring physical RF hardware.

------------------------------------------------------------------------

## 8.2 Time Representation

Time is represented as discrete slots:

``` text
t = 1, 2, 3, ..., T
```

Initial simulation:

``` text
T = 10,000 slots per episode
```

The value is configurable.

A shorter number of slots may be used for interactive visualization.

------------------------------------------------------------------------

## 8.3 Receiver Instantaneous Bandwidth

Initial configuration:

``` text
Total spectrum: 20 bands
Receiver simultaneous coverage: 1 band
```

Thus, at a scheduling decision, the receiver can observe one selected
band.

Future experiments may allow:

``` text
1 band
2 bands
4 bands
```

to study how receiver capability affects scheduler performance.

------------------------------------------------------------------------

# 9. Emitter Model

The simulator will support multiple emitter behaviours.

## 9.1 Periodic Emitter

A periodic emitter becomes active according to a repeatable time
pattern.

Example:

``` text
B5:
t = 10
t = 20
t = 30
t = 40
...
```

Parameters:

``` text
frequency
period
active duration
start time
end time
```

Purpose:

> Test whether the scheduler learns temporal regularity.

------------------------------------------------------------------------

## 9.2 Frequency-Agile Emitter

An emitter changes its frequency over time.

Example:

``` text
B3 → B8 → B5 → B11 → B4 → ...
```

Two modes will be supported:

### Predictable agile

The sequence has a hidden pattern.

### Random agile

The frequency changes according to a stochastic process.

Purpose:

> Test adaptation to changing frequency behaviour and robustness when
> future frequency is difficult to predict.

------------------------------------------------------------------------

## 9.3 Spatially Scanning / Intermittent Emitter

The first implementation abstracts spatial behaviour rather than
simulating electromagnetic propagation.

The emitter may be:

``` text
TRANSMITTING
but not currently observable
```

or:

``` text
TRANSMITTING
and observable
```

Example:

``` text
t1 → observable
t2 → not observable
t3 → not observable
t4 → observable
```

This creates intermittent interception opportunities.

------------------------------------------------------------------------

## 9.4 Dynamic / Newly Appearing Emitter

An emitter can appear after the simulation has already started.

Example:

``` text
t < 5000:
E1 → B5

t = 5000:
E2 appears → B17
```

Purpose:

> Test whether an adaptive scheduler can discover and react to
> previously unknown activity.

------------------------------------------------------------------------

# 10. Ground Truth Representation

The environment maintains a hidden ground-truth representation.

Conceptually:

``` text
                B1 B2 B3 ... B20
t1               0  0  1  ... 0
t2               0  0  0  ... 1
t3               1  0  0  ... 0
...
tT               0  1  0  ... 0
```

Where:

``` text
1 = transmission
0 = no transmission
```

For intermittent emitters, the environment additionally tracks whether
the transmission is observable/interceptable.

The scheduler cannot directly access this matrix.

------------------------------------------------------------------------

# 11. Receiver Model

The receiver receives a scheduler action:

``` text
(frequency_band, dwell_time)
```

Example:

``` text
(B7, 3)
```

means:

> Observe band B7 for three time slots.

## Initial detection parameters

``` text
Probability of Detection (Pd) = 0.90
Probability of False Alarm (Pfa) = 0.02
```

These values are configurable and are not claimed to represent a real
DRDO receiver.

------------------------------------------------------------------------

# 12. Hit / Miss Logic

If an emitter is present and observable:

``` text
Actual signal
     ↓
Pd = 0.90
     ↓
90% probability → HIT
10% probability → MISS
```

If no signal is present:

``` text
No signal
     ↓
Pfa = 0.02
     ↓
98% → correct negative
2%  → FALSE ALARM
```

The detection model will later be made dependent on configurable
conditions such as SNR.

------------------------------------------------------------------------

# 13. Action Space

The scheduler chooses:

``` text
Action = (frequency, dwell)
```

Initial frequency choices:

``` text
20 bands
```

Initial dwell choices:

``` text
1, 2, 3, 5 slots
```

Therefore:

``` text
20 × 4 = 80 possible actions
```

The common action representation is:

``` python
Action(
    frequency_band=int,
    dwell_time=int
)
```

This makes the environment compatible with all scheduler types.

------------------------------------------------------------------------

# 14. Environment API

All algorithms will interact with the environment through a common
interface.

Conceptual API:

``` python
observation = env.reset(seed=42)

observation, reward, terminated, info = env.step(action)
```

The environment must expose enough information for a scheduler to make
its next decision while hiding ground truth.

Example observation:

``` python
{
    "time": 105,
    "scanned_band": 7,
    "dwell_time": 2,
    "result": "HIT",
    "recent_hit_rate": ...,
    "time_since_last_scan": ...,
}
```

Exact schema will be finalized during implementation.

------------------------------------------------------------------------

# 15. Observation Memory

The environment/scheduler interface will maintain historical
observations.

For each frequency band we can calculate features such as:

``` text
recent_hit_count
recent_miss_count
recent_hit_rate
time_since_last_scan
time_since_last_hit
number_of_recent_scans
estimated_activity
```

These features form the basis for the adaptive models.

------------------------------------------------------------------------

# 16. Scheduler Interface

Every scheduler must implement the same conceptual interface:

``` python
class BaseScheduler:

    def reset(self):
        pass

    def select_action(self, observation):
        pass

    def update(self, observation, action, result):
        pass
```

This means the environment does not care which algorithm is being used.

------------------------------------------------------------------------

# 17. Scheduler 1 --- Open-Loop Baseline

The baseline will use a predetermined scanning schedule.

Example:

``` text
B1 → B2 → B3 → ... → B20
             ↓
            B1
```

The baseline will have:

-   fixed band order
-   fixed dwell time
-   no learning
-   no adaptation based on observations

This establishes the reference performance.

------------------------------------------------------------------------

# 18. Scheduler 2 --- XGBoost + Optimization

Pipeline:

``` text
Observation history
        ↓
Feature engineering
        ↓
XGBoost model
        ↓
Predicted usefulness/reward
        ↓
Candidate-action evaluation
        ↓
Optimization / argmax
        ↓
frequency + dwell
```

Candidate actions:

``` text
(B1,1)
(B1,2)
...
(B20,5)
```

The XGBoost component estimates the expected outcome of an action.

The optimizer selects the highest-scoring action.

Purpose:

> Provide a conventional supervised-ML benchmark against which the
> contextual bandit can be compared.

------------------------------------------------------------------------

# 19. Scheduler 3 --- Contextual Bandit

## Primary Proposed Approach

Initial algorithm:

**LinUCB**

At each decision:

``` text
Context
   +
Candidate action
   ↓
Estimated reward
   +
Uncertainty/exploration bonus
   ↓
Select action
   ↓
Receiver observation
   ↓
Reward
   ↓
Update
```

The bandit naturally supports the exploration/exploitation problem.

### Context examples

``` text
recent hit rate
recent miss rate
time since last scan
time since last hit
estimated activity
current time
candidate dwell
```

### Action

``` text
(frequency_band, dwell_time)
```

### Feedback

``` text
HIT
MISS
FALSE ALARM
```

------------------------------------------------------------------------

# 20. Scheduler 4 --- Reinforcement Learning

RL will be treated as an additional benchmark/future extension.

Conceptual architecture:

``` text
State
 ↓
RL agent
 ↓
Action
 ↓
RF environment
 ↓
Reward
 ↓
Next state
```

Possible algorithms:

-   DQN for discrete actions
-   PPO for a more general policy

The RL implementation is lower priority for the internal hackathon.

Priority order:

``` text
1. Open Loop
2. XGBoost + Optimization
3. Contextual Bandit ⭐
4. RL
```

------------------------------------------------------------------------

# 21. Reward Design

The initial reward is a configurable engineering parameter.

A candidate reward can combine:

-   successful interception
-   false alarm
-   miss
-   intercept time
-   scanning/dwell cost

Conceptually:

``` text
Successful interception → positive reward
Fast interception       → additional positive value
False alarm             → negative reward
Miss                    → negative reward
Long/wasted dwell       → negative cost
```

Example initial formulation:

``` text
R =
  detection_reward
  - intercept_time_cost
  - false_alarm_cost
  - dwell_cost
```

The exact coefficients will be tuned experimentally.

The reward definition must remain consistent across ML schedulers where
comparison is intended.

------------------------------------------------------------------------

# 22. Interception Definition

An interception opportunity occurs when an emitter is
observable/interceptable.

Example:

``` text
t=100 → opportunity begins
t=101
t=102
t=103
t=104 → opportunity ends
```

If the receiver successfully detects the emitter at:

``` text
t=102
```

then:

``` text
successful interception
intercept time = 2 slots
```

If no successful detection occurs before the opportunity ends:

``` text
missed opportunity
```

------------------------------------------------------------------------

# 23. Evaluation Framework

Every scheduler must be tested on the same RF episodes.

``` text
                  SAME SCENARIOS
                       │
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
    Open Loop      XGBoost        Bandit
        │              │              │
        └──────────────┼──────────────┘
                       ↓
                  SAME METRICS
```

Random seeds will be controlled for reproducibility.

------------------------------------------------------------------------

# 24. Primary Evaluation Metrics

## 24.1 Interception Rate

``` text
Interception Rate =
successful interceptions /
total interception opportunities
```

Higher is better.

This is one of the principal measures of whether the adaptive scheduler
achieves its objective.

------------------------------------------------------------------------

## 24.2 Average Intercept Time

For each successful interception:

``` text
intercept_time =
first successful detection time
- opportunity start time
```

Then:

``` text
Average Intercept Time =
mean(intercept_time)
```

Lower is better.

------------------------------------------------------------------------

## 24.3 Probability of Detection (Pd)

``` text
Pd =
true positives /
(true positives + false negatives)
```

Higher is better.

------------------------------------------------------------------------

## 24.4 Probability of False Alarm (Pfa)

``` text
Pfa =
false positives /
(false positives + true negatives)
```

Lower is better.

------------------------------------------------------------------------

# 25. Secondary Evaluation Metrics

## 25.1 Sensitivity

Measure detection performance as environmental conditions become harder,
for example by varying SNR or detection probability.

Example experiment:

``` text
SNR:
-10 dB
-5 dB
 0 dB
+5 dB
+10 dB
```

Measure:

``` text
Pd vs SNR
```

------------------------------------------------------------------------

## 25.2 Average Reward

``` text
Average Reward =
sum(rewards) / number of decisions
```

Higher is better.

This is particularly important for the contextual bandit.

------------------------------------------------------------------------

## 25.3 Prediction Accuracy

Where the model explicitly produces an intercept/activity prediction:

``` text
Prediction Accuracy =
correct predictions / total predictions
```

The prediction target and tolerance window must be defined before
evaluation.

------------------------------------------------------------------------

## 25.4 Average Intercept-Time Error

``` text
error_i =
abs(predicted_intercept_time
    - actual_intercept_time)
```

Then:

``` text
Average error =
mean(error_i)
```

Lower is better.

------------------------------------------------------------------------

## 25.5 Dwell Efficiency

A useful additional metric:

``` text
Dwell Efficiency =
useful dwell time / total dwell time
```

Higher is better.

This measures whether receiver time is being spent productively.

------------------------------------------------------------------------

## 25.6 Time to First Detection

``` text
TTFD =
time of first successful detection
- scenario start
```

Lower is better.

------------------------------------------------------------------------

# 26. Evaluation Matrix

  Metric                         Open Loop   XGBoost   Contextual Bandit   RL   Direction
  ---------------------------- ----------- --------- ------------------- ---- -----------
  Interception Rate                      ✓         ✓                   ✓    ✓           ↑
  Average Intercept Time                 ✓         ✓                   ✓    ✓           ↓
  Probability of Detection               ✓         ✓                   ✓    ✓           ↑
  Probability of False Alarm             ✓         ✓                   ✓    ✓           ↓
  Sensitivity                            ✓         ✓                   ✓    ✓           ↑
  Average Reward                         ✓         ✓                   ✓    ✓           ↑
  Prediction Accuracy             optional         ✓                   ✓    ✓           ↑
  Intercept-Time Error            optional         ✓                   ✓    ✓           ↓
  Dwell Efficiency                       ✓         ✓                   ✓    ✓           ↑
  Time to First Detection                ✓         ✓                   ✓    ✓           ↓

------------------------------------------------------------------------

# 27. Test Scenarios

The environment must generate multiple scenario classes.

## Scenario A --- Periodic Emitter

Tests temporal learning.

``` text
B5 active every N slots
```

------------------------------------------------------------------------

## Scenario B --- Frequency-Agile Emitter

Tests adaptation to frequency changes.

``` text
B3 → B8 → B5 → B11 → ...
```

------------------------------------------------------------------------

## Scenario C --- Intermittent / Spatially Scanning Emitter

Tests intermittent interception opportunities.

------------------------------------------------------------------------

## Scenario D --- Multiple Emitters

Example:

``` text
E1 → B3
E2 → B7
E3 → B12
E4 → B18
```

Tests resource allocation.

------------------------------------------------------------------------

## Scenario E --- New Emitter

An unknown emitter appears during the episode.

Tests discovery and adaptation.

------------------------------------------------------------------------

## Scenario F --- Dynamic Environment

Emitter behaviour changes during an episode.

Tests online adaptation.

------------------------------------------------------------------------

# 28. Robustness Experiments

After the basic implementation works, vary:

## Number of Emitters

``` text
2 → 5 → 10 → 20
```

## Frequency Agility

``` text
slow switching → fast switching
```

## Detection Probability

``` text
0.95 → 0.90 → 0.80 → 0.70
```

## False Alarm Probability

``` text
0.01 → 0.05 → 0.10
```

## Receiver Bandwidth

``` text
1 → 2 → 4 bands
```

## Emitter Density

Low, medium and high activity environments.

The goal is to determine whether the adaptive scheduler remains useful
as the environment becomes harder.

------------------------------------------------------------------------

# 29. Training / Test Design

The system should not simply memorize a few fixed scenarios.

Generate many independent episodes.

Example:

``` text
Episode 1
Episode 2
...
Episode 1000
```

Training and testing must use different generated episodes.

Where possible, test scenarios should include parameter combinations not
seen during training.

Example:

``` text
Training periods:
10, 20, 30

Testing periods:
15, 25
```

This provides a more meaningful generalization test.

------------------------------------------------------------------------

# 30. Reproducibility

Every experiment must support:

``` python
seed=42
```

The same environment configuration and seed should generate the same
episode.

For fair comparison:

``` text
Open Loop       → Scenario Seed 42
XGBoost         → Scenario Seed 42
Bandit          → Scenario Seed 42
RL              → Scenario Seed 42
```

When stochastic algorithms are used, multiple training/evaluation seeds
should be reported.

------------------------------------------------------------------------

# 31. Statistical Evaluation

A single simulation run is not sufficient.

For each scenario class:

``` text
N independent episodes
```

should be evaluated.

Report:

``` text
mean
standard deviation
```

for major metrics.

Example:

``` text
Contextual Bandit
Interception Rate = mean ± std
Average Intercept Time = mean ± std
```

This avoids making conclusions from a lucky run.

------------------------------------------------------------------------

# 32. Visualization Requirements

The prototype should provide visual evidence of adaptation.

## 32.1 RF Spectrum Timeline

``` text
Time →
B1  ░░░░░░░░
B2  ░░█░░░░░
B3  ░░░░██░░
...
```

------------------------------------------------------------------------

## 32.2 Scheduler Scan Timeline

Show:

``` text
Time → B7 → B7 → B3 → B17 → B17 → B5
```

------------------------------------------------------------------------

## 32.3 Priority Evolution

For adaptive schedulers:

``` text
Band
B3  ███
B7  ███████
B17 █████████
```

and show how priority changes after observations.

------------------------------------------------------------------------

## 32.4 Detection Events

Display:

``` text
Emitter activity
Receiver scans
Successful hits
Misses
False alarms
```

on the same timeline.

------------------------------------------------------------------------

## 32.5 Algorithm Comparison

Bar/line charts for:

-   interception rate
-   average intercept time
-   Pd
-   Pfa
-   reward
-   dwell efficiency

------------------------------------------------------------------------

# 33. Recommended Project Structure

``` text
smart_scan/
│
├── environment/
│   ├── __init__.py
│   ├── config.py
│   ├── types.py
│   ├── emitters.py
│   ├── receiver.py
│   ├── rf_environment.py
│   └── observation.py
│
├── schedulers/
│   ├── __init__.py
│   ├── base.py
│   ├── open_loop.py
│   ├── xgboost_scheduler.py
│   ├── contextual_bandit.py
│   └── rl_scheduler.py
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   ├── evaluator.py
│   ├── scenarios.py
│   └── statistical_analysis.py
│
├── visualization/
│   ├── spectrum.py
│   ├── scheduler.py
│   └── comparison.py
│
├── experiments/
│   ├── run_baseline.py
│   ├── run_xgboost.py
│   ├── run_bandit.py
│   └── run_rl.py
│
├── tests/
│   ├── test_emitters.py
│   ├── test_receiver.py
│   ├── test_environment.py
│   ├── test_metrics.py
│   └── test_scheduler_interface.py
│
├── configs/
│   └── default.yaml
│
├── requirements.txt
├── README.md
└── main.py
```

------------------------------------------------------------------------

# 34. Implementation Phases

## Phase 1 --- RF Environment

Build:

-   configuration
-   frequency bands
-   time model
-   emitter models
-   ground truth
-   receiver
-   detection model
-   hit/miss
-   false alarms
-   environment API
-   reproducibility

**Deliverable:** a working RF simulator.

------------------------------------------------------------------------

## Phase 2 --- Open-Loop Baseline

Build:

-   fixed sweep
-   fixed dwell
-   logging
-   baseline evaluation

**Deliverable:** benchmark results.

------------------------------------------------------------------------

## Phase 3 --- XGBoost + Optimization

Build:

-   feature engineering
-   training dataset generation
-   XGBoost model
-   candidate-action scoring
-   action optimization

**Deliverable:** second scheduler.

------------------------------------------------------------------------

## Phase 4 --- Contextual Bandit

Build:

-   context generation
-   action representation
-   LinUCB
-   exploration/exploitation
-   online update
-   reward function

**Deliverable:** primary adaptive scheduler.

------------------------------------------------------------------------

## Phase 5 --- RL

Build only if time permits:

-   state
-   action
-   reward
-   agent
-   training loop
-   evaluation

------------------------------------------------------------------------

## Phase 6 --- Evaluation

Run:

``` text
Open Loop
XGBoost
Contextual Bandit
RL
```

on identical scenarios.

Generate:

-   means
-   standard deviations
-   metric tables
-   plots

------------------------------------------------------------------------

## Phase 7 --- Demo Dashboard

Create a simple interface showing:

``` text
RF spectrum
Current receiver action
Emitter ground truth (demo/debug mode only)
Hit/Miss
Adaptive priority
Next selected band
Performance metrics
```

For the final presentation, ground truth should be visually separated
from what the scheduler actually observes.

------------------------------------------------------------------------

## Phase 8 --- Presentation and Abstract

Use actual measured results.

Do NOT invent improvement percentages before experiments are complete.

------------------------------------------------------------------------

# 35. Acceptance Criteria for Phase 1

Phase 1 is considered complete only when:

### Environment

-   [ ] 20 configurable frequency bands work.
-   [ ] Time-slot simulation works.
-   [ ] Seeded reproducibility works.
-   [ ] Ground truth is generated correctly.
-   [ ] Multiple emitters can coexist.
-   [ ] Periodic emitter works.
-   [ ] Frequency-agile emitter works.
-   [ ] Intermittent/spatial scanning abstraction works.
-   [ ] Dynamic emitter works.

### Receiver

-   [ ] Frequency selection works.
-   [ ] Dwell time works.
-   [ ] Pd is applied.
-   [ ] Pfa is applied.
-   [ ] HIT/MISS/false-alarm outcomes are produced.

### Interface

-   [ ] `reset()` works.
-   [ ] `step(action)` works.
-   [ ] Scheduler receives observations rather than ground truth.
-   [ ] Action format is consistent.

### Testing

-   [ ] Unit tests exist.
-   [ ] Reproducibility test passes.
-   [ ] Receiver probability tests pass statistically.
-   [ ] Environment timeline can be visualized.

------------------------------------------------------------------------

# 36. Acceptance Criteria for the Entire Project

The full system is considered successful when:

1.  Multiple schedulers can plug into the same environment without
    modifying the environment.
2.  All schedulers can be evaluated on identical RF episodes.
3.  The adaptive scheduler responds to changes in observed RF activity.
4.  Performance can be measured using a defined metric framework.
5.  Results are reproducible.
6.  The system demonstrates measurable differences between open-loop and
    adaptive scheduling.
7.  The final presentation clearly distinguishes simulation assumptions
    from official SIH requirements.
8.  No performance claim is made without experimental evidence.

------------------------------------------------------------------------

# 37. Expected Final Experiment

The final experiment should look like:

``` text
                    RF SIMULATOR
                         │
             Generate identical episodes
                         │
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
   Open Loop         XGBoost          Bandit
   Baseline        + Optimization        ⭐
        │                │                │
        └────────────────┼────────────────┘
                         ↓
                    Evaluation
                         ↓
       ┌────────────────────────────────┐
       │ Interception Rate              │
       │ Average Intercept Time         │
       │ Pd                             │
       │ Pfa                            │
       │ Reward                         │
       │ Dwell Efficiency               │
       │ Prediction Accuracy            │
       │ Intercept-Time Error           │
       └────────────────────────────────┘
```

The key result is not simply:

> "Our ML model has high accuracy."

The result should be:

> **"Our adaptive scheduler uses limited receiver observation time more
> effectively than the open-loop strategy."**

------------------------------------------------------------------------

# 38. Internal Hackathon Presentation Story

The 10-minute presentation should follow this narrative:

### 1. Problem

Wide RF spectrum + limited receiver bandwidth.

### 2. Existing approach

Predetermined/open-loop scanning.

### 3. Limitation

The scan strategy does not sufficiently exploit observations made during
operation.

### 4. Proposed idea

Adaptive frequency-time scheduling.

### 5. Why ML?

The environment is uncertain and emitter behaviour may change.

### 6. Why Contextual Bandit?

The scheduler repeatedly chooses an action based on current context and
receives immediate feedback.

### 7. Simulation

Periodic + agile + intermittent + dynamic emitters.

### 8. Evaluation

Open-loop vs XGBoost vs Contextual Bandit.

### 9. Results

Actual experimental metrics.

### 10. Future work

RL, richer RF models, hardware/SDR integration and more realistic
propagation models.

------------------------------------------------------------------------

# 39. Technical Summary

The project can be summarized as:

``` text
INPUT:
Historical receiver observations
+
Current context
+
Available receiver actions

DECISION:
Choose frequency + dwell time

FEEDBACK:
HIT / MISS / FALSE ALARM

LEARNING:
Update scheduler

OBJECTIVE:
Maximize useful interception
while minimizing scanning time
and intercept delay.

EVALUATION:
Interception Rate
Average Intercept Time
Pd
Pfa
Sensitivity
Reward
Prediction Accuracy
Intercept-Time Error
Dwell Efficiency
```

------------------------------------------------------------------------

# 40. Final Design Decision

For the internal hackathon, the recommended primary architecture is:

> **Simulation-based ESM receiver + Contextual Bandit (LinUCB) adaptive
> frequency-time scheduler**

with:

``` text
Open Loop
    ↓
baseline

XGBoost + Optimization
    ↓
ML benchmark

Contextual Bandit
    ↓
PRIMARY PROPOSED METHOD

RL
    ↓
future/optional benchmark
```

The most important engineering decision is to keep the **RF environment
independent of every scheduling algorithm**. This allows us to change
algorithms without rewriting the simulator and gives us a fair
experimental comparison.

------------------------------------------------------------------------

## Next Implementation Step

We should now implement **Phase 1A: the RF environment foundation**.

That means creating only:

``` text
environment/
├── config.py
├── types.py
└── __init__.py
```

and defining the configuration and data contracts first.

Then we'll implement the emitter models, receiver, environment loop, and
tests one component at a time.
