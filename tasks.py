"""
KitchenFlow-v1 — Task Definitions & Graders
============================================
Three tasks (easy → medium → hard) each with:
  • fixed seeds for reproducibility
  • a grader(episodes) → float [0.0, 1.0]
  • task-level metadata for openenv.yaml

Grading philosophy
------------------
Raw episodic reward is noisy — graders map it to a normalised score
using task-specific baselines and ceiling values so that:
  • 0.0 = random / always-wait agent
  • 0.5 ≈ rule-based baseline ("summon at prep > 80%")
  • 1.0 = perfect sync on every episode (theoretical ceiling)

LLM grader prompt is also exported for Phase 2 agentic evaluation.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from env import KitchenFlowEnv


# ---------------------------------------------------------------------------
# Episode result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    task:            str
    seed:            int
    total_reward:    float
    termination:     str          # delivered | driver_cancelled | timeout
    food_temp:       float = 0.0
    driver_wait_min: int   = 0
    food_wait_min:   int   = 0
    time_elapsed:    int   = 0
    sync_bonus:      bool  = False
    temp_penalty:    float = 0.0
    summoned_at:     Optional[int] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_episode(env: KitchenFlowEnv, policy: Callable, seed: int) -> EpisodeResult:
    """Run one episode with a callable policy(obs) → action."""
    obs    = env.reset(seed=seed)
    done   = False
    cumr   = 0.0
    info: Dict[str, Any] = {}

    while not done:
        action       = policy(obs, env)
        obs, r, done, info = env.step(action)
        cumr        += r

    return EpisodeResult(
        task         = env.task,
        seed         = seed,
        total_reward = round(cumr, 4),
        termination  = info.get("termination", "unknown"),
        food_temp    = info.get("food_temp", 0.0),
        driver_wait_min = info.get("driver_wait_min", 0),
        food_wait_min   = info.get("food_wait_min", 0),
        time_elapsed    = info.get("time_elapsed", 0),
        sync_bonus      = info.get("sync_bonus", False),
        temp_penalty    = info.get("temp_penalty", 0.0),
        summoned_at     = info.get("driver_summoned_at"),
    )


def _normalise(raw: float, floor: float, ceiling: float) -> float:
    """Clamp-and-scale raw reward to [0, 1]."""
    if ceiling == floor:
        return 0.5
    return max(0.0, min(1.0, (raw - floor) / (ceiling - floor)))


# ---------------------------------------------------------------------------
# Task 1 — Easy
# ---------------------------------------------------------------------------

TASK1_SEEDS   = list(range(42, 142))          # 100 seeds
TASK1_FLOOR = 5.0   # timeout or cancel
TASK1_CEILING = 40.0  # perfect sync

class Task1:
    """
    Easy: Calm traffic (σ=0.10), tight prep jitter (σ=1 min), nearby driver.
    
    The 80%-threshold baseline is competitive here.
    A good agent scores > 0.65.
    """
    name        = "easy_dispatch"
    description = "Dispatch a driver in low-traffic conditions with predictable prep times."
    difficulty  = "easy"
    seeds       = TASK1_SEEDS
    n_episodes  = 20          # subset used for grading (speed)

    @staticmethod
    def grader(results: List[EpisodeResult]) -> float:
        """
        Score = normalised mean reward across episodes.
        Bonus weight for sync deliveries.
        """
        if not results:
            return 0.0
        scores = []
        for r in results:
            base  = _normalise(r.total_reward, TASK1_FLOOR, TASK1_CEILING)
            bonus = 0.05 if r.sync_bonus else 0.0
            scores.append(min(1.0, base + bonus))
        return round(statistics.mean(scores), 4)

    @staticmethod
    def run(policy: Callable, n: int = 20) -> float:
        env     = KitchenFlowEnv(task="easy")
        results = [run_episode(env, policy, s) for s in TASK1_SEEDS[:n]]
        return Task1.grader(results)


# ---------------------------------------------------------------------------
# Task 2 — Medium
# ---------------------------------------------------------------------------

TASK2_SEEDS   = list(range(200, 300))
TASK2_FLOOR   = 5.0
TASK2_CEILING = 500.0

class Task2:
    """
    Medium: Variable traffic (σ=0.30), higher prep jitter (σ=3 min),
    driver can be up to 5–7 km away.

    The 80%-threshold baseline starts failing here because traffic
    variance makes ETA estimation unreliable.
    A good agent scores > 0.50.
    """
    name        = "medium_dispatch"
    description = "Dispatch under variable traffic with uncertain prep and driver distance."
    difficulty  = "medium"
    seeds       = TASK2_SEEDS
    n_episodes  = 20

    @staticmethod
    def grader(results: List[EpisodeResult]) -> float:
        print(f"DEBUG: Running Grader for MEDIUM with Ceiling: {TASK2_CEILING}")
        if not results:
            return 0.0
        scores = []
        for r in results:
            base = _normalise(r.total_reward, TASK2_FLOOR, TASK2_CEILING)
            # Penalise cancellations more heavily in grader
            if r.termination == "driver_cancelled":
                base *= 0.5
            bonus = 0.07 if r.sync_bonus else 0.0
            scores.append(min(1.0, base + bonus))

        # Variance penalty: unstable policies score lower
        if len(scores) > 1:
            std_pen = min(0.10, statistics.stdev(scores) * 0.3)
            return round(max(0.0, statistics.mean(scores) - std_pen), 4)
        return round(statistics.mean(scores), 4)

    @staticmethod
    def run(policy: Callable, n: int = 20) -> float:
        env     = KitchenFlowEnv(task="medium")
        results = [run_episode(env, policy, s) for s in TASK2_SEEDS[:n]]
        return Task2.grader(results)


# ---------------------------------------------------------------------------
# Task 3 — Hard
# ---------------------------------------------------------------------------

TASK3_SEEDS   = list(range(500, 600))
TASK3_FLOOR   = 5.0
TASK3_CEILING = 200.0

class Task3:
    """
    Hard: Rush-hour traffic (mean=1.8, σ=0.5), traffic spikes (5%/step),
    large prep jitter (σ=5 min), driver can be far (mean=5 km ± 2.5),
    food cools at 2°C/min.

    The baseline rule scores < 0.20 here.
    A strong agent scores > 0.45, an excellent one > 0.60.
    """
    name        = "hard_dispatch"
    description = (
        "Dispatch in rush-hour with random traffic spikes, uncertain prep times, "
        "distant drivers, and rapid food cooling."
    )
    difficulty  = "hard"
    seeds       = TASK3_SEEDS
    n_episodes  = 25   # more episodes for reliable signal

    @staticmethod
    def grader(results: List[EpisodeResult]) -> float:
        if not results:
            return 0.0

        scores       = []
        delivered    = 0
        temp_quality = []

        for r in results:
            base = _normalise(r.total_reward, TASK3_FLOOR, TASK3_CEILING)

            if r.termination == "delivered":
                delivered += 1
                # Extra weight on food temperature
                temp_score = max(0.0, (r.food_temp - 50.0) / (75.0 - 50.0))
                temp_quality.append(temp_score)
                bonus = 0.10 if r.sync_bonus else 0.0
            else:
                bonus = 0.0

            if r.termination == "driver_cancelled":
                base *= 0.4
            elif r.termination == "timeout":
                base *= 0.6

            scores.append(min(1.0, base + bonus))

        delivery_rate = delivered / len(results)
        base_score    = statistics.mean(scores)

        # Blend base score with delivery rate (50/50)
        composite = 0.5 * base_score + 0.5 * delivery_rate

        # Temperature quality modifier (only if deliveries happened)
        if temp_quality:
            tq_bonus = statistics.mean(temp_quality) * 0.08
            composite = min(1.0, composite + tq_bonus)

        return round(composite, 4)

    @staticmethod
    def run(policy: Callable, n: int = 25) -> float:
        env     = KitchenFlowEnv(task="hard")
        results = [run_episode(env, policy, s) for s in TASK3_SEEDS[:n]]
        return Task3.grader(results)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASKS: Dict[str, Any] = {
    "easy":   Task1,
    "medium": Task2,
    "hard":   Task3,
}


def grade_all(policy: Callable, verbose: bool = True) -> Dict[str, float]:
    """Run all three tasks and return a scores dict."""
    scores: Dict[str, float] = {}
    for name, TaskClass in TASKS.items():
        score = TaskClass.run(policy)
        scores[name] = score
        if verbose:
            print(f"[{name:>6}]  score = {score:.4f}")
    if verbose:
        overall = statistics.mean(scores.values())
        print(f"\n  Overall: {overall:.4f}")
    return scores


# ---------------------------------------------------------------------------
# LLM Grader Prompt (for Phase 2 agentic evaluation)
# ---------------------------------------------------------------------------

LLM_GRADER_SYSTEM = """
You are evaluating an RL agent on the KitchenFlow-v1 ghost kitchen dispatch task.

The agent must decide WHEN to summon a delivery driver so that:
  1. The driver arrives at the same time the food is ready (sync = good)
  2. The food is still hot at handoff (temp > 65°C = good)
  3. The driver does not wait more than 15 minutes (cancel = very bad)

You will be given a JSON list of episode results. For each episode analyse:
  - termination: 'delivered' is good, 'driver_cancelled'/'timeout' are bad
  - sync_bonus: True means near-perfect timing
  - food_temp: higher is better (perfect = 75°C, acceptable ≥ 65°C, poor < 55°C)
  - driver_wait_min: minutes driver waited for food (0–1 = great, > 5 = poor)
  - food_wait_min: minutes food waited for driver — food cooling (0–1 = great)
  - summoned_at: did the agent call the driver at a reasonable time?

Respond ONLY with a JSON object:
{
  "score": <float 0.0–1.0>,
  "reasoning": "<2–3 sentences>",
  "strengths": ["<brief point>", ...],
  "weaknesses": ["<brief point>", ...]
}

Scoring guide:
  0.0–0.2  Most deliveries fail or food arrives cold
  0.2–0.4  Some deliveries succeed but timing is poor
  0.4–0.6  Consistent deliveries, acceptable food quality
  0.6–0.8  Good timing, food mostly warm, few cancellations
  0.8–1.0  Near-perfect sync, hot food, no cancellations
"""
