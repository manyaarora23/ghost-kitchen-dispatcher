"""
KitchenFlow-v1 — Baseline Inference Script
==========================================
Runs three agents against the environment and produces reproducible scores:

  1. ThresholdPolicy   — rule-based: summon at prep > THRESHOLD
  2. ETAPolicy         — rule-based: summon when driver ETA <= food ETA
  3. OpenAIPolicy      — LLM agent: uses OpenAI API to decide each step

Usage
-----
  # Run all three agents (OpenAI agent requires OPENAI_API_KEY)
  python baseline.py

  # Rule-based only (no API key needed)
  python baseline.py --policy threshold
  python baseline.py --policy eta

  # OpenAI agent only
  python baseline.py --policy openai --model gpt-4o-mini

  # Single task, more episodes
  python baseline.py --policy openai --task easy --n 5

Environment variables
---------------------
  OPENAI_API_KEY   — required for --policy openai

Outputs
-------
  Prints scores to stdout.
  Saves results to results.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from typing import Any, Callable, Dict, List, Optional

from env import KitchenFlowEnv, Observation
from tasks import TASKS, run_episode, EpisodeResult


# ---------------------------------------------------------------------------
# Rule-based policies
# ---------------------------------------------------------------------------

class ThresholdPolicy:
    """Summon driver the first time food_prep_progress >= threshold."""

    def __init__(self, threshold: float = 0.80) -> None:
        self.threshold = threshold
        self.summoned  = False

    def reset(self) -> None:
        self.summoned = False

    def __call__(self, obs: Observation, env: KitchenFlowEnv) -> int:
        if env.state()["_time_elapsed"] == 1:
            self.reset()
        if not self.summoned and obs.food_prep_progress >= self.threshold:
            self.summoned = True
            return 1
        return 0


class ETAPolicy:
    """Summon when driver ETA approximately matches food ETA."""

    def __init__(self, lead_buffer: float = 1.0) -> None:
        self.lead_buffer = lead_buffer
        self.summoned    = False

    def reset(self) -> None:
        self.summoned = False

    def __call__(self, obs: Observation, env: KitchenFlowEnv) -> int:
        if env.state()["_time_elapsed"] == 1:
            self.reset()
        if self.summoned:
            return 0
        eta_food   = env.eta_food()
        speed      = 0.5 / obs.traffic_index
        eta_driver = obs.driver_dist / speed if speed > 0 else math.inf
        if eta_driver <= eta_food + self.lead_buffer:
            self.summoned = True
            return 1
        return 0


# ---------------------------------------------------------------------------
# OpenAI policy
# ---------------------------------------------------------------------------

class OpenAIPolicy:
    """
    LLM agent that uses the OpenAI API to decide each step.

    Reads OPENAI_API_KEY from environment variables.
    Sends the current observation as a structured prompt and parses
    the model's action (0=wait, 1=summon) from the response.
    """

    SYSTEM_PROMPT = """You are an AI dispatcher for a ghost kitchen (cloud kitchen).
Your job is to decide WHEN to summon a delivery driver for a food order.

Each minute you receive:
- food_prep_progress: how complete the food is (0.0 = just started, 1.0 = fully ready)
- driver_dist_km: how far the nearest driver is from the kitchen in km
- traffic_index: road congestion (1.0 = clear roads, 2.5 = heavy gridlock)
- food_temp_c: current food temperature in Celsius (starts at 75C, decays after food is ready)
- time_elapsed_min: minutes elapsed in this episode

Your goal is to summon the driver at exactly the right moment so that:
1. The driver arrives just as the food is ready (perfect sync = +10 reward bonus)
2. The food is still hot when handed over (each degree below 75C = -1 penalty)
3. The driver does NOT wait more than 15 minutes for food (driver cancels = -20 penalty)

Driver speed = 0.5 km/min divided by traffic_index.
Example: driver 3km away, traffic 1.5x -> ETA = 3 / (0.5/1.5) = 9 minutes.

Respond with ONLY a JSON object, nothing else:
{"action": 0}   <- wait this minute
{"action": 1}   <- summon driver NOW (you can only do this once per episode)
"""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model    = model
        self.summoned = False
        self._history: List[Dict[str, str]] = []

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set.\n"
                "Set it with:  export OPENAI_API_KEY=sk-..."
            )

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError(
                "openai package not installed. Run:  pip install openai"
            )

    def reset(self) -> None:
        self.summoned = False
        self._history = []

    def __call__(self, obs: Observation, env: KitchenFlowEnv) -> int:
        state = env.state()

        if state["_time_elapsed"] == 1:
            self.reset()

        # Once summoned, always return wait
        if self.summoned:
            return 0

        # Build user message with current observation
        user_msg = (
            f"Minute {state['_time_elapsed']}:\n"
            f"  food_prep_progress : {obs.food_prep_progress:.2%}\n"
            f"  driver_dist_km     : {obs.driver_dist:.2f} km\n"
            f"  traffic_index      : {obs.traffic_index:.2f}x\n"
            f"  food_temp_c        : {obs.food_temp:.1f} C\n"
            f"  time_elapsed_min   : {state['_time_elapsed']}\n"
            f"  driver_summoned    : {state['_driver_summoned']}\n\n"
            f"What is your action? Respond ONLY with JSON: "
            f'`{{"action": 0}}` or `{{"action": 1}}`'
        )

        self._history.append({"role": "user", "content": user_msg})

        try:
            response = self._client.chat.completions.create(
                model       = self.model,
                messages    = [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    *self._history,
                ],
                temperature = 0.0,   # deterministic
                max_tokens  = 20,
            )
            reply = response.choices[0].message.content.strip()
            self._history.append({"role": "assistant", "content": reply})

            # Parse action from JSON response
            # Strip markdown fences if model wraps response
            clean = reply.strip("`").replace("json", "").strip()
            parsed = json.loads(clean)
            action = int(parsed.get("action", 0))
            action = max(0, min(1, action))   # clamp to valid range

        except Exception as e:
            # On any API or parse error, default to wait
            print(f"  [OpenAI] Warning: {e} — defaulting to wait")
            action = 0

        if action == 1:
            self.summoned = True

        return action


# ---------------------------------------------------------------------------
# Wrapped policy helpers (signature: policy(obs, env) -> int)
# ---------------------------------------------------------------------------

def make_threshold_policy(threshold: float = 0.80) -> Callable:
    p = ThresholdPolicy(threshold)
    return p


def make_eta_policy(lead_buffer: float = 1.0) -> Callable:
    p = ETAPolicy(lead_buffer)
    return p


def make_openai_policy(model: str = "gpt-4o-mini") -> Callable:
    p = OpenAIPolicy(model=model)
    return p


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark(
    policy_fn:   Callable,
    policy_name: str,
    task:        Optional[str] = None,
    n_episodes:  int = 10,
    verbose:     bool = True,
) -> Dict[str, Any]:
    """Run a policy on one or all tasks and return a results dict."""

    tasks_to_run = [task] if task else ["easy", "medium", "hard"]
    results: Dict[str, Any] = {"policy": policy_name, "tasks": {}}

    for t in tasks_to_run:
        TaskClass = TASKS[t]
        env       = KitchenFlowEnv(task=t)
        episodes: List[EpisodeResult] = []

        if verbose:
            print(f"\n  Running {n_episodes} episodes on task='{t}' with policy='{policy_name}' ...")

        for seed in TaskClass.seeds[:n_episodes]:
            ep = run_episode(env, policy_fn, seed)
            episodes.append(ep)
            if verbose:
                sym = "✓" if ep.termination == "delivered" else "✗"
                print(f"    seed={seed:>4}  {sym}  reward={ep.total_reward:>8.2f}  "
                      f"term={ep.termination:<18}  temp={ep.food_temp:.1f}C")

        score     = TaskClass.grader(episodes)
        rewards   = [e.total_reward for e in episodes]
        delivered = sum(1 for e in episodes if e.termination == "delivered")
        cancelled = sum(1 for e in episodes if e.termination == "driver_cancelled")
        timed_out = sum(1 for e in episodes if e.termination == "timeout")
        synced    = sum(1 for e in episodes if e.sync_bonus)
        avg_temp  = (
            statistics.mean(e.food_temp for e in episodes if e.termination == "delivered")
            if delivered else 0.0
        )

        task_result = {
            "score":         score,
            "n_episodes":    len(episodes),
            "mean_reward":   round(statistics.mean(rewards), 3),
            "std_reward":    round(statistics.stdev(rewards) if len(rewards) > 1 else 0.0, 3),
            "delivery_rate": round(delivered / len(episodes), 3),
            "cancel_rate":   round(cancelled / len(episodes), 3),
            "timeout_rate":  round(timed_out / len(episodes), 3),
            "sync_rate":     round(synced    / len(episodes), 3),
            "avg_food_temp": round(avg_temp, 2),
        }
        results["tasks"][t] = task_result

        if verbose:
            print(f"\n  {'─'*50}")
            print(f"  Task: {t:<8} | Policy: {policy_name}")
            print(f"  {'─'*50}")
            print(f"  Score          : {score:.4f}")
            print(f"  Mean reward    : {task_result['mean_reward']:.2f}  "
                  f"(+/-{task_result['std_reward']:.2f})")
            print(f"  Delivered      : {delivered}/{len(episodes)}  "
                  f"({task_result['delivery_rate']:.0%})")
            print(f"  Synced         : {synced}/{len(episodes)}  "
                  f"({task_result['sync_rate']:.0%})")
            print(f"  Avg food temp  : {avg_temp:.1f}C")
            print(f"  Cancellations  : {cancelled}  |  Timeouts: {timed_out}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="KitchenFlow-v1 baseline benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python baseline.py                          # all three policies, all tasks
  python baseline.py --policy threshold       # rule-based threshold only
  python baseline.py --policy openai          # OpenAI LLM agent (needs OPENAI_API_KEY)
  python baseline.py --policy openai --model gpt-4o --task easy --n 3
        """,
    )
    parser.add_argument(
        "--policy",
        choices=["threshold", "eta", "openai", "all"],
        default="all",
        help="Which policy to benchmark (default: all)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.80,
        help="Prep threshold for threshold policy (default: 0.80)",
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="OpenAI model for openai policy (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Which task(s) to evaluate (default: all)",
    )
    parser.add_argument(
        "--n", type=int, default=10,
        help="Episodes per task (default: 10; use fewer for openai to save cost)",
    )
    parser.add_argument(
        "--out", default="results.json",
        help="Output file for results (default: results.json)",
    )
    args = parser.parse_args()

    print(f"\nKitchenFlow-v1 — Baseline Benchmark")
    print(f"Policy: {args.policy}  |  Task: {args.task}  |  N: {args.n}")
    print("=" * 52)

    task_arg = None if args.task == "all" else args.task
    all_results: Dict[str, Any] = {}
    t0 = time.time()

    policies_to_run = (
        ["threshold", "eta", "openai"] if args.policy == "all" else [args.policy]
    )

    for policy_name in policies_to_run:
        print(f"\n{'='*52}")
        print(f"  Policy: {policy_name.upper()}")
        print(f"{'='*52}")

        if policy_name == "threshold":
            policy_fn   = make_threshold_policy(args.threshold)
            label       = f"threshold_{args.threshold}"

        elif policy_name == "eta":
            policy_fn   = make_eta_policy()
            label       = "eta_policy"

        elif policy_name == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print(
                    "\n  [SKIP] OPENAI_API_KEY not set — skipping OpenAI policy.\n"
                    "  Set it with:  export OPENAI_API_KEY=sk-...\n"
                )
                continue
            try:
                policy_fn = make_openai_policy(model=args.model)
                label     = f"openai_{args.model}"
            except Exception as e:
                print(f"\n  [SKIP] Could not load OpenAI policy: {e}\n")
                continue

        else:
            continue

        result = benchmark(policy_fn, label, task=task_arg, n_episodes=args.n)
        all_results[label] = result

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*52}")
    print("  SUMMARY")
    print(f"{'='*52}")

    for policy_label, result in all_results.items():
        task_scores = [v["score"] for v in result["tasks"].values()]
        if task_scores:
            overall = round(statistics.mean(task_scores), 4)
            print(f"  {policy_label:<30} overall={overall:.4f}")
            for task_name, t_res in result["tasks"].items():
                print(f"    {task_name:<10} score={t_res['score']:.4f}  "
                      f"delivery={t_res['delivery_rate']:.0%}  "
                      f"sync={t_res['sync_rate']:.0%}")

    print(f"\n  Wall time: {elapsed:.1f}s")
    print(f"{'='*52}\n")

    # ── Save results ──────────────────────────────────────────────────────
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()
