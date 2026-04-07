"""
KitchenFlow-v1 — Ghost Kitchen Dispatcher Environment (CSV Integrated)
======================================================================
OpenEnv-compliant: typed Pydantic models for Observation, Action, Reward.
Pulls initial episode conditions from kitchenflow_dataset.csv.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DRIVER_SPEED_BASE     = 0.5    # km per minute at traffic_index 1.0
SYNC_BONUS            = 10.0   # driver + food arrive within 1 min of each other
TEMP_PENALTY_PER_DEG  = 1.0    # per °C below perfect serving temp at delivery
DRIVER_CANCEL_PENALTY = 20.0   # driver waited > 15 min for food
TIMEOUT_PENALTY       = 15.0   # episode expired at 60 min
WAIT_THRESHOLD        = 15     # minutes before driver cancels
PERFECT_TEMP          = 75.0   # °C — ideal serving temperature


# ---------------------------------------------------------------------------
# Pydantic Models  (satisfies OpenEnv typed spec requirement)
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """Typed observation returned by reset() and step()."""
    food_prep_progress: float = Field(..., ge=0.0, le=1.0,
        description="Fraction of food preparation complete (0=raw, 1=ready)")
    driver_dist: float = Field(..., ge=0.0, le=20.0,
        description="Distance of nearest driver from kitchen in km")
    traffic_index: float = Field(..., ge=1.0, le=2.5,
        description="Road congestion multiplier (1.0=free flow, 2.5=gridlock)")
    food_temp: float = Field(..., ge=0.0, le=100.0,
        description="Current food temperature in C (decays once food is ready)")


class Action(BaseModel):
    """Typed action consumed by step()."""
    action: int = Field(..., ge=0, le=1,
        description="0=wait this minute, 1=summon driver (one-time trigger)")


class Reward(BaseModel):
    """Typed reward breakdown returned in step() info."""
    total:          float = Field(...,        description="Net reward for this step")
    sync_bonus:     float = Field(default=0.0, description="+10 if driver and food sync within 1 min")
    temp_penalty:   float = Field(default=0.0, description="-1 per C below 75C at delivery")
    cancel_penalty: float = Field(default=0.0, description="-20 if driver waited > 15 min")
    timeout_penalty:float = Field(default=0.0, description="-15 if episode hits 60 min limit")
    shaping:        float = Field(default=0.0, description="Per-step cooling penalty while food waits")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class KitchenFlowEnv:
    """
    KitchenFlow-v1: Ghost Kitchen Dispatcher

    OpenEnv API:
        reset(seed?)  -> Observation
        step(action)  -> (Observation, float, bool, dict)
        state()       -> full internal state dict

    Observation space (4 floats):
        food_prep_progress  [0.0, 1.0]
        driver_dist         [0.0, 20.0] km
        traffic_index       [1.0, 2.5]
        food_temp           [0.0, 100.0] C

    Action space (discrete, n=2):
        0 -> wait
        1 -> summon driver (one-time trigger)
    """

    metadata = {
        "name":    "KitchenFlow-v1",
        "version": "1.0.0",
        "tasks":   ["easy", "medium", "hard"],
    }

    def __init__(self, task: str = "easy") -> None:
        if task not in ("easy", "medium", "hard"):
            raise ValueError(f"Unknown task '{task}'. Choose from: easy, medium, hard")
        self.task = task
        self.dataset_path = "kitchenflow_dataset.csv"

        if os.path.exists(self.dataset_path):
            df = pd.read_csv(self.dataset_path)
            self.task_data = df[df["task"] == self.task]
            if self.task_data.empty:
                print(f"Warning: No rows for task='{task}' in CSV. Using full dataset.")
                self.task_data = df
        else:
            raise FileNotFoundError(
                f"Could not find {self.dataset_path}. "
                "Please ensure it is in the same directory as env.py."
            )

        self._s: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Observation:
        """Reset the episode from a CSV scenario. Returns typed Observation."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        scenario = self.task_data.sample(n=1).iloc[0]

        self._s = {
            # Visible to agent (mirrors Observation fields)
            "food_prep_progress":  0.0,
            "driver_dist":         float(scenario["driver_dist_km"]),
            "traffic_index":       float(scenario["traffic_index"]),
            "food_temp":           PERFECT_TEMP,

            # Hidden simulation variables
            "_prep_increment":     1.0 / float(scenario["prep_time_min"]),
            "_food_ready":         False,
            "_driver_summoned":    False,
            "_driver_arrived":     False,
            "_driver_wait_time":   0,
            "_food_wait_time":     0,
            "_time_elapsed":       0,
            "_episode_reward":     0.0,
            "_done":               False,
            "_driver_summoned_at": None,
            "_driver_arrived_at":  None,
            "_food_ready_at":      None,
        }

        return self._make_obs()

    def step(self, action: int) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Advance simulation by 1 minute.

        Parameters
        ----------
        action : int  0=wait, 1=summon driver

        Returns
        -------
        observation : Observation (typed Pydantic model)
        reward      : float
        done        : bool
        info        : dict  (includes typed Reward breakdown)
        """
        if self._s.get("_done"):
            raise RuntimeError("Episode is done — call reset() first.")

        s = self._s
        reward_breakdown = Reward(total=0.0)
        info: Dict[str, Any] = {}

        # 1. Action
        if action == 1 and not s["_driver_summoned"]:
            s["_driver_summoned"]    = True
            s["_driver_summoned_at"] = s["_time_elapsed"]
            info["driver_summoned_at"] = s["_time_elapsed"]

        # 2. Time tick
        s["_time_elapsed"] += 1

        # 3. Food prep
        if not s["_food_ready"]:
            s["food_prep_progress"] = min(1.0, s["food_prep_progress"] + s["_prep_increment"])
            if s["food_prep_progress"] >= 1.0:
                s["_food_ready"]    = True
                s["_food_ready_at"] = s["_time_elapsed"]
                info["food_ready_at"] = s["_time_elapsed"]

        # 4. Driver movement
        if s["_driver_summoned"] and not s["_driver_arrived"]:
            speed = DRIVER_SPEED_BASE / s["traffic_index"]
            s["driver_dist"] = max(0.0, s["driver_dist"] - speed)
            if s["driver_dist"] <= 0.0:
                s["_driver_arrived"]    = True
                s["_driver_arrived_at"] = s["_time_elapsed"]
                info["driver_arrived_at"] = s["_time_elapsed"]

        # 5. Wait tracking + food cooling
        if s["_driver_arrived"] and not s["_food_ready"]:
            s["_driver_wait_time"] += 1

        if s["_food_ready"] and not s["_driver_arrived"]:
            s["_food_wait_time"] += 1
            s["food_temp"] = max(0.0, s["food_temp"] - 0.5)

        # 6. Per-step shaping signal
        if s["_food_ready"] and not s["_driver_arrived"]:
            temp_drop = max(0.0, PERFECT_TEMP - s["food_temp"])
            reward_breakdown.shaping = round(-0.05 * temp_drop, 4)

        # 7. Terminal conditions
        done = False

        # Driver cancels
        if s["_driver_arrived"] and s["_driver_wait_time"] > WAIT_THRESHOLD:
            reward_breakdown.cancel_penalty = -DRIVER_CANCEL_PENALTY
            done = True
            info["termination"] = "driver_cancelled"

        # Successful delivery
        elif s["_driver_arrived"] and s["_food_ready"]:
            temp_drop = max(0.0, PERFECT_TEMP - s["food_temp"])
            temp_pen  = TEMP_PENALTY_PER_DEG * temp_drop
            reward_breakdown.temp_penalty = round(-temp_pen, 4)

            synced = s["_driver_wait_time"] <= 1 and s["_food_wait_time"] <= 1
            if synced:
                reward_breakdown.sync_bonus = SYNC_BONUS
                info["sync_bonus"] = True

            done = True
            info.update({
                "termination":     "delivered",
                "food_temp":       round(s["food_temp"], 2),
                "temp_penalty":    round(temp_pen, 2),
                "driver_wait_min": s["_driver_wait_time"],
                "food_wait_min":   s["_food_wait_time"],
                "time_elapsed":    s["_time_elapsed"],
            })

        # Timeout
        elif s["_time_elapsed"] >= 60:
            reward_breakdown.timeout_penalty = -TIMEOUT_PENALTY
            done = True
            info["termination"] = "timeout"

        if not done:
            info["termination"] = "running"

        # 8. Total reward
        total_reward = (
            reward_breakdown.sync_bonus
            + reward_breakdown.temp_penalty
            + reward_breakdown.cancel_penalty
            + reward_breakdown.timeout_penalty
            + reward_breakdown.shaping
        )
        reward_breakdown.total = round(total_reward, 4)

        s["_done"]           = done
        s["_episode_reward"] += total_reward
        info["reward_breakdown"] = reward_breakdown.model_dump()

        return self._make_obs(), total_reward, done, info

    def state(self) -> Dict[str, Any]:
        """Full internal state including hidden variables."""
        return {k: v for k, v in self._s.items()}

    # ------------------------------------------------------------------
    # OpenEnv Spaces  (restored static methods)
    # ------------------------------------------------------------------

    @staticmethod
    def observation_space() -> Dict[str, Any]:
        return {
            "type": "dict",
            "fields": {
                "food_prep_progress": {"type": "float", "low": 0.0,  "high": 1.0},
                "driver_dist":        {"type": "float", "low": 0.0,  "high": 20.0},
                "traffic_index":      {"type": "float", "low": 1.0,  "high": 2.5},
                "food_temp":          {"type": "float", "low": 0.0,  "high": 100.0},
            },
        }

    @staticmethod
    def action_space() -> Dict[str, Any]:
        return {
            "type":     "discrete",
            "n":        2,
            "meanings": ["wait", "summon_driver"],
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def eta_food(self) -> float:
        """Estimated minutes until food is ready."""
        s = self._s
        if s["_food_ready"]:
            return 0.0
        remaining = 1.0 - s["food_prep_progress"]
        return remaining / s["_prep_increment"] if s["_prep_increment"] > 0 else float("inf")

    def eta_driver(self) -> float:
        """Estimated minutes until driver arrives (if summoned)."""
        s = self._s
        if s["_driver_arrived"]:
            return 0.0
        if not s["_driver_summoned"]:
            return float("inf")
        speed = DRIVER_SPEED_BASE / s["traffic_index"]
        return s["driver_dist"] / speed if speed > 0 else float("inf")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _make_obs(self) -> Observation:
        s = self._s
        return Observation(
            food_prep_progress = round(float(s["food_prep_progress"]), 4),
            driver_dist        = round(float(s["driver_dist"]),        4),
            traffic_index      = round(float(s["traffic_index"]),      4),
            food_temp          = round(float(s["food_temp"]),          4),
        )

    def __repr__(self) -> str:
        s = self._s
        return (
            f"KitchenFlowEnv(task={self.task!r}, "
            f"t={s.get('_time_elapsed', 0)}min, "
            f"prep={s.get('food_prep_progress', 0):.0%}, "
            f"dist={s.get('driver_dist', 0):.1f}km, "
            f"summoned={s.get('_driver_summoned', False)})"
        )
