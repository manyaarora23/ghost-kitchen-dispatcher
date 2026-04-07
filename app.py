"""
KitchenFlow-v1 — HTTP API Server
=================================
FastAPI app exposing the OpenEnv step()/reset()/state() API over HTTP.
Designed for Hugging Face Spaces deployment.

Endpoints
---------
POST /reset            → { "observation": {...} }
POST /step             → { "observation": {...}, "reward": float, "done": bool, "info": {...} }
GET  /state            → full internal state dict
GET  /observation_space
GET  /action_space
GET  /tasks
POST /grade            → run baseline policy on all tasks, return scores
GET  /health
GET  /                 → Swagger UI redirect
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Optional

import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from env import KitchenFlowEnv
from tasks import TASKS, grade_all
from baseline import make_threshold_policy, make_eta_policy


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "KitchenFlow-v1",
    description = (
        "Ghost Kitchen Dispatcher — a real-world RL environment where an agent "
        "learns to synchronise driver arrival with food readiness."
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ---------------------------------------------------------------------------
# Session management (thread-safe, one env per session)
# ---------------------------------------------------------------------------

_sessions: Dict[str, KitchenFlowEnv] = {}
_lock = threading.Lock()
_MAX_SESSIONS = 64


def _get_or_create(session_id: str, task: str = "easy") -> KitchenFlowEnv:
    with _lock:
        if session_id not in _sessions:
            if len(_sessions) >= _MAX_SESSIONS:
                # Evict oldest session
                oldest = next(iter(_sessions))
                del _sessions[oldest]
            _sessions[session_id] = KitchenFlowEnv(task=task)
        return _sessions[session_id]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    session_id: str  = Field(default="default", description="Session identifier")
    task:       str  = Field(default="easy",    description="Task difficulty: easy | medium | hard")
    seed:       Optional[int] = Field(default=None, description="Random seed for reproducibility")


class StepRequest(BaseModel):
    session_id: str = Field(default="default")
    action:     int = Field(..., ge=0, le=1, description="0=wait, 1=summon_driver")


class GradeRequest(BaseModel):
    policy:     str   = Field(default="threshold", description="threshold | eta")
    threshold:  float = Field(default=0.80,         description="Threshold for threshold policy")
    n_episodes: int   = Field(default=10,            ge=1, le=50)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "environment": "KitchenFlow-v1", "version": "1.0.0"}


# ── Reset ────────────────────────────────────────────────────────────────────

@app.post("/reset")
def reset(req: ResetRequest):
    """Reset the environment and return the initial observation."""
    if req.task not in ("easy", "medium", "hard"):
        raise HTTPException(400, f"Unknown task '{req.task}'. Use: easy | medium | hard")

    with _lock:
        _sessions[req.session_id] = KitchenFlowEnv(task=req.task)

    env = _sessions[req.session_id]
    obs = env.reset(seed=req.seed)
    return {
        "observation": obs,
        "session_id":  req.session_id,
        "task":        req.task,
    }


# ── Step ─────────────────────────────────────────────────────────────────────

@app.post("/step")
def step(req: StepRequest):
    """Advance the simulation by one minute."""
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(404, f"Session '{req.session_id}' not found. Call /reset first.")

    try:
        obs, reward, done, info = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))

    return {
        "observation": obs,
        "reward":      reward,
        "done":        done,
        "info":        info,
        "session_id":  req.session_id,
    }


# ── State ─────────────────────────────────────────────────────────────────────

@app.get("/state")
def state(session_id: str = "default"):
    """Full internal state, including hidden variables."""
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, f"Session '{session_id}' not found. Call /reset first.")
    return {"state": env.state(), "session_id": session_id}


# ── Spaces ────────────────────────────────────────────────────────────────────

@app.get("/observation_space")
def observation_space():
    return KitchenFlowEnv.observation_space()


@app.get("/action_space")
def action_space():
    return KitchenFlowEnv.action_space()


# ── Tasks ─────────────────────────────────────────────────────────────────────

@app.get("/tasks")
def list_tasks():
    return {
        name: {
            "name":           cls.name,
            "description":    cls.description,
            "difficulty":     cls.difficulty,
            "baseline_score": {"easy": 0.55, "medium": 0.35, "hard": 0.18}[name],
        }
        for name, cls in TASKS.items()
    }


# ── Grade ─────────────────────────────────────────────────────────────────────

@app.post("/grade")
def grade(req: GradeRequest):
    """Run a built-in baseline policy on all tasks and return graded scores."""
    if req.policy == "threshold":
        policy_fn   = make_threshold_policy(req.threshold)
        policy_name = f"threshold_{req.threshold}"
    elif req.policy == "eta":
        policy_fn   = make_eta_policy()
        policy_name = "eta_policy"
    else:
        raise HTTPException(400, f"Unknown policy '{req.policy}'. Use: threshold | eta")

    scores: Dict[str, float] = {}
    for task_name, TaskClass in TASKS.items():
        env     = KitchenFlowEnv(task=task_name)
        results = []
        from tasks import run_episode
        for seed in TaskClass.seeds[:req.n_episodes]:
            results.append(run_episode(env, policy_fn, seed))
        scores[task_name] = TaskClass.grader(results)

    import statistics
    return {
        "policy":      policy_name,
        "scores":      scores,
        "overall":     round(statistics.mean(scores.values()), 4),
        "n_episodes":  req.n_episodes,
    }


# ── Demo: single episode trace ────────────────────────────────────────────────

@app.get("/demo")
def demo(task: str = "easy", seed: int = 42, policy: str = "threshold"):
    """
    Run one full episode with the threshold policy and return a step-by-step trace.
    Useful for visualisation and understanding the environment.
    """
    if task not in ("easy", "medium", "hard"):
        raise HTTPException(400, "task must be easy | medium | hard")

    env  = KitchenFlowEnv(task=task)
    obs  = env.reset(seed=seed)
    p    = make_threshold_policy() if policy == "threshold" else make_eta_policy()

    trace  = []
    done   = False
    cumr   = 0.0
    step_n = 0

    while not done:
        action              = p(obs, env)
        new_obs, r, done, info = env.step(action)
        cumr  += r
        step_n += 1
        trace.append({
            "step":   step_n,
            "action": "summon" if action == 1 else "wait",
            "obs":    new_obs,
            "reward": round(r, 3),
            "info":   info,
        })
        obs = new_obs

    return {
        "task":          task,
        "seed":          seed,
        "policy":        policy,
        "total_reward":  round(cumr, 3),
        "total_steps":   step_n,
        "termination":   trace[-1]["info"].get("termination", "unknown"),
        "trace":         trace,
    }


# ── Dataset ───────────────────────────────────────────────────────────────────

@app.get("/dataset")
def get_dataset(task: str = None, limit: int = 50):
    """
    Browse the KitchenFlow dataset (kitchenflow_dataset.csv).
    Filter by task=easy | medium | hard. Default returns 50 rows.
    """
    try:
        df = pd.read_csv("kitchenflow_dataset.csv")
    except FileNotFoundError:
        raise HTTPException(404, "kitchenflow_dataset.csv not found. Make sure it is in the same folder as app.py")

    if task:
        if task not in ("easy", "medium", "hard"):
            raise HTTPException(400, "task must be easy | medium | hard")
        df = df[df["task"] == task]

    return {
        "total_rows":  len(df),
        "showing":     min(limit, len(df)),
        "columns":     list(df.columns),
        "rows":        df.head(limit).to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Gradio UI (optional — only if gradio is installed)
# ---------------------------------------------------------------------------

def _make_gradio_demo():
    try:
        import gradio as gr
    except ImportError:
        return None

    def run_demo(task, seed, threshold):
        env  = KitchenFlowEnv(task=task)
        obs  = env.reset(seed=int(seed))
        p    = make_threshold_policy(float(threshold))
        done = False
        log  = []
        cumr = 0.0
        while not done:
            action          = p(obs, env)
            obs, r, done, info = env.step(action)
            cumr += r
            log.append(
                f"t={env.state()['_time_elapsed']:>2}m  "
                f"prep={obs['food_prep_progress']:.0%}  "
                f"dist={obs['driver_dist']:.1f}km  "
                f"traffic={obs['traffic_index']:.2f}  "
                f"temp={obs['food_temp']:.1f}°C  "
                f"action={'SUMMON' if action == 1 else 'wait':6}  "
                f"r={r:+.1f}"
            )
        term = info.get("termination", "?")
        summary = (
            f"\n{'─'*60}\n"
            f"Result: {term.upper()} | Total reward: {cumr:.2f}\n"
            f"{'─'*60}"
        )
        return "\n".join(log) + summary

    demo = gr.Interface(
        fn          = run_demo,
        inputs      = [
            gr.Dropdown(["easy", "medium", "hard"], label="Task", value="easy"),
            gr.Slider(0, 999, step=1, value=42, label="Seed"),
            gr.Slider(0.5, 1.0, step=0.05, value=0.80, label="Summon threshold"),
        ],
        outputs     = gr.Textbox(label="Episode trace", lines=35),
        title       = "KitchenFlow-v1 — Ghost Kitchen Dispatcher",
        description = (
            "An RL environment for cloud kitchen dispatch optimisation. "
            "The agent decides when to summon a delivery driver so food "
            "arrives hot and the driver doesn't idle."
        ),
    )
    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
