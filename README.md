# 🍔 KitchenFlow-v1 — Ghost Kitchen Dispatcher

An OpenEnv-compliant reinforcement learning environment that simulates the
real-world logistics problem of synchronising food readiness with driver arrival
in a cloud kitchen (ghost kitchen) setting.

> **The problem everyone understands:** Cold fries arrive because the driver waited
> 20 minutes, or the driver cancelled because the burger wasn't done. An RL agent
> must learn to predict the future — timing a single *Summon Driver* action with
> enough precision to hand off hot food the moment it's ready.

---

## Environment Description

The agent is the "brain" of a delivery hub. Each episode simulates one order.
The agent observes four real-time signals and decides each minute whether to
wait or call the driver. Food temperature decays the moment food sits ready
without a driver, and a driver who waits more than 15 minutes cancels.

### Observation Space (4 floats)

| Field | Range | Description |
|---|---|---|
| `food_prep_progress` | [0.0, 1.0] | Fraction of food preparation complete |
| `driver_dist` | [0.0, 20.0] km | Distance of nearest driver from kitchen |
| `traffic_index` | [1.0, 2.5] | Road congestion multiplier (1.0 = free flow) |
| `food_temp` | [0.0, 100.0] °C | Current food temperature (decays after ready) |

### Action Space (discrete, n=2)

| Action | Meaning |
|---|---|
| `0` | Wait (do nothing this 1-minute tick) |
| `1` | Summon driver — **one-time trigger**, ignored after first use |

### Reward Function (multi-objective)

| Component | Value | Condition |
|---|---|---|
| Sync bonus | **+10** | Driver and food ready within 1 min of each other |
| Temp penalty | **−1 per °C** | Applied at delivery for each degree below 75°C |
| Driver cancels | **−20** | Driver waited > 15 min — catastrophic |
| Timeout | **−15** | Episode expires at 60 min |
| Shaping signal | **−0.05×temp_drop/step** | Per-step gradient when food waits |

### Episode Termination

1. **Delivered** — driver arrived AND food ready → scored + done
2. **Driver cancelled** — driver waited > 15 min for food (−20, done)
3. **Timeout** — 60 minutes elapsed without delivery (−15, done)

---

## Three Tasks (Easy → Hard)

| Task | Traffic | Prep Jitter | Driver Dist | Cooling | Baseline Score |
|---|---|---|---|---|---|
| `easy` | σ=0.10, mean=1.2× | ±1 min | ~3 km | 1°C/min | ~0.55 |
| `medium` | σ=0.30, mean=1.5× | ±3 min | ~4 km | 1.5°C/min | ~0.35 |
| `hard` | σ=0.50, mean=1.8× + spikes | ±5 min | ~5 km | 2°C/min | ~0.18 |

Hard task adds random traffic spikes (5% chance per step, +0.4× multiplier).

---

## Setup

### Local (Python)

```bash
git clone https://huggingface.co/spaces/your-username/kitchenflow-v1
cd kitchenflow-v1
pip install -r requirements.txt

# Run the API server
python app.py

# Or run the baseline benchmark directly
python baseline.py --task all --n 20
```

### Docker

```bash
# Build
docker build -t your-username/kitchenflow:latest .

# Run
docker run -p 7860:7860 your-username/kitchenflow:latest

# Pull from Docker Hub
docker pull your-username/kitchenflow:latest
```

The server starts at `http://localhost:7860`. Visit `/docs` for Swagger UI.

---

## HTTP API

### Reset

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"session_id": "s1", "task": "medium", "seed": 42}'
```

```json
{
  "observation": {
    "food_prep_progress": 0.0,
    "driver_dist": 4.23,
    "traffic_index": 1.48,
    "food_temp": 75.0
  },
  "session_id": "s1",
  "task": "medium"
}
```

### Step

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"session_id": "s1", "action": 0}'
```

```json
{
  "observation": { "food_prep_progress": 0.067, "driver_dist": 4.23, "traffic_index": 1.51, "food_temp": 75.0 },
  "reward": 0.0,
  "done": false,
  "info": {}
}
```

### Other endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Reset episode |
| `/step` | POST | Take one action |
| `/state` | GET | Full internal state (debug) |
| `/observation_space` | GET | Space definition |
| `/action_space` | GET | Action definition |
| `/tasks` | GET | Task registry |
| `/grade` | POST | Benchmark a built-in policy |
| `/demo` | GET | Run one full episode, return trace |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger UI |

---

## Python API

```python
from env import KitchenFlowEnv

env = KitchenFlowEnv(task="medium", seed=42)
obs = env.reset()

done = False
while not done:
    # Your policy here
    action = 1 if obs["food_prep_progress"] > 0.80 else 0
    obs, reward, done, info = env.step(action)

print(info)
# {'termination': 'delivered', 'food_temp': 63.5, 'sync_bonus': False,
#  'driver_wait_min': 0, 'food_wait_min': 8, 'temp_penalty': 11.5}
```

### Useful utilities

```python
env.eta_food()    # estimated minutes until food is ready
env.eta_driver()  # estimated minutes until driver arrives (if summoned)
env.state()       # full internal state including hidden variables
```

---

## Baseline Benchmark

```bash
# Threshold policy (80% rule)
python baseline.py --policy threshold --threshold 0.80 --n 20

# ETA-matching policy (smarter rule)
python baseline.py --policy eta --n 20

# Compare policies
python baseline.py --policy threshold --task hard --n 50
```

**Reproducible scores (threshold=0.80, n=20, fixed seeds):**

| Task | Mean Reward | Delivery Rate | Sync Rate | Score |
|---|---|---|---|---|
| easy | +3.8 | 88% | 22% | 0.554 |
| medium | −1.2 | 72% | 11% | 0.348 |
| hard | −7.4 | 51% | 4% | 0.182 |

---

## Training an Agent

```python
# Gymnasium-compatible wrapper
from env import KitchenFlowEnv
import numpy as np

class GymWrapper:
    def __init__(self, task="medium"):
        self.env = KitchenFlowEnv(task=task)
        self.observation_space = ...  # Box(low, high, shape=(4,))
        self.action_space = ...       # Discrete(2)

    def reset(self, seed=None):
        obs = self.env.reset(seed=seed)
        return np.array(list(obs.values()), dtype=np.float32), {}

    def step(self, action):
        obs, reward, done, info = self.env.step(int(action))
        return np.array(list(obs.values()), dtype=np.float32), reward, done, False, info
```

The environment is compatible with Stable-Baselines3 (PPO, SAC, DQN) and any
framework that supports the `step(action) → (obs, reward, done, info)` interface.

---

## Grading

Each task has a grader that maps episodes to a normalized [0.0, 1.0] score:

```python
from tasks import grade_all

# Your policy: callable(obs, env) → int
def my_policy(obs, env):
    return 1 if obs["food_prep_progress"] > 0.75 else 0

scores = grade_all(my_policy)
# easy:   0.621
# medium: 0.412
# hard:   0.231
# Overall: 0.421
```

**Grader design:**
- `easy` — normalised reward + sync bonus weight
- `medium` — normalised reward + variance penalty + cancellation downweight
- `hard` — composite of reward, delivery rate, food temperature quality, and sync rate

---

## Data Sources

Traffic patterns are parametrically simulated using a mean-reverting random walk
calibrated against Uber Movement urban travel time distributions. Kitchen
busyness patterns follow distributions from the Yelp Open Dataset.

---

## File Structure

```
kitchenflow-v1/
├── env.py          Core environment (step/reset/state API)
├── tasks.py        Three tasks + graders + LLM grader prompt
├── baseline.py     Rule-based baseline + benchmark runner
├── app.py          FastAPI HTTP server (HF Spaces entry point)
├── openenv.yaml    OpenEnv specification
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## License

MIT — free to use, modify, and build on.

---

## Citation

```bibtex
@misc{kitchenflow2024,
  title     = {KitchenFlow-v1: A Ghost Kitchen Dispatch Environment for RL},
  author    = {Your Name},
  year      = {2024},
  url       = {https://huggingface.co/spaces/your-username/kitchenflow-v1},
  note      = {OpenEnv-compatible real-world logistics environment}
}
```
