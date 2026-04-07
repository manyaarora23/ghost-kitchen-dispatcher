# 🍳 KitchenFlow-v1 — Ghost Kitchen Dispatcher

<p align="center">
<img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python">
<img src="https://img.shields.io/badge/FastAPI-API-green?logo=fastapi">
<img src="https://img.shields.io/badge/Reinforcement-Learning-orange">
<img src="https://img.shields.io/badge/License-MIT-lightgrey">
<img src="https://img.shields.io/badge/Status-Active-success">
</p>

<p align="center">
A reinforcement learning environment where an agent learns the optimal time to summon delivery drivers in a ghost kitchen.
</p>

---

# 📌 Overview

**KitchenFlow-v1** is a simulation environment for a **cloud kitchen dispatcher**.

The system must coordinate:

* 🍲 Food preparation time
* 🚗 Driver arrival time
* 🌡 Food temperature
* 🚦 Traffic conditions

The challenge is deciding **when to summon a driver** so that:

* Food is **ready when the driver arrives**
* The driver **doesn't wait**
* The food **remains hot**

This problem is framed as a **Reinforcement Learning environment**.

---

# 🎯 Motivation

Food delivery platforms like Uber Eats, DoorDash, and Swiggy face a real operational problem:

| Scenario                 | Outcome            |
| ------------------------ | ------------------ |
| Driver arrives too early | Driver idle cost   |
| Driver arrives too late  | Cold food          |
| Perfect timing           | Efficient delivery |

KitchenFlow allows training RL agents to **optimize dispatch timing** under uncertainty.

---

# 🧠 Environment Mechanics

At each **timestep (1 minute)** the agent observes the kitchen state and decides an action.

## Episode Flow

```
Food starts cooking
        ↓
Agent observes environment
        ↓
Agent chooses action
(wait or summon driver)
        ↓
Environment updates
        ↓
Reward calculated
        ↓
Episode ends when food is picked up or spoiled
```

---

# 🎮 Action Space

Discrete actions.

| Action | Description   |
| ------ | ------------- |
| `0`    | Wait          |
| `1`    | Summon driver |

Example:

```python
action = 0  # wait
action = 1  # summon driver
```

---

# 👁 Observation Space

The agent receives a structured observation dictionary.

| Variable           | Meaning                 | Range             |
| ------------------ | ----------------------- | ----------------- |
| food_prep_progress | % food prepared         | 0 → 1             |
| driver_dist        | distance of driver (km) | 0 → 5             |
| traffic_index      | traffic congestion      | 0 → 1             |
| food_temp          | food temperature °C     | 20 → 100          |
| driver_called      | driver summoned         | 0 / 1             |
| time_elapsed       | elapsed minutes         | 0 → episode limit |

Example:

```json
{
  "food_prep_progress": 0.72,
  "driver_dist": 2.1,
  "traffic_index": 0.43,
  "food_temp": 83.5,
  "driver_called": 0,
  "time_elapsed": 11
}
```

---

# 🧪 Tasks

KitchenFlow provides **three difficulty levels**.

## 🟢 Easy

* Predictable prep time
* Low traffic variability
* Faster drivers

Baseline score: **0.55**

---

## 🟡 Medium

* Moderate uncertainty
* Traffic fluctuations
* Variable prep speed

Baseline score: **0.35**

---

## 🔴 Hard

* High uncertainty
* Heavy traffic variability
* Tight timing window

Baseline score: **0.18**

---

# 🏆 Reward System

The reward function encourages **synchronization** between driver arrival and food readiness.

| Situation             | Reward      |
| --------------------- | ----------- |
| Perfect pickup timing | High reward |
| Driver waits          | Penalty     |
| Food becomes cold     | Penalty     |
| Successful delivery   | Bonus       |

---

# 🏗 Project Architecture

```
KitchenFlow
│
├── app.py
│   FastAPI server exposing the RL environment
│
├── env.py
│   Core environment simulation
│
├── tasks.py
│   Difficulty tasks and graders
│
├── baseline.py
│   Baseline policies
│
└── README.md
```

---

# ⚙️ Setup

## 1️⃣ Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/kitchenflow
cd kitchenflow
```

---

## 2️⃣ Install dependencies

```bash
pip install fastapi uvicorn pydantic
```

---

## 3️⃣ Run the API server

```bash
python app.py
```

Server runs at:

```
http://localhost:9500
```

Swagger API documentation:

```
http://localhost:9500/docs
```

---

# 🚀 Usage

## Reset environment

```
POST /reset
```

Example:

```json
{
  "session_id": "demo",
  "task": "easy"
}
```

---

## Take a step

```
POST /step
```

Example:

```json
{
  "session_id": "demo",
  "action": 0
}
```

---

## Get environment state

```
GET /state?session_id=demo
```

---

## List tasks

```
GET /tasks
```

---

## Evaluate baseline

```
POST /grade
```

Example:

```json
{
  "policy": "threshold",
  "threshold": 0.8,
  "n_episodes": 10
}
```

---

# 🤖 Baseline Agents

KitchenFlow includes built-in baseline policies.

## Threshold Policy

Summon a driver when food preparation crosses a threshold.

```
summon if prep_progress > 0.8
```

---

## ETA Policy

Estimates driver arrival time using:

* traffic
* distance
* prep progress

Then decides the summon time.

---

# 📊 Baseline Scores

| Task   | Score    |
| ------ | -------- |
| Easy   | **0.55** |
| Medium | **0.35** |
| Hard   | **0.18** |

These scores provide reference performance for RL agents.

---

# 🧪 Example RL Loop

```python
import requests

BASE = "http://localhost:9500"

requests.post(BASE + "/reset", json={"session_id":"agent"})

done = False

while not done:

    action = 0

    r = requests.post(
        BASE + "/step",
        json={"session_id":"agent","action":action}
    )

    data = r.json()

    print("reward:", data["reward"])

    done = data["done"]
```

---

# 📈 Possible Extensions

KitchenFlow can be extended with:

* Deep Reinforcement Learning agents
* Multi-driver dispatch
* Multi-order kitchens
* Real traffic APIs
* Food cooling physics

---

# 👩‍💻 Author

Created as a reinforcement learning environment for **delivery logistics optimization**.

---

# 📜 License

MIT License
