"""
KitchenFlow-v1 — Streamlit Dashboard
=====================================
Talks to your running app.py server at localhost:7860.

Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import pandas as pd

BASE_URL   = "http://localhost:7860"
SESSION_ID = "streamlit_ui"

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "KitchenFlow-v1",
    page_icon  = "🍔",
    layout     = "wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0f0f0f; color: #ececec; }
[data-testid="stSidebar"] { background: #141414; border-right: 1px solid #222; }

.kf-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem; font-weight: 700;
    color: #f5a623; margin: 0; line-height: 1;
}
.kf-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem; color: #444;
    letter-spacing: 3px; text-transform: uppercase;
    margin-top: 4px; margin-bottom: 20px;
}
.kf-card {
    background: #1a1a1a; border: 1px solid #252525;
    border-radius: 10px; padding: 18px 20px; text-align: center;
}
.kf-card-label {
    font-family: 'Space Mono', monospace; font-size: 0.6rem;
    color: #555; text-transform: uppercase;
    letter-spacing: 2px; margin-bottom: 8px;
}
.kf-card-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem; font-weight: 700; color: #f5a623;
}
.kf-card-value.green { color: #4caf7d; }
.kf-card-value.red   { color: #e05252; }
.kf-card-value.blue  { color: #64b5f6; }

.kf-log {
    background: #111; border: 1px solid #1e1e1e;
    border-radius: 8px; padding: 14px 16px;
    font-family: 'Space Mono', monospace; font-size: 0.7rem;
    color: #777; max-height: 300px; overflow-y: auto; line-height: 2;
}
.kf-section {
    font-family: 'Space Mono', monospace; font-size: 0.62rem;
    color: #444; text-transform: uppercase; letter-spacing: 3px;
    border-bottom: 1px solid #1e1e1e;
    padding-bottom: 6px; margin: 20px 0 14px;
}
.score-bar-wrap {
    background: #1a1a1a; border-radius: 6px; height: 8px; width: 100%; margin: 4px 0 12px;
}
.score-bar-fill { height: 8px; border-radius: 6px; }

.stButton > button {
    background: #f5a623 !important; color: #0f0f0f !important;
    font-family: 'Space Mono', monospace !important; font-weight: 700 !important;
    font-size: 0.72rem !important; border: none !important;
    border-radius: 8px !important; width: 100% !important;
}
.stButton > button:hover { opacity: 0.8 !important; }
.stButton > button:disabled { background: #2a2a2a !important; color: #555 !important; }

#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

for k, v in {
    "obs": None, "done": False, "reward_total": 0.0,
    "step_count": 0, "summoned": False, "termination": None,
    "log": [], "grade_results": None, "dataset_df": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ───────────────────────────────────────────────────────────────────

def api_post(path, payload):
    try:
        r = requests.post(f"{BASE_URL}{path}", json=payload, timeout=30)
        return r.json(), r.status_code
    except Exception as e:
        return {"error": str(e)}, 500

def api_get(path, params=None):
    try:
        r = requests.get(f"{BASE_URL}{path}", params=params, timeout=30)
        return r.json(), r.status_code
    except Exception as e:
        return {"error": str(e)}, 500

def server_ok():
    try:
        return requests.get(f"{BASE_URL}/health", timeout=2).status_code == 200
    except:
        return False

def temp_color(t):
    if t >= 70: return "green"
    if t >= 60: return ""
    return "red"

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown('<div class="kf-title">🍔 KitchenFlow-v1</div>', unsafe_allow_html=True)
st.markdown('<div class="kf-sub">Ghost Kitchen Dispatcher · RL Environment</div>', unsafe_allow_html=True)

if server_ok():
    st.markdown(
        '🟢 <span style="font-family:\'Space Mono\',monospace;font-size:0.68rem;color:#4caf7d">'
        'SERVER LIVE · localhost:7860</span>',
        unsafe_allow_html=True,
    )
else:
    st.error("⚠️  Cannot reach localhost:7860 — make sure `python app.py` is running first.")
    st.stop()

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_play, tab_grade, tab_data = st.tabs(["🎮  Play Episode", "📊  Grade Policy", "📂  Dataset"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PLAY  (Reset + Step)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_play:

    ctrl, main = st.columns([1, 2.4], gap="large")

    # ── Controls ──────────────────────────────────────────────────────────────
    with ctrl:
        st.markdown('<div class="kf-section">Episode Settings</div>', unsafe_allow_html=True)

        task = st.selectbox("Task Difficulty", ["easy", "medium", "hard"])
        seed = st.number_input("Seed", min_value=0, max_value=9999, value=42)

        st.markdown('<div class="kf-section">Actions</div>', unsafe_allow_html=True)

        # RESET
        if st.button("🔄  Reset Episode"):
            data, code = api_post("/reset", {
                "session_id": SESSION_ID,
                "task":       task,
                "seed":       int(seed),
            })
            if code == 200:
                st.session_state.obs          = data["observation"]
                st.session_state.done         = False
                st.session_state.reward_total = 0.0
                st.session_state.step_count   = 0
                st.session_state.summoned     = False
                st.session_state.termination  = None
                st.session_state.log          = [f"▶ Episode started — task={task}  seed={seed}"]
                st.rerun()
            else:
                st.error(f"Reset failed: {data.get('error', data)}")

        st.markdown("")

        # Action picker
        is_ready = st.session_state.obs is not None and not st.session_state.done
        action_choice = st.radio(
            "Next Action",
            ["0 — Wait", "1 — Summon Driver"],
            index=0,
            disabled=not is_ready,
        )
        action = 1 if action_choice.startswith("1") else 0

        # STEP
        if st.button("▶  Take Step", disabled=not is_ready):
            data, code = api_post("/step", {
                "session_id": SESSION_ID,
                "action":     action,
            })
            if code == 200:
                obs = data["observation"]
                st.session_state.obs           = obs
                st.session_state.reward_total += data["reward"]
                st.session_state.done          = data["done"]
                st.session_state.step_count   += 1
                term = data.get("info", {}).get("termination", "running")
                st.session_state.termination   = term

                # Build log entry
                if term == "delivered":
                    entry = f"✅ DELIVERED — temp={obs['food_temp']:.1f}°C  reward={data['reward']:+.2f}"
                elif term == "timeout":
                    entry = f"⏰ TIMEOUT — episode ended at 60 min"
                else:
                    summon_tag = " 🚗 SUMMON DRIVER" if (action == 1 and not st.session_state.summoned) else ""
                    entry = (
                        f"t={st.session_state.step_count:>2}m  "
                        f"prep={obs['food_prep_progress']:.0%}  "
                        f"dist={obs['driver_dist']:.1f}km  "
                        f"traffic={obs['traffic_index']:.2f}×  "
                        f"temp={obs['food_temp']:.1f}°C  "
                        f"r={data['reward']:+.2f}{summon_tag}"
                    )

                if action == 1:
                    st.session_state.summoned = True

                st.session_state.log.append(entry)
                st.rerun()
            else:
                st.error(f"Step failed: {data.get('error', data)}")

        # Episode result banner
        if st.session_state.done:
            term = st.session_state.termination
            if term == "delivered":
                st.success("✅ Delivered successfully!")
            elif term == "timeout":
                st.warning("⏰ Episode timed out (60 min)")
            else:
                st.info(f"Episode ended: {term}")

    # ── Main display ──────────────────────────────────────────────────────────
    with main:

        if st.session_state.obs is None:
            st.markdown("""
            <div style="text-align:center; padding:100px 0; color:#2a2a2a;">
                <div style="font-size:3rem;">🍔</div>
                <div style="font-family:'Space Mono',monospace; font-size:0.75rem;
                            letter-spacing:3px; margin-top:16px;">
                    PRESS RESET TO START AN EPISODE
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            obs = st.session_state.obs

            # 4 observation cards
            c1, c2, c3, c4 = st.columns(4)
            prep_pct = int(obs["food_prep_progress"] * 100)

            with c1:
                color = "green" if prep_pct >= 100 else ""
                st.markdown(f"""
                <div class="kf-card">
                    <div class="kf-card-label">Food Prep</div>
                    <div class="kf-card-value {color}">{prep_pct}%</div>
                </div>""", unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="kf-card">
                    <div class="kf-card-label">Driver Distance</div>
                    <div class="kf-card-value blue">{obs['driver_dist']:.2f} km</div>
                </div>""", unsafe_allow_html=True)

            with c3:
                color = "red" if obs["traffic_index"] > 1.8 else ""
                st.markdown(f"""
                <div class="kf-card">
                    <div class="kf-card-label">Traffic Index</div>
                    <div class="kf-card-value {color}">{obs['traffic_index']:.2f}×</div>
                </div>""", unsafe_allow_html=True)

            with c4:
                tc = temp_color(obs["food_temp"])
                st.markdown(f"""
                <div class="kf-card">
                    <div class="kf-card-label">Food Temp</div>
                    <div class="kf-card-value {tc}">{obs['food_temp']:.1f}°C</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")

            # Food prep progress bar + episode stats
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown('<div class="kf-section">Food Preparation Progress</div>', unsafe_allow_html=True)
                st.progress(float(obs["food_prep_progress"]))
                st.caption(f"{'✅ Food is ready!' if obs['food_prep_progress'] >= 1.0 else 'Still cooking...'}")

            with col_b:
                st.markdown('<div class="kf-section">Episode Stats</div>', unsafe_allow_html=True)
                s1, s2, s3 = st.columns(3)
                s1.metric("Steps", st.session_state.step_count)
                s2.metric("Total Reward", f"{st.session_state.reward_total:.2f}")
                s3.metric("Driver", "Summoned ✅" if st.session_state.summoned else "Waiting ❌")

            # Step log
            st.markdown('<div class="kf-section">Step Log</div>', unsafe_allow_html=True)
            log_lines = list(reversed(st.session_state.log))
            log_html  = "<br>".join(
                f'<span style="color:#f5a623;font-weight:700">{l}</span>'
                if "SUMMON" in l or "DELIVERED" in l
                else f'<span style="color:#e05252">{l}</span>'
                if "TIMEOUT" in l
                else f'<span>{l}</span>'
                for l in log_lines
            )
            st.markdown(f'<div class="kf-log">{log_html}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GRADE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_grade:

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown('<div class="kf-section">Grade Settings</div>', unsafe_allow_html=True)

        policy = st.selectbox(
            "Policy",
            ["threshold", "eta"],
            format_func=lambda x: "Threshold (80% rule)" if x == "threshold" else "ETA Match (smarter)",
        )

        threshold = 0.80
        if policy == "threshold":
            threshold = st.slider("Summon Threshold", 0.50, 1.0, 0.80, 0.05,
                                  help="Summon driver when food prep hits this %")

        n_eps = st.slider("Episodes per Task", 5, 50, 10,
                          help="More episodes = more accurate score but slower")

        st.markdown("")

        if st.button("🏁  Run Grader"):
            with st.spinner("Running across easy / medium / hard..."):
                data, code = api_post("/grade", {
                    "policy":     policy,
                    "threshold":  threshold,
                    "n_episodes": n_eps,
                })
                if code == 200:
                    st.session_state.grade_results = data
                    st.rerun()
                else:
                    st.error(f"Grade failed: {data.get('error', data)}")

    with right:
        st.markdown('<div class="kf-section">Results</div>', unsafe_allow_html=True)

        if st.session_state.grade_results is None:
            st.markdown("""
            <div style="text-align:center; padding:60px 0; color:#2a2a2a;">
                <div style="font-family:'Space Mono',monospace; font-size:0.75rem; letter-spacing:3px;">
                    RUN THE GRADER TO SEE SCORES
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            g       = st.session_state.grade_results
            scores  = g.get("scores", {})
            overall = g.get("overall", 0)

            # Overall score card
            oc = "green" if overall >= 0.6 else "red" if overall < 0.35 else ""
            st.markdown(f"""
            <div class="kf-card" style="margin-bottom:24px;">
                <div class="kf-card-label">Overall Score · {g.get('policy','')}</div>
                <div class="kf-card-value {oc}">{overall:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

            # Per-task bars
            task_colors = {"easy": "#4caf7d", "medium": "#f5a623", "hard": "#e05252"}
            for task_name, score in scores.items():
                pct   = int(score * 100)
                color = task_colors.get(task_name, "#f5a623")
                st.markdown(f"""
                <div style="margin-bottom:18px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span style="font-size:0.9rem; color:#aaa; text-transform:capitalize;">{task_name}</span>
                        <span style="font-family:'Space Mono',monospace; font-size:0.9rem;
                                     color:{color}; font-weight:700;">{score:.4f}</span>
                    </div>
                    <div class="score-bar-wrap">
                        <div class="score-bar-fill" style="width:{pct}%; background:{color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.caption(f"Policy: {g.get('policy')} · {g.get('n_episodes')} episodes per task")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATASET
# ═══════════════════════════════════════════════════════════════════════════════

with tab_data:

    st.markdown('<div class="kf-section">Browse kitchenflow_dataset.csv</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns([1, 1, 1])
    with f1:
        filter_task = st.selectbox("Filter by Task", ["all", "easy", "medium", "hard"])
    with f2:
        limit = st.slider("Max rows", 10, 500, 50)
    with f3:
        st.markdown("<br>", unsafe_allow_html=True)
        load = st.button("📂  Load Dataset")

    if load:
        params = {"limit": limit}
        if filter_task != "all":
            params["task"] = filter_task
        data, code = api_get("/dataset", params=params)
        if code == 200:
            st.session_state.dataset_df = pd.DataFrame(data["rows"])
            st.caption(f"Showing {data['showing']} of {data['total_rows']} rows")
        else:
            st.error(f"Could not load dataset: {data.get('error', data)}")

    if st.session_state.dataset_df is not None:
        df = st.session_state.dataset_df

        # Quick stats row
        st.markdown('<div class="kf-section">Quick Stats</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", len(df))
        m2.metric("Avg Prep Time",
                  f"{df['prep_time_min'].mean():.1f} min" if "prep_time_min" in df.columns else "—")
        m3.metric("Avg Driver Dist",
                  f"{df['driver_dist_km'].mean():.2f} km" if "driver_dist_km" in df.columns else "—")
        m4.metric("Hot Outcomes",
                  f"{(df['outcome']=='hot').sum()}" if "outcome" in df.columns else "—")

        # Table
        st.markdown('<div class="kf-section">Data Table</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, height=400)

        # Download
        st.download_button(
            "⬇️  Download as CSV",
            data      = df.to_csv(index=False).encode(),
            file_name = f"kitchenflow_{filter_task}.csv",
            mime      = "text/csv",
        )