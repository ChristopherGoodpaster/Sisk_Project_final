# Sisk_final_2.0.py
# Sisk Ballistics — Bore vs LOS + Cant + G1 trajectory graph (with view modes + ballistic table)

import math
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd  # for the ballistic table & shot details

from external_ballistics_g1 import (
    DEFAULT_V0_FPS,
    DEFAULT_BC_G1,
    ENV,
    solve_bore_angle,
    impact_at_range_with_cant,
    trajectory_vs_los_table,
)

# ---------- Page ----------
st.set_page_config(page_title="Sisk — Bore vs LOS + Cant + G1 Trajectory", layout="centered")
st.title("Sisk Ballistics — Bore axis vs Line-of-Sight, Cant & G1 Trajectory")
st.caption("Muzzle velocity locked at 2650 fps, BC(G1)=0.462 (168gr SMK). G1 external ballistics + cant rotation.")

# ---------- Constants ----------
MUZZLE_V_FPS = DEFAULT_V0_FPS
SHOT_COLORS = ["red", "blue", "green"]
BULLET_WEIGHT_GRAINS = 168.0  # for energy calculation

# ---------- Session & callbacks ----------
def init_session_state():
    defaults = {
        "shots": [],
        "range_yd": 200,
        "cant_deg": 0.0,
        "show_grid": True,
        "sight_height": 2.5,
        "zero_range": 100,
        "cant_deg_num": 0.0,
        "cant_deg_slide": 0.0,
        "auto_zoom": True,
        "target_radius_in": 9.0,
        "traj_max_range_yd": 1000.0,   # default dev max range out to 1000 yd
        "traj_step_yd": 50.0,          # default 50 yd steps
        # static profile (shown under plot)
        "profile_caliber": "7.62 x 51",
        "profile_bullet": "168 gr Sierra MatchKing",
        "profile_bc": ".462 (G1)",
        "profile_twist": "1:12",
        "profile_notes": "Muzzle velocity 2650 FPS",
        "view_mode": "Target / Cant view",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    st.session_state.cant_deg_num = float(st.session_state.cant_deg)
    st.session_state.cant_deg_slide = float(st.session_state.cant_deg)

def reset_all():
    st.session_state.shots = []
    st.session_state.range_yd = 200
    st.session_state.cant_deg = 0.0
    st.session_state.cant_deg_num = 0.0
    st.session_state.cant_deg_slide = 0.0
    st.session_state.show_grid = True
    st.session_state.sight_height = 2.5
    st.session_state.zero_range = 100
    st.session_state.auto_zoom = True
    st.session_state.target_radius_in = 9.0
    st.session_state.traj_max_range_yd = 1000.0
    st.session_state.traj_step_yd = 50.0
    st.session_state.profile_caliber = "7.62 x 51"
    st.session_state.profile_bullet = "168 gr Sierra MatchKing"
    st.session_state.profile_bc = ".462 (G1)"
    st.session_state.profile_twist = "1:12"
    st.session_state.profile_notes = "Muzzle velocity 2650 FPS"
    st.session_state.view_mode = "Target / Cant view"

def add_shot(index: int, target_radius_in: float, theta_bore_rad: float):
    h_in, v_in, t_sec, y_rel = impact_at_range_with_cant(
        MUZZLE_V_FPS, DEFAULT_BC_G1, theta_bore_rad,
        st.session_state.range_yd,
        st.session_state.sight_height,
        st.session_state.cant_deg,
        ENV,
    )
    off = math.hypot(h_in, v_in) > target_radius_in
    if len(st.session_state.shots) < 3:
        st.session_state.shots.append({
            "index": index,
            "cant": float(st.session_state.cant_deg),
            "h_in": float(h_in),
            "v_in": float(v_in),
            "t_sec": float(t_sec),
            "y_rel_LOS_in": float(y_rel),
            "off_target": bool(off),
            "color": SHOT_COLORS[len(st.session_state.shots)]
        })

# --- Cant sync callbacks ---
def _set_cant_from_num():
    st.session_state.cant_deg = float(st.session_state.cant_deg_num)
    st.session_state.cant_deg_slide = float(st.session_state.cant_deg)

def _set_cant_from_slide():
    st.session_state.cant_deg = float(st.session_state.cant_deg_slide)
    st.session_state.cant_deg_num = float(st.session_state.cant_deg)

init_session_state()

# ---------- Compute bore angle upfront (so we can use it in sidebar buttons) ----------
theta_bore_rad = solve_bore_angle(
    MUZZLE_V_FPS,
    DEFAULT_BC_G1,
    st.session_state.zero_range,
    st.session_state.sight_height,
    ENV,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Controls")
    st.radio(
        "View",
        ["Target / Cant view", "Trajectory (dev view)", "Ballistics table (dev)"],
        key="view_mode",
    )

    # Fire shot buttons at the top of the sidebar in target view
    if st.session_state.view_mode == "Target / Cant view":
        st.markdown("### Fire shots")
        cols2 = st.columns(3)
        for i, col in enumerate(cols2, start=1):
            with col:
                st.button(
                    f"Shot {i}",
                    key=f"shoot_btn_active_{i}",
                    on_click=add_shot,
                    args=(i, st.session_state.target_radius_in, theta_bore_rad),
                    disabled=(len(st.session_state.shots) >= 3),
                )

    # Common controls (apply to all views)
    st.number_input(
        "Range (yards)",
        min_value=10,
        max_value=2000,
        step=50,              # 50-yard increments
        key="range_yd",
    )
    st.number_input(
        "Sight height (in) — LOS above bore (2.5–5.0)",
        min_value=2.5,
        max_value=5.0,
        step=0.1,
        key="sight_height",
    )
    st.number_input(
        "Zero range (yards)",
        min_value=10,
        max_value=1000,
        step=50,              # 50-yard increments
        key="zero_range",
    )

    st.markdown("### Cant (degrees clockwise)")
    ccols = st.columns([1, 1.6])
    with ccols[0]:
        st.number_input(
            "Cant (°)",
            min_value=-60.0,
            max_value=60.0,
            step=0.1,
            key="cant_deg_num",
            on_change=_set_cant_from_num,
        )
    with ccols[1]:
        st.slider(
            " ",
            -60.0,
            60.0,
            step=0.1,
            key="cant_deg_slide",
            on_change=_set_cant_from_slide,
            label_visibility="collapsed",
        )

    st.checkbox("Show 1-in grid", key="show_grid")
    st.checkbox("Auto-zoom to include all impacts", key="auto_zoom")
    st.number_input(
        "Target radius (in)",
        min_value=2.0,
        max_value=36.0,
        step=0.5,
        key="target_radius_in",
    )

    # Trajectory controls ONLY in trajectory dev view
    if st.session_state.view_mode == "Trajectory (dev view)":
        st.markdown("### Trajectory graph (dev)")
        st.number_input(
            "Trajectory max range (yd)",
            min_value=100.0,
            max_value=2000.0,
            step=50.0,
            key="traj_max_range_yd",
        )
        st.selectbox(
            "Trajectory step (yd)",
            options=[25.0, 50.0, 100.0],
            key="traj_step_yd",
            index=[25.0, 50.0, 100.0].index(st.session_state.traj_step_yd),
        )

    st.button("Reset all", on_click=reset_all)

    with st.expander("Rifle & Ammo Profile (static)"):
        st.text_input("Caliber", key="profile_caliber")
        st.text_input("Bullet", key="profile_bullet")
        st.text_input("BC", key="profile_bc")
        st.text_input("Twist rate", key="profile_twist")
        st.text_area("Notes", key="profile_notes", height=80)

# ---------- Calculations (G1 model) ----------
# Recompute bore after any widget changes (cheap, and keeps everything in sync)
theta_bore_rad = solve_bore_angle(
    MUZZLE_V_FPS,
    DEFAULT_BC_G1,
    st.session_state.zero_range,
    st.session_state.sight_height,
    ENV,
)

h_in, v_in, t_sec, y_rel_LOS_in = impact_at_range_with_cant(
    MUZZLE_V_FPS,
    DEFAULT_BC_G1,
    theta_bore_rad,
    st.session_state.range_yd,
    st.session_state.sight_height,
    st.session_state.cant_deg,
    ENV,
)
off_preview = math.hypot(h_in, v_in) > st.session_state.target_radius_in

# ---------- VIEW 1: Target / Cant ----------
if st.session_state.view_mode == "Target / Cant view":

    fig, ax = plt.subplots(figsize=(6.8, 6.8))

    def draw_target(ax, title, show_grid=False, radius=12.0):
        ax.set_aspect("equal", "box")
        ax.set_title(title)
        rings = [1, 3, 6, 9, radius]
        for r in rings:
            ax.add_patch(plt.Circle((0, 0), r, fill=False, linewidth=0.7))
        ax.axhline(0, linewidth=0.7)
        ax.axvline(0, linewidth=0.7)
        if show_grid:
            ticks = np.arange(-max(radius, 9), max(radius, 9) + 1, 1)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.grid(True, linewidth=0.3)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

    title = f"Impacts at {st.session_state.range_yd} yd (target center = 0,0)"
    draw_target(
        ax,
        title,
        show_grid=st.session_state.show_grid,
        radius=st.session_state.target_radius_in,
    )

    # --- Preview impact (still shows a small label) ---
    ax.scatter(
        h_in,
        v_in,
        s=80,
        facecolors="none",
        edgecolors="k",
        linewidths=1.0,
        zorder=6,
    )
    if off_preview:
        ax.annotate(
            "PREVIEW: OFF TARGET",
            xy=(0.5, 0.95),
            xycoords="axes fraction",
            color="red",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.9),
        )
    else:
        ax.annotate(
            f"Preview\n{h_in:+.2f} in, {v_in:+.2f} in",
            xy=(h_in, v_in),
            xytext=(8, -8),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
        )

    # --- Recorded shots: dots only, NO text label to avoid clutter ---
    for s in st.session_state.shots:
        display_h, display_v, color = s["h_in"], s["v_in"], s["color"]
        if math.hypot(display_h, display_v) > st.session_state.target_radius_in:
            angle = math.atan2(display_v, display_h)
            clip_x = (
                st.session_state.target_radius_in * 0.98 * math.cos(angle)
            )
            clip_y = (
                st.session_state.target_radius_in * 0.98 * math.sin(angle)
            )
            ax.scatter(
                clip_x,
                clip_y,
                s=140,
                marker="X",
                c=color,
                edgecolors="k",
                linewidths=1.0,
                zorder=7,
            )
        else:
            ax.scatter(
                display_h,
                display_v,
                s=90,
                c=color,
                edgecolors="k",
                linewidths=0.8,
                zorder=7,
            )

    # auto-zoom
    xs = [h_in] + [s["h_in"] for s in st.session_state.shots]
    ys = [v_in] + [s["v_in"] for s in st.session_state.shots]
    if st.session_state.auto_zoom and xs and ys:
        max_abs = max(1.0, max(map(abs, xs + ys)))
        pad = 0.1 * max_abs
        xmin = min(-st.session_state.target_radius_in, min(xs) - pad)
        xmax = max(st.session_state.target_radius_in, max(xs) + pad)
        ymin = min(-st.session_state.target_radius_in, min(ys) - pad)
        ymax = max(st.session_state.target_radius_in, max(ys) + pad)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_xlim(
            -st.session_state.target_radius_in,
            st.session_state.target_radius_in,
        )
        ax.set_ylim(
            -st.session_state.target_radius_in,
            st.session_state.target_radius_in,
        )

    if st.session_state.shots:
        handles, labels = [], []
        for s in st.session_state.shots:
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=s["color"],
                    markeredgecolor="k",
                    markersize=8,
                )
            )
            labels.append(
                f"Shot {s['index']} ({s['cant']:+.2f}°)"
                + (" — OFF" if s["off_target"] else "")
            )
        ax.legend(handles, labels, loc="upper right", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)

    # Optional: shot details as a table (so you can still see the numbers cleanly)
    if st.session_state.shots:
        st.subheader("Shot details (numerical)")
        df_shots = pd.DataFrame(
            [
                {
                    "Shot": s["index"],
                    "Cant (deg)": s["cant"],
                    "Horizontal (in)": s["h_in"],
                    "Vertical (in)": s["v_in"],
                    "Time of flight (s)": s["t_sec"],
                    "Path vs LOS (in)": s["y_rel_LOS_in"],
                    "Off target": s["off_target"],
                }
                for s in st.session_state.shots
            ]
        )
        st.dataframe(
            df_shots.style.format(
                {
                    "Cant (deg)": "{:+.2f}",
                    "Horizontal (in)": "{:+.2f}",
                    "Vertical (in)": "{:+.2f}",
                    "Time of flight (s)": "{:.4f}",
                    "Path vs LOS (in)": "{:+.2f}",
                }
            ),
            width="stretch",
        )

    # Below-plot metrics for current preview
    st.subheader("Current preview (G1 + cant)")
    pp1, pp2, pp3 = st.columns(3)
    pp1.metric("Time of flight (s)", f"{t_sec:.4f}")
    pp2.metric("Bullet vs LOS (in)", f"{y_rel_LOS_in:+.2f}")
    pp3.metric("Cant (deg)", f"{st.session_state.cant_deg:+.2f}")

    cant_rad = math.radians(st.session_state.cant_deg)
    cant_error_in = abs(y_rel_LOS_in) * abs(math.sin(cant_rad))
    st.write(
        f"Estimated cant error: **{cant_error_in:.2f} in**  "
        f"*(|y_vs_LOS| × sin|cant|)*"
    )

    if off_preview:
        st.error(
            f"Preview at {st.session_state.range_yd} yd → OFF TARGET "
            f"(impact rel center = h={h_in:+.2f} in, v={v_in:+.2f} in)"
        )
    else:
        st.write(
            f"Preview at {st.session_state.range_yd} yd → impact rel center = "
            f"h = **{h_in:+.2f} in**, v = **{v_in:+.2f} in**"
        )

    st.subheader("Rifle & Ammo Profile")
    st.markdown(
        f"- **Caliber:** {st.session_state.profile_caliber}\n"
        f"- **Bullet:** {st.session_state.profile_bullet}\n"
        f"- **BC:** {st.session_state.profile_bc}\n"
        f"- **Twist:** {st.session_state.profile_twist}\n"
        + (
            f"- **Notes:** {st.session_state.profile_notes}"
            if st.session_state.profile_notes.strip()
            else ""
        )
    )

    st.markdown(
        "- G1 external ballistics engine approximates tools like the Federal calculator.\n"
        "- Differences can still occur due to drag table details, atmospherics, and rounding.\n"
        "- Cant error is modeled as vertical drop rotated around LOS (gravity-only windage from cant).\n"
    )

# ---------- VIEW 2: Trajectory (dev) ----------
elif st.session_state.view_mode == "Trajectory (dev view)":
    st.subheader("Trajectory vs Line-of-Sight (G1 model)")

    traj = trajectory_vs_los_table(
        MUZZLE_V_FPS,
        DEFAULT_BC_G1,
        st.session_state.zero_range,
        st.session_state.sight_height,
        ENV,
        st.session_state.traj_max_range_yd,
        step_yd=st.session_state.traj_step_yd,
    )

    fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
    ax2.plot(
        traj["range_yd"],
        traj["path_in"],
        marker="o",
        linestyle="-",
        linewidth=1.2,
    )
    ax2.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Range (yd)")
    ax2.set_ylabel("Bullet path vs LOS (in)")
    ax2.grid(True, linestyle=":")
    st.pyplot(fig2)

    st.subheader("Zeroing (G1)")
    theta_deg = math.degrees(traj["theta_bore_rad"])
    theta_moa = theta_deg * 60.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Zero range (yd)", f"{st.session_state.zero_range:.0f}")
    c2.metric("Sight height (in)", f"{st.session_state.sight_height:.2f}")
    c3.metric("Bore angle", f"{theta_deg:.4f}°  ({theta_moa:.2f} MOA)")

    st.info(
        "Trajectory view is for development/verification. "
        "For demos, use the **Target / Cant view** so the curve and its controls stay hidden."
    )

# ---------- VIEW 3: Ballistics table (dev) ----------
else:
    st.subheader("Ballistics table (G1, 0–1000 yd, 50-yd steps)")

    # Generate trajectory out to 1000 yd in 50-yd steps
    table_max_range_yd = 1000.0
    table_step_yd = 50.0

    traj = trajectory_vs_los_table(
        MUZZLE_V_FPS,
        DEFAULT_BC_G1,
        st.session_state.zero_range,
        st.session_state.sight_height,
        ENV,
        table_max_range_yd,
        step_yd=table_step_yd,
    )

    ranges = traj["range_yd"]
    drops = traj["path_in"]  # already "bullet vs LOS" in inches
    velocities = traj["vel_fps"]

    # Energy (ft-lb): E = (w_grains * v^2) / 450240
    energies = (BULLET_WEIGHT_GRAINS * velocities**2) / 450240.0

    # Wind drift is not modeled yet: placeholder zeros to match the Federal-style column
    wind_drift = np.zeros_like(ranges)

    df = pd.DataFrame(
        {
            "Range (yd)": ranges.astype(int),
            "Drop (in)": drops,
            "Wind Drift (in)": wind_drift,
            "Velocity (fps)": velocities,
            "Energy (ft-lb)": energies,
        }
    )

    # Header-style parameters (similar to the Federal popup)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"**Bullet Weight (grains):** {BULLET_WEIGHT_GRAINS:.0f}\n\n"
            f"**G1 Ballistic Coefficient:** {DEFAULT_BC_G1:.3f}\n\n"
            f"**Sight Height:** {st.session_state.sight_height:.2f} in\n\n"
            f"**Zero Range:** {st.session_state.zero_range:.0f} yd"
        )
    with c2:
        st.markdown(
            f"**Muzzle Velocity:** {MUZZLE_V_FPS:.0f} fps\n\n"
            f"**Temperature:** 59 °F (assumed)\n\n"
            f"**Altitude:** 0 ft (sea level)\n\n"
            f"**Max Range:** {int(table_max_range_yd)} yd"
        )

    st.markdown("### Table (compare with Federal ballistics calculator)")
    st.dataframe(
        df.style.format(
            {
                "Drop (in)": "{:+.1f}",
                "Wind Drift (in)": "{:+.1f}",
                "Velocity (fps)": "{:.0f}",
                "Energy (ft-lb)": "{:.0f}",
            }
        ),
        width="stretch",
    )

    st.info(
        "Drop values are bullet path vs line-of-sight (same convention as Federal). "
        "Wind drift is shown as 0.0 in this version; crosswind is not modeled yet."
    )
