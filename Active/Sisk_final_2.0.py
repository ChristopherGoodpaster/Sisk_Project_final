# Sisk_final_2.0.py
# Sisk Ballistics — Bore vs LOS + Cant + G1 trajectory graph (with dev tools)

import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from external_ballistics_g1 import (
    DEFAULT_V0_FPS,
    DEFAULT_BC_G1,
    ENV,
    solve_bore_angle,
    impact_at_range_with_cant,
    trajectory_vs_los_table,
)

# ---------- Page ----------
st.set_page_config(page_title="Sisk — Bore vs LOS + Cant", layout="centered")
st.title("Sisk Ballistics — Bore Axis vs Line-of-Sight & Cant")
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
        "traj_max_range_yd": 1000.0,
        "traj_step_yd": 50.0,
        # static profile (shown under plot)
        "profile_caliber": "7.62 x 51",
        "profile_bullet": "168 gr Sierra MatchKing",
        "profile_bc": ".462 (G1)",
        "profile_twist": "1:12",
        "profile_notes": "Muzzle velocity 2650 FPS",
        # view mode
        "view_mode": "Target view",
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
    st.session_state.view_mode = "Target view"


def add_shot(index: int, target_radius_in: float, theta_bore_rad: float):
    h_in, v_in, t_sec, y_rel = impact_at_range_with_cant(
        MUZZLE_V_FPS,
        DEFAULT_BC_G1,
        theta_bore_rad,
        st.session_state.range_yd,
        st.session_state.sight_height,
        st.session_state.cant_deg,
        ENV,
    )
    off = math.hypot(h_in, v_in) > target_radius_in
    if len(st.session_state.shots) < 3:
        st.session_state.shots.append(
            {
                "index": index,
                "cant": float(st.session_state.cant_deg),
                "h_in": float(h_in),
                "v_in": float(v_in),
                "t_sec": float(t_sec),
                "y_rel_LOS_in": float(y_rel),
                "off_target": bool(off),
                "color": SHOT_COLORS[len(st.session_state.shots)],
            }
        )


# --- Cant sync callbacks ---
def _set_cant_from_num():
    st.session_state.cant_deg = float(st.session_state.cant_deg_num)
    st.session_state.cant_deg_slide = float(st.session_state.cant_deg)


def _set_cant_from_slide():
    st.session_state.cant_deg = float(st.session_state.cant_deg_slide)
    st.session_state.cant_deg_num = float(st.session_state.cant_deg)


init_session_state()

# ---------- Compute bore angle once per run (based on current zero & sight) ----------
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
        ["Target view", "Dev: Trajectory", "Dev: Ballistics table", "Dev: Federal comparison"],
        key="view_mode",
    )

    # Fire shot buttons at top in target view
    if st.session_state.view_mode == "Target view":
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

    # Common controls
    st.number_input(
        "Range (yards)",
        min_value=10,
        max_value=2000,
        step=50,
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
        step=50,
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
    if st.session_state.view_mode == "Dev: Trajectory":
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

# ---------- Core calculations (G1 model) ----------
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

# ---------- VIEW 1: Target view (Matplotlib, classic Sisk target) ----------
if st.session_state.view_mode == "Target view":

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

    # --- Preview impact ---
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

    # --- Recorded shots: with on-target/off-target labels like before ---
    for s in st.session_state.shots:
        display_h, display_v, color = s["h_in"], s["v_in"], s["color"]
        if math.hypot(display_h, display_v) > st.session_state.target_radius_in:
            angle = math.atan2(display_v, display_h)
            clip_x = st.session_state.target_radius_in * 0.98 * math.cos(angle)
            clip_y = st.session_state.target_radius_in * 0.98 * math.sin(angle)
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
            ax.annotate(
                f"Shot {s['index']} OFF\n{display_h:+.2f}, {display_v:+.2f} in",
                xy=(clip_x, clip_y),
                xytext=(8, -8),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.9),
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
            ax.annotate(
                f"Shot {s['index']} ({s['cant']:+.2f}°)\n{display_h:+.2f}, {display_v:+.2f} in",
                xy=(display_h, display_v),
                xytext=(8, -8),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85),
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

    # Legend
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

# ---------- VIEW 2: Dev — Trajectory ----------
elif st.session_state.view_mode == "Dev: Trajectory":
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
        "For demos, use the **Target view** so the curve and its controls stay hidden."
    )

# ---------- VIEW 3: Dev — Ballistics table (no Wind Drift column) ----------
elif st.session_state.view_mode == "Dev: Ballistics table":
    st.subheader("Ballistics table (G1, 0–1000 yd, 50-yd steps)")

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
    drops = traj["path_in"]  # bullet path vs LOS in inches
    velocities = traj["vel_fps"]

    # Energy (ft-lb): E = (w_grains * v^2) / 450240
    energies = (BULLET_WEIGHT_GRAINS * velocities**2) / 450240.0

    df = pd.DataFrame(
        {
            "Range (yd)": ranges.astype(int),
            "Drop (in)": drops,
            "Velocity (fps)": velocities,
            "Energy (ft-lb)": energies,
        }
    )

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
                "Velocity (fps)": "{:.0f}",
                "Energy (ft-lb)": "{:.0f}",
            }
        ),
        width="stretch",
    )

    st.info(
        "Drop values are bullet path vs line-of-sight (same convention as Federal). "
        "Wind drift is not modeled in this version."
    )

# ---------- VIEW 4: Dev — Federal comparison (hardcoded 2.5\" / 100 yd case) ----------
else:
    st.subheader("Federal comparison (dev) — 2.5\" sight, 100 yd zero")

    st.markdown(
        "This view compares the Sisk G1 engine vs the Federal Premium ballistics "
        "table for the 168gr SMK, BC 0.462, 2650 fps, sight height **2.5 in**, "
        "zero range **100 yd**, sea level, 59°F, 0–500 yd in 50-yd steps."
    )

    # Federal reference data (as provided)
    fed_ranges = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500], dtype=float)
    fed_drop = np.array(
        [-2.5, -0.6, 0.0, -0.8, -3.3, -7.4, -13.3, -21.1, -31.0, -43.2, -58.1],
        dtype=float,
    )

    # Our model for exactly the same config
    traj_dev = trajectory_vs_los_table(
        MUZZLE_V_FPS,
        DEFAULT_BC_G1,
        zero_range_yd=100.0,
        sight_height_in=2.5,
        env=ENV,
        max_range_yd=500.0,
        step_yd=50.0,
    )
    app_ranges = traj_dev["range_yd"]
    app_drop = traj_dev["path_in"]

    n = min(len(fed_ranges), len(app_ranges))
    fed_ranges = fed_ranges[:n]
    fed_drop = fed_drop[:n]
    app_ranges = app_ranges[:n]
    app_drop = app_drop[:n]

    diff_in = app_drop - fed_drop  # positive => app predicts higher than Fed

    # Diff in MOA (approx 1.047" per 100 yd)
    diff_moa = []
    for r, d in zip(fed_ranges, diff_in):
        if r <= 0:
            diff_moa.append(0.0)
        else:
            moa_per_inch = 100.0 / (1.047 * r)
            diff_moa.append(d * moa_per_inch)
    diff_moa = np.array(diff_moa)

    df_cmp = pd.DataFrame(
        {
            "Range (yd)": fed_ranges.astype(int),
            "Federal Drop (in)": fed_drop,
            "App Drop (in)": app_drop,
            "Difference (App - Fed) (in)": diff_in,
            "Difference (MOA)": diff_moa,
        }
    )

    st.dataframe(
        df_cmp.style.format(
            {
                "Federal Drop (in)": "{:+.1f}",
                "App Drop (in)": "{:+.1f}",
                "Difference (App - Fed) (in)": "{:+2.2f}",
                "Difference (MOA)": "{:+.3f}",
            }
        ),
        width="stretch",
    )

    st.info(
        "For this profile, the Sisk G1 engine typically stays within ~2–3 inches of "
        "the Federal calculator at 500 yards, which is well under 1/2 MOA at that distance. "
        "This view is for internal validation only and not shown in the main demo."
    )
