"""
external_ballistics_g1.py
G1 drag-based ballistics model for Sisk Ballistics app.
Muzzle velocity & BC defaults match Sisk setup:
 - v0 = 2650 fps
 - BC (G1) = 0.462 (168gr SMK)

NOTE / CALIBRATION:
This engine uses a simplified G1 drag fit. To align it with the
Federal Premium ballistics calculator (same MV, BC, zero, sight
height, standard atmosphere), we apply a global drag scale.

DRAG_SCALE was tuned so that:
 - v0 = 2650 fps
 - BC (G1) = 0.462
 - Sight height = 3.0"
 - Zero = 200 yd
 - Atmosphere ~ sea level, 59°F

gives a close match to the Federal 0–1000 yd drop/velocity table
(50-yd steps). Small residual differences will still exist, but
are on the order of a few inches at 1000 yd for this profile.
"""

import math
import numpy as np
from dataclasses import dataclass

# -----------------------------
# DEFAULTS
# -----------------------------
DEFAULT_V0_FPS = 2650.0       # locked muzzle velocity
DEFAULT_BC_G1  = 0.462        # 168gr Sierra MatchKing, G1 approx

# -----------------------------
# ENVIRONMENT PARAMETERS
# -----------------------------
@dataclass
class Environment:
    air_density_slug_ft3: float = 0.0023769   # sea level, 59°F
    g_ft_s2: float = 32.174

ENV = Environment()

# -----------------------------
# DRAG CALIBRATION
# -----------------------------
# Global factor applied to the simplified G1 drag term.
# Tuned against the Federal 3.0" / 200 yd table for this load.
DRAG_SCALE = 0.495

# -----------------------------
# G1 DRAG MODEL (APPROXIMATE)
# -----------------------------
def g1_drag_coefficient(v_fps: float) -> float:
    """
    Approximate Cd(v) for the G1 standard projectile.
    v in fps.
    """
    v = v_fps

    if v > 4230: return 0.2629
    elif v > 3680: return 0.2556
    elif v > 3450: return 0.2509
    elif v > 3295: return 0.2455
    elif v > 3130: return 0.2420
    elif v > 2960: return 0.2388
    elif v > 2830: return 0.2356
    elif v > 2680: return 0.2320
    elif v > 2460: return 0.2250
    elif v > 2225: return 0.2180
    elif v > 2015: return 0.2100
    elif v > 1890: return 0.2050
    elif v > 1810: return 0.2000
    elif v > 1730: return 0.1950
    elif v > 1595: return 0.1900
    elif v > 1520: return 0.1850
    elif v > 1420: return 0.1800
    elif v > 1360: return 0.1750
    elif v > 1315: return 0.1700
    elif v > 1280: return 0.1650
    elif v > 1220: return 0.1600
    elif v > 1185: return 0.1550
    elif v > 1150: return 0.1500
    elif v > 1100: return 0.1450
    elif v > 1060: return 0.1400
    elif v > 980:  return 0.1350
    elif v > 900:  return 0.1300
    elif v > 820:  return 0.1250
    elif v > 750:  return 0.1200
    elif v > 700:  return 0.1150
    elif v > 640:  return 0.1100
    elif v > 600:  return 0.1050
    elif v > 550:  return 0.1000
    elif v > 500:  return 0.0950
    elif v > 450:  return 0.0900
    elif v > 420:  return 0.0850
    elif v > 380:  return 0.0800
    elif v > 350:  return 0.0750
    elif v > 300:  return 0.0700
    elif v > 250:  return 0.0650
    else:          return 0.0600


# ---------------------------------
# CORE INTEGRATOR
# ---------------------------------
def _integrate_trajectory_to_range(
    v0_fps: float,
    theta_rad: float,
    bc_g1: float,
    env: Environment,
    target_range_yd: float,
    dt: float = 0.001,
):
    """
    Integrate the trajectory until x reaches target_range_yd (in yards).
    Returns (x_ft, y_ft, v_fps, t_sec).
    x,y in feet, v in fps, t in seconds.
    """
    g   = env.g_ft_s2
    rho = env.air_density_slug_ft3

    x = 0.0
    y = 0.0
    v = v0_fps
    t = 0.0

    vx = v * math.cos(theta_rad)
    vy = v * math.sin(theta_rad)

    target_x_ft = target_range_yd * 3.0

    while x < target_x_ft and v > 200:  # stop if very slow or past range
        # Drag
        Cd   = g1_drag_coefficient(v)
        # Scaled drag force per unit mass, further scaled by 1/BC
        drag = DRAG_SCALE * 0.5 * rho * Cd * (v ** 2) / max(bc_g1, 1e-6)

        ax = -drag * (vx / (v + 1e-9))
        ay = -g    - drag * (vy / (v + 1e-9))

        vx += ax * dt
        vy += ay * dt
        x  += vx * dt
        y  += vy * dt
        v   = math.hypot(vx, vy)
        t  += dt

        if x >= target_x_ft:
            break

    return x, y, v, t


# ---------------------------------
# SOLVE BORE ANGLE FOR ZERO RANGE
# ---------------------------------
def solve_bore_angle(
    v0_fps: float,
    bc_g1: float,
    zero_range_yd: float,
    sight_height_in: float,
    env: Environment,
) -> float:
    """
    Find bore angle theta such that the bullet intersects LOS at zero_range_yd.
    LOS is sight_height_in above bore at the muzzle and is straight.
    """
    zr_ft       = zero_range_yd * 3.0
    target_y_in = sight_height_in

    # Search between -1° and +3°
    low  = math.radians(-1.0)
    high = math.radians(3.0)

    for _ in range(40):
        mid = 0.5 * (low + high)
        x_ft, y_ft, _, _ = _integrate_trajectory_to_range(
            v0_fps, mid, bc_g1, env, zero_range_yd
        )
        y_in = y_ft * 12.0

        # Compare bullet height to LOS height at zero range
        if y_in > target_y_in:
            high = mid
        else:
            low = mid

    return 0.5 * (low + high)


# ---------------------------------
# IMPACT AT RANGE WITH CANT ROTATION
# ---------------------------------
def impact_at_range_with_cant(
    v0_fps: float,
    bc_g1: float,
    theta_bore_rad: float,
    range_yd: float,
    sight_height_in: float,
    cant_deg: float,
    env: Environment,
):
    """
    Compute impact at a given range using G1 model and rotate by cant.
    Returns: (h_in, v_in, t_sec, y_rel_LOS_in)
    """
    x_ft, y_ft, v_fps, t_sec = _integrate_trajectory_to_range(
        v0_fps, theta_bore_rad, bc_g1, env, range_yd
    )

    y_in         = y_ft * 12.0
    y_rel_LOS_in = y_in - sight_height_in  # +up, -down vs LOS

    cant = math.radians(cant_deg)
    x0, y0 = 0.0, y_rel_LOS_in

    h_in = -y0 * math.sin(cant)   # right(+) due to roll
    v_in =  y0 * math.cos(cant)   # up(+)

    return h_in, v_in, t_sec, y_rel_LOS_in


# ---------------------------------
# TRAJECTORY TABLE VS LOS (FOR GRAPH)
# ---------------------------------
def trajectory_vs_los_table(
    v0_fps: float,
    bc_g1: float,
    zero_range_yd: float,
    sight_height_in: float,
    env: Environment,
    max_range_yd: float,
    step_yd: float = 25.0,
):
    """
    Generate a trajectory table from 0 to max_range_yd in steps of step_yd.
    All vertical values are relative to LOS (inches), matching Federal-style curves.
    Returns dict with keys: 'range_yd', 'path_in', 'tof_s', 'vel_fps'
    """
    theta_bore = solve_bore_angle(
        v0_fps, bc_g1, zero_range_yd, sight_height_in, env
    )

    # Targets in feet
    ranges_yd = np.arange(0.0, max_range_yd + 1e-6, step_yd)
    targets_ft = ranges_yd * 3.0

    g   = env.g_ft_s2
    rho = env.air_density_slug_ft3

    x = 0.0
    y = 0.0
    v = v0_fps
    t = 0.0
    vx = v * math.cos(theta_bore)
    vy = v * math.sin(theta_bore)

    dt = 0.001

    path_in_list = []
    tof_list     = []
    vel_list     = []

    idx = 0
    last_target_ft = targets_ft[-1]

    while x <= last_target_ft + 1.0 and v > 200 and idx < len(targets_ft):
        Cd   = g1_drag_coefficient(v)
        drag = DRAG_SCALE * 0.5 * rho * Cd * (v ** 2) / max(bc_g1, 1e-6)

        ax = -drag * (vx / (v + 1e-9))
        ay = -g    - drag * (vy / (v + 1e-9))

        vx += ax * dt
        vy += ay * dt
        x  += vx * dt
        y  += vy * dt
        v   = math.hypot(vx, vy)
        t  += dt

        # Catch up to all sample points we've passed
        while idx < len(targets_ft) and x >= targets_ft[idx]:
            y_in  = y * 12.0
            y_rel = y_in - sight_height_in
            path_in_list.append(y_rel)
            tof_list.append(t)
            vel_list.append(v)
            idx += 1

    # If we stopped early, pad remaining with last known values
    while idx < len(targets_ft):
        if path_in_list:
            path_in_list.append(path_in_list[-1])
            tof_list.append(tof_list[-1])
            vel_list.append(vel_list[-1])
        else:
            path_in_list.append(0.0)
            tof_list.append(0.0)
            vel_list.append(v0_fps)
        idx += 1

    return {
        "range_yd":       ranges_yd,
        "path_in":        np.array(path_in_list),
        "tof_s":          np.array(tof_list),
        "vel_fps":        np.array(vel_list),
        "theta_bore_rad": theta_bore,
    }
