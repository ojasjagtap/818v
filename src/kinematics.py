"""Smoothing, kinematic-bicycle and point-mass fits, and per-window features.

Every function takes a batch of windows: arrays of shape (N, T) of positions x, y [m] and
heading h [rad] sampled every DT seconds. Nothing here sees the class label, so both models
are fit to every track and the comparison between them is itself a feature.

Kinematic bicycle model (rear-axle reference): x' = v cos(psi), y' = v sin(psi),
psi' = v tan(delta) / L. The tracked point is the box center, a distance d (~L/2) ahead of the
rear axle, so in a turn it slides sideways: v_lat = d * psi'. That relation is the only place
a length enters the kinematics; path curvature alone gives only the bound L <= tan(delta_max)/kappa.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.signal import detrend, savgol_filter

from common import DT

SG_WIN, SG_POLY = 11, 3            # Savitzky-Golay smoothing: 1.1 s cubic fit
MOVE_SPEED = 1.0                   # m/s; slower samples carry no reliable direction
TURN_SPEED, TURN_YAW = 2.0, 0.15   # m/s, rad/s; samples that inform the lever arm
TURN_MIN = 8                       # minimum turning samples for a lever-arm estimate
DELTA_MAX = np.radians(35.0)       # steering limit assumed by the turning-radius bound
KAPPA_FLOOR = 1e-3                 # 1/m; caps the turning-radius bound at ~700 m for straight paths
ACC_FLOOR = 0.05                   # m/s^2; denominator floor for the acceleration ratio
GAIT_BAND = (0.8, 2.6)             # Hz; walking steps and pedalling cadence


def _sg(a, deriv=0, dt=DT):
    return savgol_filter(a, SG_WIN, SG_POLY, deriv=deriv, delta=dt, axis=-1)


def _quiet(fn, *args, **kw):
    """Row-wise nan-reductions return NaN for all-NaN rows; silence the warning they emit."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(*args, **kw)


def _pct(a, q):
    return _quiet(np.nanpercentile, a, q, axis=-1)


def _rms(a):
    return np.sqrt(_quiet(np.nanmean, a ** 2, axis=-1))


def _frac(cond, valid):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (cond & valid).sum(-1) / valid.sum(-1)


def smooth(x, y, h, dt=DT):
    """World-frame velocity, acceleration and jerk from positions; smoothed heading and yaw rate."""
    hu = np.unwrap(h, axis=-1)
    return dict(vx=_sg(x, 1, dt), vy=_sg(y, 1, dt), ax=_sg(x, 2, dt), ay=_sg(y, 2, dt),
                jx=_sg(x, 3, dt), jy=_sg(y, 3, dt), h=_sg(hu, 0, dt), yaw=_sg(hu, 1, dt))


def body_frame(s, tau=0.0):
    """Velocity and acceleration resolved along (lon) and across (lat) the heading.

    tau [s] is how far the heading lags the positions; shifting the heading forward by
    tau * yaw_rate undoes it to first order.
    """
    h = s["h"] + tau * s["yaw"]
    c, n = np.cos(h), np.sin(h)
    return dict(h=h,
                v_lon=s["vx"] * c + s["vy"] * n, v_lat=-s["vx"] * n + s["vy"] * c,
                a_lon=s["ax"] * c + s["ay"] * n, a_lat=-s["ax"] * n + s["ay"] * c)


def turning(speed, yaw):
    return (speed > TURN_SPEED) & (np.abs(yaw) > TURN_YAW)


def lever_arm(v_lat, yaw, mask):
    """Least-squares d in v_lat = d * yaw_rate (no intercept) over masked samples.

    Returns d [m], its standard error, and the sample count; d is NaN with < TURN_MIN samples.
    """
    m = mask.astype(float)
    n = m.sum(-1)
    syy = (yaw ** 2 * m).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (v_lat * yaw * m).sum(-1) / syy
        r = (v_lat - d[..., None] * yaw) * m
        se = np.sqrt((r ** 2).sum(-1) / np.maximum(n - 1, 1) / syy)
    few = n < TURN_MIN
    d[few], se[few] = np.nan, np.nan
    return d, se, n


def estimate_heading_lag(x, y, h, dt=DT):
    """One global lag tau [s] of heading behind position.

    Pooled within-window regression v_lat = d_i * yaw + tau * v_lon * yaw over turning samples.
    Each window's lever arm d_i is a nuisance fixed effect, so tau is identified only by how the
    sideslip grows with speed: a rigid lever arm does not scale with speed, a heading lag does.
    Returns (tau, number of windows that contributed).
    """
    s = smooth(x, y, h, dt)
    b = body_frame(s)
    m = turning(np.hypot(s["vx"], s["vy"]), s["yaw"])
    ok = m.sum(-1) >= TURN_MIN
    m = (m & ok[:, None]).astype(float)
    yaw, u, w = s["yaw"], b["v_lon"] * s["yaw"], b["v_lat"]
    syy = np.where(ok, (yaw ** 2 * m).sum(-1), 1.0)[:, None]
    u_r = (u - (u * yaw * m).sum(-1)[:, None] / syy * yaw) * m
    w_r = (w - (w * yaw * m).sum(-1)[:, None] / syy * yaw) * m
    return float((u_r * w_r).sum() / (u_r ** 2).sum()), int(ok.sum())


def _band_power(sig, dt=DT, band=GAIT_BAND):
    """Fraction of detrended, Hann-windowed spectral power inside `band`, and the peak frequency."""
    t = sig.shape[-1]
    z = detrend(sig, axis=-1) * np.hanning(t)
    p = np.abs(np.fft.rfft(z, axis=-1))[..., 1:] ** 2
    f = np.fft.rfftfreq(t, dt)[1:]
    inband = (f >= band[0]) & (f <= band[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = p[..., inband].sum(-1) / p.sum(-1)
    return frac, f[np.argmax(p, axis=-1)]


def physics_features(x, y, h, tau=0.0, dt=DT):
    """Physical features of each window. Deliberately no speed-level statistics."""
    s = smooth(x, y, h, dt)
    b = body_frame(s, tau)
    speed = np.hypot(s["vx"], s["vy"])
    yaw = s["yaw"]
    moving = speed > MOVE_SPEED
    f = {}

    # Implied wheelbase. (a) Sideslip of the box center vs yaw rate: lever arm d ~ L/2.
    d, d_se, _ = lever_arm(b["v_lat"], yaw, turning(speed, yaw))
    f["wheelbase_slip"] = 2 * d
    f["wheelbase_slip_se"] = 2 * d_se
    # (b) Tightest turn: rear-axle curvature kappa = yaw / v_lon and L <= tan(delta_max) / kappa.
    with np.errstate(invalid="ignore", divide="ignore"):
        kappa = np.where(b["v_lon"] > MOVE_SPEED, np.abs(yaw) / b["v_lon"], np.nan)
    f["wheelbase_turn_bound"] = np.tan(DELTA_MAX) / np.maximum(_pct(kappa, 95), KAPPA_FLOOR)
    # (c) Lateral-acceleration envelope (curvature x speed^2 = v_lon * yaw).
    f["lat_acc_p95"] = _pct(np.where(moving, np.abs(b["v_lon"] * yaw), np.nan), 95)

    # Sideways motion relative to heading.
    beta = np.where(moving, np.abs(np.arctan2(b["v_lat"], b["v_lon"])), np.nan)
    f["slip_p50"] = _pct(beta, 50)
    f["slip_p95"] = _pct(beta, 95)
    f["slip_frac_30deg"] = _frac(beta > np.radians(30), moving)
    f["reverse_frac"] = _frac(b["v_lon"] < 0, moving)

    # Bicycle vs free point mass, scored by one-step-ahead prediction of the next displacement.
    # Point mass: v(t) dt. Bicycle: speed along the heading plus the lever-arm slide d * yaw.
    step = np.stack([np.diff(x, axis=-1), np.diff(y, axis=-1)])
    v_point = np.stack([s["vx"], s["vy"]])[..., :-1]
    hc = b["h"][..., :-1]
    lon = b["v_lon"][..., :-1]
    lat = np.nan_to_num(d)[:, None] * yaw[..., :-1]
    v_bike = np.stack([lon * np.cos(hc) - lat * np.sin(hc), lon * np.sin(hc) + lat * np.cos(hc)])
    mv = moving[..., :-1]
    err_point = np.sqrt(np.where(mv, ((step - v_point * dt) ** 2).sum(0), np.nan))
    err_bike = np.sqrt(np.where(mv, ((step - v_bike * dt) ** 2).sum(0), np.nan))
    f["pred_rmse_point"] = _rms(err_point)
    f["pred_rmse_bicycle"] = _rms(err_bike)
    with np.errstate(invalid="ignore", divide="ignore"):
        f["pred_rmse_ratio"] = f["pred_rmse_bicycle"] / f["pred_rmse_point"]
    # Lateral acceleration the bicycle cannot explain (centripetal v_lon * yaw removed), relative
    # to longitudinal acceleration: near 0 for non-holonomic movers, larger for omnidirectional ones.
    a_res = np.where(moving, b["a_lat"] - b["v_lon"] * yaw, np.nan)
    a_lon = np.where(moving, b["a_lon"], np.nan)
    f["acc_lat_resid_ratio"] = _rms(a_res) / np.maximum(_rms(a_lon), ACC_FLOOR)

    # Acceleration along the heading, and jerk.
    for q in (5, 50, 95):
        f[f"acc_lon_p{q}"] = np.percentile(b["a_lon"], q, axis=-1)
    jerk = np.hypot(s["jx"], s["jy"])
    f["jerk_rms"] = np.sqrt((jerk ** 2).mean(-1))
    f["jerk_p95"] = np.percentile(jerk, 95, axis=-1)

    # Periodic motion (gait, pedalling).
    for name, sig in (("speed", speed), ("vlat", b["v_lat"])):
        f[f"gait_{name}_frac"], f[f"gait_{name}_peak"] = _band_power(sig, dt)
    return pd.DataFrame(f)


def baseline_features(x, y, dt=DT):
    """Speed and tangential-acceleration summary statistics: the gradient-boosted-tree baseline."""
    vx, vy, ax, ay = _sg(x, 1, dt), _sg(y, 1, dt), _sg(x, 2, dt), _sg(y, 2, dt)
    speed = np.hypot(vx, vy)
    acc = (vx * ax + vy * ay) / np.maximum(speed, 0.1)
    f = {"speed_mean": speed.mean(-1), "speed_std": speed.std(-1), "speed_max": speed.max(-1)}
    for q in (5, 50, 95):
        f[f"speed_p{q}"] = np.percentile(speed, q, axis=-1)
    f.update(acc_mean=acc.mean(-1), acc_std=acc.std(-1), acc_absmax=np.abs(acc).max(-1),
             acc_p5=np.percentile(acc, 5, axis=-1), acc_p95=np.percentile(acc, 95, axis=-1))
    return pd.DataFrame(f)


def all_features(xyh, tau, chunk=100_000):
    """Physics + baseline features for an (N, T, 3) array of [x, y, heading] windows, in chunks."""
    parts = []
    for i in range(0, len(xyh), chunk):
        w = xyh[i:i + chunk].astype(np.float64)
        x, y, h = w[..., 0], w[..., 1], w[..., 2]
        parts.append(pd.concat([physics_features(x, y, h, tau), baseline_features(x, y)], axis=1))
    return pd.concat(parts, ignore_index=True)


def smooth_windows(xyh, dt=DT):
    """Savitzky-Golay-smoothed copy of (N, T, 3) [x, y, h] windows: the same low-pass the physics
    features use, so a sequence model fed these cannot read high-frequency tracker jitter."""
    h = np.unwrap(xyh[..., 2], axis=-1)
    return np.stack([_sg(xyh[..., 0], 0, dt), _sg(xyh[..., 1], 0, dt), _sg(h, 0, dt)], -1).astype(np.float32)


def agent_frame(xyh, dt=DT):
    """Raw windows for the sequence model: start at the origin, initial heading along +x.

    (N, T, 3) [x, y, h] -> (N, T, 6) float32 [x, y, vx, vy, cos(h - h0), sin(h - h0)], where
    (vx, vy) is the per-step displacement / dt (first step repeated). Only differencing; nothing
    else is engineered.
    """
    x = xyh[..., 0] - xyh[:, :1, 0]
    y = xyh[..., 1] - xyh[:, :1, 1]
    dh = xyh[..., 2] - xyh[:, :1, 2]
    c, s = np.cos(xyh[:, :1, 2]), np.sin(xyh[:, :1, 2])
    xr, yr = c * x + s * y, -s * x + c * y
    vx, vy = np.diff(xr, axis=1) / dt, np.diff(yr, axis=1) / dt
    vx, vy = np.concatenate([vx[:, :1], vx], 1), np.concatenate([vy[:, :1], vy], 1)
    return np.stack([xr, yr, vx, vy, np.cos(dh), np.sin(dh)], -1).astype(np.float32)
