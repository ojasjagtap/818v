"""M1: data sanity, heading lag, wheelbase identifiability gate, RQ2 feasibility (laptop).

    python src/m1_sanity.py            # after scripts/fetch_m1_sample.py
writes results/m1/report.md and the figures it references.
"""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from common import CLASSES, CITIES, DATA, DT, MIN_PATH_M, RESULTS, SENSOR_TO_CLASS, WINDOW  # noqa: E402
from extract import COLS, scenario_paths, windows_from_frame  # noqa: E402
from kinematics import (_band_power, all_features, body_frame, estimate_heading_lag,  # noqa: E402
                        lever_arm, smooth, turning)
from sensor_tracks import log_tracks, longest_run, windows_from_log  # noqa: E402

OUT = RESULTS / "m1"
CATEGORY = {0: "fragment", 1: "unscored", 2: "scored", 3: "focal"}
WHEELED4 = ["REGULAR_VEHICLE", "LARGE_VEHICLE", "BOX_TRUCK", "TRUCK", "TRUCK_CAB",
            "BUS", "SCHOOL_BUS", "ARTICULATED_BUS"]
BUSES = ["BUS", "SCHOOL_BUS", "ARTICULATED_BUS"]
GATE_SPEARMAN, GATE_AUROC = 0.4, 0.7
rng = np.random.default_rng(0)
report = []


def add(*parts):
    report.extend(parts)
    report.append("")


def md(df, fmt="{:.3g}"):
    """DataFrame -> markdown table (keeps the index when it is named)."""
    if df.index.name is not None or isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    head = [str(c) for c in df.columns]
    rows = [[fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
            for r in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                     + ["| " + " | ".join(r) + " |" for r in rows])


def boot_ci(stat, a, b, n=1000):
    """95% bootstrap interval of stat(a, b) over resampled pairs."""
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        try:
            vals.append(stat(a[i], b[i]))
        except ValueError:      # a resample with a single class
            pass
    return np.nanpercentile(vals, [2.5, 97.5])


def auroc(y, s):
    return roc_auc_score(y, s)


def spearman(a, b):
    return spearmanr(a, b)[0]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def best_lag(pairs, max_k=8):
    """Steps k maximising corr(a[t + k], b[t]) pooled over (a, b, mask) series; k > 0: a lags b."""
    best = (None, -np.inf)
    for k in range(-max_k, max_k + 1):
        aa, bb = [], []
        for a, b, m in pairs:
            n = len(a)
            if k >= 0:
                a2, b2, mm = a[k:], b[:n - k], m[k:] & m[:n - k]
            else:
                a2, b2, mm = a[:n + k], b[-k:], m[:n + k] & m[-k:]
            aa.append(a2[mm])
            bb.append(b2[mm])
        r = np.corrcoef(np.concatenate(aa), np.concatenate(bb))[0, 1]
        if r > best[1]:
            best = (k, r)
    return best[0]


def track_lever_arm(x, y, h, tau):
    """Lever-arm wheelbase (2d) over a whole variable-length track."""
    s = smooth(x[None], y[None], h[None])
    b = body_frame(s, tau)
    d, _, _ = lever_arm(b["v_lat"], s["yaw"], turning(np.hypot(s["vx"], s["vy"]), s["yaw"]))
    return 2 * d[0]


# ----------------------------------------------------------------------------------------------
def load_mf():
    frames, with_av = [], 0
    paths = scenario_paths(DATA / "mf" / "val")
    for p in paths:
        df = pd.read_parquet(p, columns=COLS + ["velocity_x", "velocity_y"])
        with_av += bool((df.track_id == "AV").any())
        frames.append(df[df.object_type.isin(CLASSES) & (df.track_id != "AV")])
    raw = pd.concat(frames, ignore_index=True).sort_values(["scenario_id", "track_id", "timestep"])
    return raw.reset_index(drop=True), len(paths), with_av


def mf_counts(raw, meta):
    g = raw.groupby(["scenario_id", "track_id"], sort=False)
    trk = g.agg(cls=("object_type", "first"), cat=("object_category", "first"), n=("timestep", "size"))
    step = np.hypot(g.position_x.diff(), g.position_y.diff())
    trk["path"] = step.groupby([raw.scenario_id, raw.track_id]).sum()
    t1 = pd.crosstab(trk.cls, trk.cat.map(CATEGORY)).reindex(CLASSES)
    t1.index.name = "class"
    rows = []
    for secs in (3, 5, 8):
        ok = trk[(trk.n >= secs * 10) & (trk.path >= MIN_PATH_M)]
        for c in CLASSES:
            s = ok[ok.cls == c]
            rows.append(dict(min_len=f"{secs}s", cls=c, tracks=len(s),
                             fragment_share=(s.cat == 0).mean() if len(s) else np.nan))
    t2 = pd.DataFrame(rows).pivot(index="cls", columns="min_len", values=["tracks", "fragment_share"])
    t2.columns = [f"{a}@{b}" for a, b in t2.columns]
    t2 = t2.reindex(CLASSES)
    t2.index.name = "class"
    t3 = pd.crosstab(meta.city, meta.cls).reindex(index=CITIES, columns=CLASSES).fillna(0).astype(int)
    t3.index.name = "city"
    return t1, t2, t3


def mf_consistency(raw):
    """How the provided velocity relates to heading and to the position derivative, per class."""
    rows = []
    for cls in CLASSES:
        sub = raw[raw.object_type == cls]
        dh, dv, pairs = [], [], []
        for _, t in sub.groupby(["scenario_id", "track_id"], sort=False):
            if len(t) < 30:
                continue
            x, y = t.position_x.to_numpy(), t.position_y.to_numpy()
            vx, vy, h = t.velocity_x.to_numpy(), t.velocity_y.to_numpy(), t.heading.to_numpy()
            sx = savgol_filter(x, 11, 3, deriv=1, delta=DT)
            sy = savgol_filter(y, 11, 3, deriv=1, delta=DT)
            sp = np.hypot(sx, sy)
            mv = sp > 2
            if mv.sum() < 10:
                continue
            dh.append(np.abs(np.degrees(wrap(h - np.arctan2(vy, vx))))[mv])
            dv.append(np.hypot(vx - sx, vy - sy)[mv])
            pairs.append((np.hypot(vx, vy), sp, mv))
        if not dh:
            continue
        rows.append(dict(cls=cls, tracks=len(dh),
                         heading_vs_provided_vel_deg=np.median(np.concatenate(dh)),
                         provided_vs_position_vel_mps=np.median(np.concatenate(dv)),
                         provided_speed_lag_s=best_lag(pairs) * DT))
    return pd.DataFrame(rows).set_index("cls")


# ----------------------------------------------------------------------------------------------
def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    add("# M1 report: data sanity, heading lag, wheelbase gate, RQ2 feasibility",
        f"Generated by `src/m1_sanity.py` on {time.strftime('%Y-%m-%d %H:%M')}. Numbers below are "
        "computed by the same code the cluster pipeline uses (`extract.py`, `kinematics.py`, "
        "`sensor_tracks.py`, `sensor_crops.py`).")

    # ---- MF sample ------------------------------------------------------------------------
    raw, n_scen, with_av = load_mf()
    meta, xyh = windows_from_frame(raw)
    x, y, h = (xyh[..., i].astype(float) for i in range(3))
    add("## 1. MF sample",
        f"{n_scen} val scenarios; the ego track `AV` appears in {with_av} of them (excluded). "
        f"{raw.groupby(['scenario_id', 'track_id']).ngroups} tracks of the five dynamic classes; "
        f"{len(meta)} moving 5-s windows (one per track with >= {WINDOW} steps and >= {MIN_PATH_M:g} m "
        "of path inside the window).")
    t1, t2, t3 = mf_counts(raw, meta)
    add("**Tracks by class and category** (all lengths):", md(t1),
        "**Tracks long enough for a window and moving**, by minimum length (fragment share = "
        "fraction that are TRACK_FRAGMENTs):", md(t2, "{:.2f}"),
        "**Moving 5-s windows by city** (feeds the held-out-city choice):", md(t3))

    cons = mf_consistency(raw)
    add("## 2. Provided velocity vs heading vs positions (MF, tracks >= 3 s, samples > 2 m/s)",
        md(cons, "{:.3g}"),
        "If the provided velocity were the box-center velocity, it would differ from the heading "
        "by the sideslip during turns and match the position derivative. A positive lag means the "
        "provided speed trails the position-derived speed.")

    # ---- heading lag -----------------------------------------------------------------------
    tau_mf, n_tau = estimate_heading_lag(x, y, h)
    per_class = {c: estimate_heading_lag(x[meta.cls == c], y[meta.cls == c], h[meta.cls == c])
                 for c in CLASSES}
    tau_rows = pd.DataFrame([dict(source="MF, all classes", tau_s=tau_mf, windows=n_tau)]
                            + [dict(source=f"MF, {c}", tau_s=t, windows=n) for c, (t, n) in per_class.items()])

    # ---- features on MF windows (pipeline setting: no lag correction; see section 3) ---------
    feats = all_features(xyh, 0.0)
    feats_tau = all_features(xyh, tau_mf)

    # ---- Sensor ------------------------------------------------------------------------------
    s_meta, s_xyh, full_rows, dropped, n_ann, s_logs = [], [], [], 0, 0, {}
    for split in ("train", "val"):
        logs = sorted(p for p in (DATA / "sensor" / split).iterdir() if (p / "annotations.feather").exists())
        s_logs[split] = len(logs)
        for log in logs:
            tracks, n_drop = log_tracks(log)
            dropped += n_drop
            n_ann += len(tracks) + n_drop
            m, w = windows_from_log(tracks, log.name)
            m["split"] = split
            s_meta.append(m)
            s_xyh.append(w)
            for tid, g in tracks[tracks.category.isin(WHEELED4)].groupby("track_uuid", sort=False):
                i0, i1 = longest_run(g.timestamp_ns.to_numpy())
                if i1 - i0 < WINDOW:
                    continue
                a = g[["x", "y", "h"]].to_numpy()[i0:i1]
                full_rows.append(dict(category=g.category.iat[0], length_m=g.length_m.median(),
                                      xyh=a - [a[0, 0], a[0, 1], 0]))
    s_meta = pd.concat(s_meta, ignore_index=True)
    s_xyh = np.concatenate(s_xyh)
    sx, sy, sh = (s_xyh[..., i].astype(float) for i in range(3))
    tau_s, n_tau_s = estimate_heading_lag(sx, sy, sh)
    tau_rows.loc[len(tau_rows)] = dict(source="Sensor (human-annotated yaw), all classes",
                                       tau_s=tau_s, windows=n_tau_s)
    add("## 3. Heading lag (diagnostic)",
        "Pooled within-window fit `v_lat = d_i * yaw + tau * v_lon * yaw` over turning samples "
        "(per-window lever arms as fixed effects). A lag of the heading behind the positions "
        "produces sideslip that grows with speed; a rigid lever arm does not.",
        md(tau_rows, "{:.3f}"),
        "Within windows the MF heading shows no meaningful lag. A sizeable lag in human-annotated "
        "Sensor yaw would be implausible, and within one window `v_lon * yaw` is nearly collinear "
        "with `yaw`, so this estimator is fragile. **The pipeline therefore applies no lag "
        "correction (tau = 0)**; lag-corrected numbers are shown below only for comparison.")
    s_feats = all_features(s_xyh, 0.0)
    s_feats_tau = all_features(s_xyh, tau_s)

    # ---- wheelbase identifiability -----------------------------------------------------------
    wb = pd.DataFrame({"cls": meta.cls, "wb": feats.wheelbase_slip, "wb_tau": feats_tau.wheelbase_slip})
    cov = wb.groupby("cls").agg(windows=("wb", "size"), coverage=("wb", lambda v: v.notna().mean()),
                                median=("wb", "median"),
                                p25=("wb", lambda v: v.quantile(.25)), p75=("wb", lambda v: v.quantile(.75)),
                                median_lag_corrected=("wb_tau", "median")).reindex(CLASSES)
    cov.index.name = "class"
    bv = wb[wb.cls.isin(["bus", "vehicle"]) & wb.wb.notna()]
    yb, sb = (bv.cls == "bus").to_numpy(), bv.wb.to_numpy()
    auc_mf = auroc(yb, sb)
    auc_mf_ci = boot_ci(auroc, yb, sb)
    bvt = wb[wb.cls.isin(["bus", "vehicle"]) & wb.wb_tau.notna()]
    auc_mf_tau = auroc((bvt.cls == "bus").to_numpy(), bvt.wb_tau.to_numpy())

    # full MF tracks (best case: up to 11 s of turning instead of 5)
    full_mf = []
    for (sid, tid), t in raw.groupby(["scenario_id", "track_id"], sort=False):
        c = t.object_type.iat[0]
        if c in ("bus", "vehicle") and len(t) >= WINDOW:
            full_mf.append((c, track_lever_arm(t.position_x.to_numpy() - t.position_x.iat[0],
                                               t.position_y.to_numpy() - t.position_y.iat[0],
                                               t.heading.to_numpy(), 0.0)))
    full_mf = pd.DataFrame(full_mf, columns=["cls", "wb"]).dropna()
    auc_mf_full = auroc((full_mf.cls == "bus").to_numpy(), full_mf.wb.to_numpy())

    sv = pd.DataFrame({"category": s_meta.category, "L": s_meta.length_m, "wb": s_feats.wheelbase_slip,
                       "wb_tau": s_feats_tau.wheelbase_slip})
    sv4 = sv[sv.category.isin(WHEELED4) & sv.wb.notna()]
    sp_s = spearman(sv4.wb.to_numpy(), sv4.L.to_numpy())
    sp_s_ci = boot_ci(spearman, sv4.wb.to_numpy(), sv4.L.to_numpy())
    sv4t = sv[sv.category.isin(WHEELED4) & sv.wb_tau.notna()]
    sp_s_tau = spearman(sv4t.wb_tau.to_numpy(), sv4t.L.to_numpy())
    sbv = sv4[sv4.category.isin(BUSES + ["REGULAR_VEHICLE"])]
    auc_s = auroc(sbv.category.isin(BUSES).to_numpy(), sbv.wb.to_numpy())
    full_s = pd.DataFrame([dict(category=r["category"], L=r["length_m"],
                                wb=track_lever_arm(r["xyh"][:, 0], r["xyh"][:, 1], r["xyh"][:, 2], 0.0))
                           for r in full_rows]).dropna()
    sp_s_full = spearman(full_s.wb.to_numpy(), full_s.L.to_numpy())
    fb = full_s[full_s.category.isin(BUSES + ["REGULAR_VEHICLE"])]
    auc_s_full = auroc(fb.category.isin(BUSES).to_numpy(), fb.wb.to_numpy())
    by_cat = sv4.groupby("category").agg(windows=("wb", "size"), true_length_m=("L", "median"),
                                         implied_wheelbase_m=("wb", "median"))
    passed = (sp_s >= GATE_SPEARMAN) and (auc_mf >= GATE_AUROC)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].boxplot([wb.wb[wb.cls == c].dropna() for c in CLASSES], showfliers=False)
    ax[0].set_xticks(range(1, 6), CLASSES)
    ax[0].set_ylabel("implied wheelbase 2d [m]")
    ax[0].set_title("MF windows (tracker heading)")
    for cat, sub in sv4.groupby("category"):
        ax[1].scatter(sub.L, sub.wb, s=8, alpha=.6, label=cat)
    ax[1].plot([0, 20], [0, 20 * 0.6], "k--", lw=.8, label="wheelbase = 0.6 x length")
    ax[1].set_xlabel("true box length [m]")
    ax[1].set_ylabel("implied wheelbase 2d [m]")
    ax[1].set_ylim(-10, 20)
    ax[1].set_title(f"Sensor windows (human-annotated yaw): Spearman {sp_s:.2f}")
    ax[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "wheelbase.png", dpi=120)
    plt.close(fig)

    add("## 4. Is the wheelbase identifiable? (the gate)",
        "`wheelbase_slip` = 2 x lever arm from `v_lat = d * yaw`, defined only for windows with "
        ">= 8 turning samples (speed > 2 m/s, |yaw rate| > 0.15 rad/s).",
        "**MF windows by class:**", md(cov, "{:.2f}"),
        f"- MF bus-vs-vehicle AUROC of the implied wheelbase (windows where it is defined, n={len(bv)}, "
        f"buses={int(yb.sum())}): **{auc_mf:.2f}** (95% CI {auc_mf_ci[0]:.2f}-{auc_mf_ci[1]:.2f}); "
        f"with the lag correction {auc_mf_tau:.2f}; on full tracks (up to 11 s, n={len(full_mf)}, "
        f"buses={int((full_mf.cls == 'bus').sum())}) {auc_mf_full:.2f}.",
        f"- Sensor 4+-wheel windows (human-annotated yaw, n={len(sv4)}): Spearman with the true box "
        f"length **{sp_s:.2f}** (95% CI {sp_s_ci[0]:.2f}-{sp_s_ci[1]:.2f}); with the lag correction "
        f"{sp_s_tau:.2f}; full tracks (n={len(full_s)}) {sp_s_full:.2f}. "
        f"Bus-vs-regular-vehicle AUROC: windows {auc_s:.2f} "
        f"(buses={int(sbv.category.isin(BUSES).sum())}), full tracks {auc_s_full:.2f} "
        f"(buses={int(fb.category.isin(BUSES).sum())}).",
        "**Sensor windows by category:**", md(by_cat, "{:.2f}"),
        f"**Gate** (Spearman >= {GATE_SPEARMAN} on Sensor AND bus-vs-vehicle AUROC >= {GATE_AUROC} on "
        f"MF windows): **{'PASS' if passed else 'FAIL'}**. "
        + ("The implied wheelbase can be a headline feature." if passed else
           "The implied wheelbase stays in the feature set as one feature among many; the report "
           "presents this identifiability analysis as a finding, and the physics story rests on "
           "constraint violation and the capability envelope."),
        "![wheelbase](wheelbase.png)")

    # ---- periodicity -----------------------------------------------------------------------
    s = smooth(x, y, h)
    b = body_frame(s, tau_mf)
    speed = np.hypot(s["vx"], s["vy"])
    freqs = np.fft.rfftfreq(WINDOW, DT)[1:]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    rows = []
    for c in CLASSES:
        sel = (meta.cls == c).to_numpy()
        for k, (name, sig) in enumerate((("speed", speed), ("v_lat", b["v_lat"]))):
            z = sig[sel] - sig[sel].mean(1, keepdims=True)
            p = np.abs(np.fft.rfft(z * np.hanning(WINDOW), axis=1))[:, 1:] ** 2
            p = p / p.sum(1, keepdims=True)
            ax[k].plot(freqs, np.nanmean(p, 0), label=c)
            ax[k].set_title(f"mean normalised spectrum of {name}")
            ax[k].set_xlabel("Hz")
        frac, _ = _band_power(b["v_lat"][sel])
        rows.append(dict(cls=c, windows=int(sel.sum()), gait_band_frac_vlat=np.nanmedian(frac),
                         gait_band_frac_speed=np.nanmedian(_band_power(speed[sel])[0])))
    ax[0].axvspan(0.8, 2.6, color="grey", alpha=.15)
    ax[1].axvspan(0.8, 2.6, color="grey", alpha=.15)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "spectra.png", dpi=120)
    plt.close(fig)
    add("## 5. Periodic motion (gait / pedalling band 0.8-2.6 Hz)",
        md(pd.DataFrame(rows).set_index("cls"), "{:.3f}"),
        "If gait survived the upstream smoothing, pedestrians (and cyclists) would show extra "
        "power in the shaded band.", "![spectra](spectra.png)")

    # ---- feature sanity ----------------------------------------------------------------------
    med = feats.groupby(meta.cls.to_numpy()).median().T[CLASSES]
    nan = feats.isna().groupby(meta.cls.to_numpy()).mean().T[CLASSES]
    uni = []
    for col in feats.columns:
        r = dict(feature=col)
        for name, (pos, neg) in {"bus_vs_vehicle": ("bus", "vehicle"),
                                 "cyclist_vs_motorcyclist": ("cyclist", "motorcyclist"),
                                 "pedestrian_vs_rest": ("pedestrian", None)}.items():
            sel = meta.cls.isin([pos, neg]) if neg else meta.cls.notna()
            v = feats.loc[sel.to_numpy(), col]
            ok = v.notna().to_numpy()
            yy = (meta.cls[sel] == pos).to_numpy()[ok]
            a = auroc(yy, v.to_numpy()[ok]) if 0 < yy.sum() < len(yy) else np.nan
            r[name] = max(a, 1 - a)
        uni.append(r)
    uni = pd.DataFrame(uni).set_index("feature")
    med.index.name = nan.index.name = "feature"
    add("## 6. Feature sanity (MF windows)",
        "**Median by class:**", md(med, "{:.3g}"),
        "**Missing (NaN) rate by class:**", md(nan, "{:.2f}"),
        "**Single-feature separability** (direction-agnostic AUROC; 0.5 = useless). Diagnostic "
        "only, not a reported metric:", md(uni, "{:.2f}"))

    # ---- is the bus signal physics or tracker noise? -------------------------------------------
    noisy = ["jerk_rms", "pred_rmse_point", "acc_lon_p5", "acc_lon_p95", "acc_std", "speed_p50"]
    rows = []
    for col in noisy:
        mf_v, mf_b = (feats.loc[(meta.cls == c).to_numpy(), col].median() for c in ("vehicle", "bus"))
        s_v, s_b = (s_feats.loc[(s_meta.cls == c).to_numpy(), col].median() for c in ("vehicle", "bus"))
        rows.append(dict(feature=col, mf_vehicle=mf_v, mf_bus=mf_b, mf_bus_over_vehicle=mf_b / mf_v,
                         sensor_vehicle=s_v, sensor_bus=s_b, sensor_bus_over_vehicle=s_b / s_v))
    add("## 6b. Bus vs vehicle: physics or tracker noise?",
        "Medians of noise-sensitive features in MF (automatic tracker) and Sensor (human-annotated "
        "cuboids). A real bus drives more smoothly than a car; if MF buses look *rougher* while "
        "Sensor buses do not, the MF bus signal is box-center noise that grows with object size.",
        md(pd.DataFrame(rows).set_index("feature"), "{:.3g}"))

    # ---- trajectory gallery ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))
    for ax_, c in zip(axes, CLASSES):
        idx = rng.choice(np.flatnonzero(meta.cls == c), size=min(6, int((meta.cls == c).sum())), replace=False)
        for i in idx:
            w = xyh[i].astype(float)
            ax_.plot(w[:, 0], w[:, 1], lw=1)
            k = slice(0, WINDOW, 10)
            ax_.quiver(w[k, 0], w[k, 1], np.cos(w[k, 2]), np.sin(w[k, 2]), angles="xy", width=.004, alpha=.6)
        ax_.set_title(c)
        ax_.set_aspect("equal", "datalim")
    fig.tight_layout()
    fig.savefig(OUT / "gallery.png", dpi=110)
    plt.close(fig)
    add("## 7. Trajectory gallery", "Six random MF windows per class (arrows = reported heading).",
        "![gallery](gallery.png)")

    # ---- RQ2 feasibility ---------------------------------------------------------------------
    cnt = pd.crosstab(s_meta.cls, s_meta.split).reindex(CLASSES).fillna(0).astype(int)
    cnt.index.name = "class"
    add("## 8. RQ2 feasibility (AV2 Sensor)",
        f"{s_logs['train']} train + {s_logs['val']} val logs; {n_ann} annotations, {dropped} without an "
        "ego pose within 10 ms. **Moving 5-s windows per class and split** (these are the RQ2 "
        "tracks before the camera-visibility filter):", md(cnt))
    overlay_report = overlays()
    add(*overlay_report)

    add("## 9. Defaults",
        f"- Window: {WINDOW / 10:.0f} s; moving = path >= {MIN_PATH_M:g} m. See the length table in "
        "section 1 for what 3 s or 8 s would change.",
        "- Held-out city: pick one with all five classes present in section 1's city table.",
        f"- Heading lag for the cluster pipeline: re-estimated on the full train split by "
        f"`aggregate.py` (this sample: {tau_mf:.3f} s).",
        f"\n_Run time {time.time() - t0:.0f} s._")
    (OUT / "report.md").write_text("\n".join(report))
    print(f"wrote {OUT / 'report.md'}  (tau_mf={tau_mf:.3f}s, tau_sensor={tau_s:.3f}s, "
          f"gate={'PASS' if passed else 'FAIL'}: Spearman={sp_s:.2f}, AUROC={auc_mf:.2f})")


def overlays():
    """Projected 3D boxes on the downloaded val images (checks camera model + motion compensation)."""
    from PIL import Image

    from sensor_crops import LogProjector

    logs = sorted({p.parents[3] for p in (DATA / "sensor" / "val").glob("*/sensors/cameras/*/*.jpg")})
    if not logs:
        return ["No overlay images downloaded; skipped the projection check."]
    shown = sorted(SENSOR_TO_CLASS) + ["BICYCLE", "MOTORCYCLE"]
    colors = {"vehicle": "lime", "bus": "orange", "pedestrian": "cyan", "cyclist": "magenta",
              "motorcyclist": "red", None: "yellow"}
    out = ["**Projection overlays** (green vehicle, orange bus, cyan pedestrian, magenta cyclist, "
           "red motorcyclist, yellow bike under a rider; thick = usable as a crop):"]
    for log in logs:
        proj = LogProjector(log)
        cams = sorted(p.name for p in (log / "sensors/cameras").iterdir())
        fig, axes = plt.subplots(1, len(cams), figsize=(7 * len(cams), 6))
        for ax_, cam in zip(np.atleast_1d(axes), cams):
            imgs = sorted((log / "sensors/cameras" / cam).glob("*.jpg"))
            img_path = imgs[len(imgs) // 2]
            cubs, boxes, ok = proj.boxes(cam, int(img_path.stem), shown)
            ax_.imshow(Image.open(img_path))
            for c, bx, good in zip(cubs, boxes, ok):
                if bx[2] - bx[0] <= 0 or bx[3] - bx[1] <= 0:
                    continue
                ax_.add_patch(plt.Rectangle(bx[:2], bx[2] - bx[0], bx[3] - bx[1], fill=False,
                                            ec=colors[SENSOR_TO_CLASS.get(c.category)], lw=2.5 if good else .8))
            ax_.set_title(f"{log.name[:8]} {cam}: {int(ok.sum())} usable boxes")
            ax_.axis("off")
        fig.tight_layout()
        name = f"overlay_{log.name[:8]}.png"
        fig.savefig(OUT / name, dpi=70)
        plt.close(fig)
        out.append(f"![overlay]({name})")
    return out


if __name__ == "__main__":
    main()
