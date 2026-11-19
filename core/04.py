"""
04.py  -  Unified pipeline, v7

  ENTRY    the optical flow confirms that the plug is MOVING through the gate (centre of the ROI).
  EXIT     the moving plug height at the gate falls below `exit_frac` (80 %) of its maximum.
  VELOCITY optical flow (03b_flow_engine v2) measured between real video frames inside the U-Net plug mask
           (resting layer removed), 9 gates across the ROI.  The entry->exit span is cut into groups of
           `group_frames` frames; length_group = v_group * dt; total length = sum of the groups.
  HEIGHT   plug height is measured from the pipe bottom; initial/final heights come from the first/last frame.
  ANGLE    taken in the `n_angle_frames` frames AFTER the entry-side edge of the ROI has reached ~95 % of the
           maximum plug height (steepest part of the front face).
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import importlib
import math
from collections import deque

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import theilslopes
from scipy.ndimage import median_filter

CFG = {
    # --- motion-event scan (frame differences): event window + direction of travel ---
    "act_width": 160, "lags": (2, 6, 16, 40), "k_sigma": 4.0, "min_pix_thr": 1.0, "col_frac": 0.10,
    "min_event_frames": 6, "event_gap": 12, "hi_k": 8.0, "lo_k": 4.0, "min_active_frac": 0.03,
    "min_run": 3, "gap_close": 3, "min_valid_cols": 0.35, "min_corr": 0.90, "min_span": 0.30,
    # --- entry (optical flow) / exit (height) ---
    "entry_min_disp": 0.5,       # centre gate must move >= 0.5 px between the compared frames ...
    "entry_confirm_pairs": 3,    # ... in this many consecutive frame pairs
    "exit_frac": 0.80,           # exit: moving height < 80 % of its maximum
    # --- optical flow / groups ---
    "flow_n_gates": 9, "flow_step": 2, "flow_k_max": 16, "flow_target_disp": 2.5, "group_frames": 20,
    # --- angle ---
    "n_angle_frames": 10,        # frames used, starting right AFTER the edge reaches angle_hi_frac of max height
    "angle_hi_frac": 0.95,
    "angle_edge_lo": 0.02, "angle_edge_hi": 0.08,   # entry-side columns (fraction of ROI width) that are monitored
    "angle_win_frac": 0.06,      # width of the window used to find the steepest part of the front face
    "angle_side": "entry",       # "entry" (auto from travel direction), "right" or "left"
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def calculate_px_to_mm(config):
    p1 = np.array(config.get('p1', [0, 0]), dtype=float)
    p2 = np.array(config.get('p2', [0, 0]), dtype=float)
    pixel_dist = np.linalg.norm(p1 - p2)
    return config.get('dist_mm', 0) / pixel_dist if pixel_dist != 0 else 0.0


def clamp_roi(roi, shape):
    H, W = shape[:2]
    x, y, w, h = [int(v) for v in roi]
    x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
    return x, y, max(1, min(w, W - x)), max(1, min(h, H - y))


def get_robust_median_height(mask, px_to_mm):
    """Median (over columns) of the vertical extent of the mask - used for the first/last frame heights."""
    m = mask > 127
    cols = m.any(axis=0)
    if not cols.any():
        return 0.0
    rows = np.arange(m.shape[0])[:, None]
    ymax = np.where(m, rows, -1).max(axis=0)
    ymin = np.where(m, rows, m.shape[0]).min(axis=0)
    return float(np.median((ymax - ymin + 1)[cols]) * px_to_mm)


def _runs(b):
    b = np.asarray(b, dtype=bool)
    if not b.any():
        return []
    d = np.diff(np.concatenate(([0], b.astype(np.int8), [0])))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def _close_gaps(b, gap):
    b = np.asarray(b, dtype=bool)
    out = b.copy()
    runs = _runs(b)
    for (_, e0), (s1, _) in zip(runs, runs[1:]):
        if s1 - e0 - 1 <= gap:
            out[e0:s1] = True
    return out


def read_frames(video_path, indices):
    """Read only the requested frame indices (sequential decode, no unreliable seeking)."""
    wanted = sorted(set(int(i) for i in indices if i >= 0))
    out = {}
    if not wanted:
        return out
    cap = cv2.VideoCapture(video_path)
    idx, k = 0, 0
    while idx <= wanted[-1]:
        if idx == wanted[k]:
            ret, f = cap.read()
            if not ret:
                break
            out[idx] = f
            k += 1
        elif not cap.grab():
            break
        idx += 1
    cap.release()
    return out


def read_range(video_path, f0, f1, stride=1):
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while idx < f0:
        if not cap.grab():
            cap.release()
            return
        idx += 1
    while idx <= f1:
        ret, f = cap.read()
        if not ret:
            break
        if (idx - f0) % stride == 0:
            yield idx, f
        idx += 1
    cap.release()


# --------------------------------------------------------------------------- #
# Pass 1: motion-event window and travel direction (cheap frame-difference scan)
# --------------------------------------------------------------------------- #
def build_activity_stack(video_path, roi, width):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None, None, None
    small, roi_c, ws, hs = [], None, None, None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if roi_c is None:
            roi_c = clamp_roi(roi, frame.shape)
            ws = min(width, roi_c[2])
            hs = max(8, int(round(roi_c[3] * ws / roi_c[2])))
        x, y, w, h = roi_c
        gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        small.append(cv2.resize(gray, (ws, hs), interpolation=cv2.INTER_AREA))
    cap.release()
    if len(small) < 30:
        print("Error: video too short.")
        return None, None, None
    return np.stack(small), roi_c, (ws, hs)


def _robust_line(xs, ts, cfg):
    """Theil-Sen line of crossing time vs x with outlier rejection; None if not credible."""
    ok = np.isfinite(ts)
    if ok.sum() < max(6, int(0.15 * len(xs))):
        return None
    x, t = xs[ok], ts[ok]
    if (x.max() - x.min()) < cfg["min_span"] * (xs[-1] - xs[0]):
        return None
    slope, icpt, _, _ = theilslopes(t, x)
    if abs(slope) < 1e-7:
        return None
    resid = t - (slope * x + icpt)
    mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    inl = np.abs(resid) <= max(1.5, 3.0 * mad)
    if inl.sum() < max(6, 0.6 * ok.sum()):
        return None
    slope, icpt, _, _ = theilslopes(t[inl], x[inl])
    r = abs(np.corrcoef(x[inl], t[inl])[0, 1])
    if abs(slope) < 1e-7 or not np.isfinite(r) or r < cfg["min_corr"]:
        return None
    return {"slope": float(slope), "icpt": float(icpt), "frac": float(inl.sum() / len(xs))}


def find_motion_event(G, lag, sx, cfg):
    """
    Finds the interval in which a plug travels through the ROI and the direction it travels.
    A real event needs a front that PROPAGATES across the pipe (linear arrival time vs x), so a plug resting
    at the bottom, flicker or sensor noise is never accepted.  Returns None if there is no such event.
    """
    n, hs, ws = G.shape
    T = n - lag
    if T < 4 * cfg["min_event_frames"]:
        return None

    def diff(t):
        d = G[t + lag].astype(np.float32) - G[t].astype(np.float32)
        return d - np.median(d)

    sig = np.array([1.4826 * np.median(np.abs(diff(t))) for t in range(T)])
    thr = max(cfg["k_sigma"] * float(np.percentile(sig, 30)), cfg["min_pix_thr"])
    colfrac = np.stack([(np.abs(diff(t)) > thr).mean(axis=0) for t in range(T)]).astype(np.float32)
    col_active = (colfrac - np.percentile(colfrac, 20, axis=0)) > cfg["col_frac"]

    a = median_filter(col_active.mean(axis=1).astype(np.float32), size=3)
    q = np.sort(a)[:max(10, int(0.3 * T))]
    a0 = float(np.median(q))
    sig_a = float(1.4826 * np.median(np.abs(q - a0)))
    hi = a0 + max(cfg["hi_k"] * sig_a, cfg["min_active_frac"])
    lo = a0 + max(cfg["lo_k"] * sig_a, 0.5 * cfg["min_active_frac"])

    best, best_score = None, 0.0
    for (s0, e0) in _runs(_close_gaps(a > lo, cfg["event_gap"])):
        if e0 - s0 + 1 >= cfg["min_event_frames"] and a[s0:e0 + 1].max() >= hi:
            score = float(a[s0:e0 + 1].sum())
            if score > best_score:
                best, best_score = (s0, e0), score
    if best is None:
        return None

    pad = max(5, 2 * lag)
    ev_s, ev_e = max(0, best[0] - pad), min(T - 1, best[1] + pad)

    t_in = np.full(ws, np.nan)
    for x in range(ws):
        seg = _close_gaps(col_active[ev_s:ev_e + 1, x], cfg["gap_close"])
        runs = [r for r in _runs(seg) if r[1] - r[0] + 1 >= cfg["min_run"]]
        if runs:
            r0, r1 = max(runs, key=lambda r: r[1] - r[0])
            if r0 > 0:
                t_in[x] = ev_s + r0 + lag
    xs = (np.arange(ws) + 0.5) * sx
    fit = _robust_line(xs, t_in, cfg)
    if fit is None or fit["frac"] < cfg["min_valid_cols"]:
        return None
    return {"lag": lag, "ev": (ev_s, ev_e), "fit": fit, "xs": xs}


# --------------------------------------------------------------------------- #
# U-Net height map (plug height measured from the pipe BOTTOM)
# --------------------------------------------------------------------------- #
def height_spacetime(video_path, roi_c, masker, f0, f1, stride, ws):
    """H[k, x] = plug height (px, from the pipe bottom) of ROI column-bin x at sampled frame k."""
    idxs, rows = [], []
    for idx, fr in read_range(video_path, f0, f1, stride):
        m = masker.create_mask(fr, roi=roi_c) > 127
        y_top = np.where(m.any(axis=0), m.argmax(axis=0), m.shape[0])
        col_h = (m.shape[0] - y_top).astype(np.float32)[None, :]
        rows.append(cv2.resize(col_h, (ws, 1), interpolation=cv2.INTER_AREA)[0])
        idxs.append(idx)
    if len(rows) < 12:
        return None
    return np.asarray(idxs, dtype=float), np.stack(rows)


def gate_exit_and_anchor(fidx, mh, exit_frac):
    """
    mh = moving plug height at the gate (above the resting layer) vs time.
    exit   : after the peak, first time it falls below exit_frac * max      (the registered exit)
    anchor : rise through 20 % of max - only used to place the flow probe, NOT the registered entry
    """
    mx = float(mh.max())
    if mx < 3.0:
        return None
    kp = int(np.argmax(mh))
    idx = np.where(mh[:kp + 1] >= 0.2 * mx)[0]
    if idx.size == 0 or idx[0] < 1:
        return None
    k = idx[0]
    t_anchor = fidx[k - 1] + (0.2 * mx - mh[k - 1]) / (mh[k] - mh[k - 1]) * (fidx[k] - fidx[k - 1])
    idx = np.where(mh[kp:] < exit_frac * mx)[0]
    if idx.size == 0:
        return None
    k = kp + idx[0]
    thr = exit_frac * mx
    t_exit = fidx[k - 1] + (mh[k - 1] - thr) / (mh[k - 1] - mh[k]) * (fidx[k] - fidx[k - 1])
    return None if t_exit - t_anchor < 5 else {"t_anchor": float(t_anchor), "t_exit": float(t_exit)}


# --------------------------------------------------------------------------- #
# Optical-flow velocity of the real plug, entry confirmation, groups
# --------------------------------------------------------------------------- #
def plug_region_from_mask(mask, base_cols, margin=3):
    """U-Net plug mask minus the stationary resting layer (everything below the layer's resting surface)."""
    m = mask > 127
    Hm, Wm = m.shape
    rows = np.arange(Hm)[:, None]
    return m & (rows < (Hm - base_cols[None, :Wm] - margin))


def measure_flow(video_path, roi_c, masker, engine, scan_start, t_probe, t_exit, base_cols,
                 direction, to_mps, n_frames, step, cfg):
    """
    Scans frames from scan_start to the exit.  For every evaluated frame pair (gap k):
      * plug velocity from optical flow in the plug mask (9 gates, dense flow + particle tracks)
      * ENTRY: the centre gate moves >= entry_min_disp px in `entry_confirm_pairs` consecutive pairs
    k is chosen at t_probe so that particles move >= flow_target_disp px between the compared frames.
    Returns (records, k, entry_time or None).
    """
    x0, y0, w, h = roi_c
    crop = lambda fr: fr[y0:y0 + h, x0:x0 + w]
    n_g = cfg["flow_n_gates"]

    ks = [k for k in (1, 2, 4, 8, 16, 32) if k <= cfg["flow_k_max"]] or [1]
    t_p = int(round(t_probe))
    fr = read_frames(video_path, {t_p} | {t_p + k for k in ks if t_p + k < n_frames})
    k_lag, last_ok = None, ks[min(1, len(ks) - 1)]
    if t_p in fr:
        c0 = crop(fr[t_p])
        region = plug_region_from_mask(masker.create_mask(fr[t_p], roi=roi_c), base_cols)
        for k in ks:
            if (t_p + k) not in fr:
                break
            r = engine.plug_velocity(c0, crop(fr[t_p + k]), region, lag=k, direction=direction, n_gates=n_g)
            if np.isfinite(r["v_px"]):
                last_ok = k
                if abs(r["v_px"]) * k >= cfg["flow_target_disp"]:
                    k_lag = k
                    break
    k_lag = k_lag or last_ok
    del fr

    a, b = max(0, int(scan_start)), min(n_frames - 1, int(round(t_exit)) + k_lag)
    buf, recs, run, entry_t = deque(maxlen=k_lag + 1), [], [], None
    for idx, frame in read_range(video_path, a, b):
        buf.append((idx, crop(frame), plug_region_from_mask(masker.create_mask(frame, roi=roi_c), base_cols)))
        if len(buf) == k_lag + 1 and (buf[0][0] - a) % step == 0:
            i0, c0, r0 = buf[0]
            i1, c1, _ = buf[-1]
            r = engine.plug_velocity(c0, c1, r0, lag=k_lag, direction=direction, n_gates=n_g)
            t = 0.5 * (i0 + i1)
            recs.append({"t": t, "v": r["v_px"] * to_mps if np.isfinite(r["v_px"]) else np.nan,
                         "n_valid_gates": r["n_valid_gates"], "n_tracks": r["n_tracks"],
                         "gate_v": r["gate_v"], "gate_x": r["gate_x"]})
            if entry_t is None:
                gv = r["gate_v"][len(r["gate_v"]) // 2] if r["gate_v"] else np.nan
                run = run + [t] if (np.isfinite(gv) and gv * k_lag >= cfg["entry_min_disp"]) else []
                if len(run) >= cfg["entry_confirm_pairs"]:
                    entry_t = run[0]
        if (idx - a) % 60 == 0:
            print(f"    flow: frame {idx}/{b}")
    return recs, k_lag, entry_t


def make_groups(entry_f, exit_f, recs, G, fps):
    """Groups of G frames; v_group = mean of the measured frame-pair velocities in the group (outliers removed)."""
    start, end_total = int(round(entry_f)), int(round(exit_f))
    bounds = [start]
    while bounds[-1] < end_total:
        nxt = bounds[-1] + G
        if end_total - nxt < G / 2:
            nxt = end_total
        bounds.append(min(nxt, end_total))

    rows = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        sel = [r for r in recs if a <= r["t"] < b and np.isfinite(r["v"])]
        v = np.nan
        if sel:
            vv = np.array([r["v"] for r in sel])
            med = np.median(vv)
            mad = 1.4826 * np.median(np.abs(vv - med))
            v = float(np.mean(vv[np.abs(vv - med) <= max(3 * mad, 0.05 * abs(med), 1e-6)]))
        rows.append({"a": a, "b": b, "dt": (b - a) / fps, "v": v, "n": len(sel), "src": "flow",
                     "gates": float(np.median([r["n_valid_gates"] for r in sel])) if sel else 0.0,
                     "tracks": float(np.median([r["n_tracks"] for r in sel])) if sel else 0.0})
    vv = np.array([r["v"] for r in rows])
    ok = np.isfinite(vv)
    if not ok.any():
        return None
    cen = np.array([0.5 * (r["a"] + r["b"]) for r in rows])
    cum = 0.0
    for i, r in enumerate(rows):
        if not ok[i]:                         # only if a group has no measurement at all
            r["v"], r["src"] = float(np.interp(cen[i], cen[ok], vv[ok])), "interp"
        r["L"] = r["v"] * r["dt"] * 1000.0
        cum += r["L"]
        r["cum"] = cum
    return rows


# --------------------------------------------------------------------------- #
# Front angle
# --------------------------------------------------------------------------- #
def find_edge_peak_time(fidx, H, entry_side, cfg):
    """Time at which the plug height (from the bottom) at the entry-side edge of the ROI first reaches angle_hi_frac of its max."""
    K, ws = H.shape
    b0 = int(round(cfg["angle_edge_lo"] * ws))
    b1 = max(b0 + 2, int(round(cfg["angle_edge_hi"] * ws)))
    cols = slice(ws - b1, ws - b0) if entry_side == "right" else slice(b0, b1)
    he = median_filter(np.median(H[:, cols], axis=1), size=3)
    base, peak = float(np.median(he[:max(3, K // 12)])), float(np.percentile(he, 95))
    if peak - base < 3.0:
        return None
    idx = np.where(he >= cfg["angle_hi_frac"] * peak)[0]
    if idx.size == 0 or idx[0] < 1:
        return None
    k, thr = idx[0], cfg["angle_hi_frac"] * peak
    return float(fidx[k - 1] + (thr - he[k - 1]) / (he[k] - he[k - 1]) * (fidx[k] - fidx[k - 1]))


def ramp_angle(mask, direction, cfg):
    """
    Angle of the plug front face from the U-Net surface profile h(x) (height from the bottom).
    The front face = flank between the plug body (entry side) and the resting layer ahead of the front.
    The steepest straight part of that flank is fitted (window = angle_win_frac of the ROI width).
    direction = +1 if the plug travels toward +x, -1 otherwise.
    """
    m = mask > 127
    H, W = m.shape
    y_top = np.where(m.any(axis=0), m.argmax(axis=0), H)
    h = median_filter((H - y_top).astype(np.float32), size=9)

    k, kb = max(10, int(0.3 * W)), int(0.4 * W)
    body, ahead = (h[:kb], h[W - k:]) if direction > 0 else (h[W - kb:], h[:k])
    P, B = float(np.percentile(body, 98)), float(np.median(ahead))
    if P - B < max(8.0, 0.06 * H):
        return None

    band = (h >= B + 0.10 * (P - B)) & (h <= B + 0.97 * (P - B))
    runs = _runs(_close_gaps(band, 3))
    win = max(15, int(cfg["angle_win_frac"] * W))
    if not runs:
        return None
    r0, r1 = max(runs, key=lambda r: r[1] - r[0])
    best = None
    for s in range(r0, r1 - win + 2, 2):
        xw, hw = np.arange(s, s + win).astype(float), h[s:s + win].astype(float)
        a, b, _, _ = theilslopes(hw, xw)
        if (direction > 0 and a >= 0) or (direction < 0 and a <= 0):
            continue
        if abs(np.corrcoef(xw, hw)[0, 1]) < 0.95:
            continue
        if best is None or abs(a) > abs(best[0]):
            best = (a, b, xw, hw)
    if best is None:
        return None
    a, b, xw, hw = best
    return float(np.degrees(np.arctan(abs(a)))), {"a": float(a), "b": float(b), "x_pts": xw, "h_pts": hw,
                                                  "h": h, "H": H, "W": W}


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_groups(groups, recs, entry_f, exit_f, path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for g in groups:
        ax.bar((g["a"] + g["b"]) / 2, g["v"], width=(g["b"] - g["a"]) * 0.95, alpha=0.35,
               color="tab:gray", edgecolor="k")
        ax.text((g["a"] + g["b"]) / 2, g["v"], f"{g['L']:.0f} mm", ha="center", va="bottom", fontsize=8)
    pts = [(r["t"], r["v"]) for r in recs if np.isfinite(r["v"])]
    if pts:
        ax.scatter(*zip(*pts), c="tab:green", s=14, label="measured frame-pair velocity", zorder=3)
    ax.axvline(entry_f, color="blue", ls=":", label="entry (flow)")
    ax.axvline(exit_f, color="green", ls=":", label="exit (height)")
    ax.set_xlabel("frame")
    ax.set_ylabel("plug velocity (m/s)")
    ax.set_title("Velocity and length per frame group")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_ramp(frame_rgb, roi_c, mask, info, angle, title, path):
    x0, y0, w, hh = roi_c
    H, W = info["H"], info["W"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(frame_rgb[y0:y0 + hh, x0:x0 + w])
    ov = np.zeros((mask.shape[0], mask.shape[1], 4))
    ov[mask > 127] = [1, 0, 0, 0.25]
    ax.imshow(ov)
    ax.plot(np.arange(W), H - info["h"], "c-", lw=1, label="U-Net surface")
    ax.scatter(info["x_pts"], H - info["h_pts"], c="yellow", s=10, label="front face used")
    xl = np.array([info["x_pts"].min(), info["x_pts"].max()])
    ax.plot(xl, H - (info["a"] * xl + info["b"]), color="lime", lw=3, label=f"Angle: {angle:.1f}°")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_unified_analysis(video_path, roi, calibration_config, video_props):
    cfg = CFG
    if not roi or len(roi) != 4 or roi[2] <= 0 or roi[3] <= 0:
        print("Error: Invalid ROI. Please run pipeline_setup.py again.")
        return None
    px_to_mm = calculate_px_to_mm(calibration_config)
    if px_to_mm <= 0:
        print("Error: invalid scale calibration (p1/p2/dist_mm).")
        return None
    fps = video_props.get('fps', 500.0)
    to_mps = px_to_mm * fps / 1000.0
    os.makedirs("test_results", exist_ok=True)

    print(f"\nStarting Unified Pipeline v7 (Scale: {px_to_mm:.4f} mm/px, FPS: {fps})...")

    # ---- Pass 1: event window + travel direction ----
    print("Pass 1: scanning for the moving plug...")
    G, roi_c, (ws, hs) = build_activity_stack(video_path, roi, cfg["act_width"])
    if G is None:
        return None
    n = G.shape[0]
    _, _, w_roi, h_roi = roi_c
    sx = w_roi / ws
    event = None
    for lag in cfg["lags"]:
        event = find_motion_event(G, lag, sx, cfg)
        if event is not None:
            break
    if event is None:
        print("\nNo propagating moving plug found in this video (resting bed / noise is correctly ignored).")
        return None
    ev_s, ev_e = event["ev"]
    direction = 1 if event["fit"]["slope"] > 0 else -1       # +1: plug travels toward +x
    v_px_act = abs(1.0 / event["fit"]["slope"])
    del G

    print("Loading U-Net and flow engine...")
    masker = importlib.import_module("core.03a_unet_masker").PlugMasker()
    engine = importlib.import_module("core.03b_flow_engine").RAFTFlowEngine()

    # ---- Pass 2a: plug height from the bottom -> exit ----
    stride = int(np.clip(np.floor((w_roi / v_px_act) / 40.0), 1, 3))
    pad = max(40, int(0.3 * (ev_e - ev_s)))
    f0, f1 = max(0, ev_s - pad), min(n - 1, ev_e + event["lag"] + pad)
    print(f"Pass 2a: U-Net plug height over frames {f0}-{f1} (stride {stride})...")
    hm = height_spacetime(video_path, roi_c, masker, f0, f1, stride, ws)
    if hm is None:
        print("\nError: could not build the plug height map.")
        return None
    fidx, H = hm
    g0, g1 = max(0, ws // 2 - 1), min(ws, ws // 2 + 2)
    hg = median_filter(np.median(H[:, g0:g1], axis=1), size=3)           # plug height from the bottom at the gate
    nb = max(3, len(hg) // 12)
    ev = gate_exit_and_anchor(fidx, np.clip(hg - np.median(hg[:nb]), 0, None), cfg["exit_frac"])
    if ev is None:
        print("\nError: could not register the exit of the moving plug at the gate.")
        return None
    exit_f = ev["t_exit"]

    base_bins = np.median(H[:max(3, H.shape[0] // 12)], axis=0)           # resting layer height per column
    base_cols = np.interp(np.arange(w_roi) + 0.5, event["xs"], base_bins)

    # ---- Pass 2b: optical flow on the real plug -> entry + velocities ----
    prov_transit = exit_f - ev["t_anchor"]
    step = 1 if prov_transit < 80 else cfg["flow_step"]
    t_probe = 0.5 * (ev["t_anchor"] + exit_f)
    print(f"Pass 2b: optical flow on the real plug (U-Net plug mask, {cfg['flow_n_gates']} gates)...")
    recs, k_lag, entry_f = measure_flow(video_path, roi_c, masker, engine, ev_s, t_probe, exit_f, base_cols,
                                        direction, to_mps, n, step, cfg)
    if entry_f is None:
        print(" -> WARNING: flow never confirmed the plug entering the gate; using the height anchor instead.")
        entry_f = ev["t_anchor"]
    if exit_f - entry_f < 5:
        print("\nError: exit comes before the flow-confirmed entry.")
        return None
    entry_frame, exit_frame = int(round(entry_f)), int(round(exit_f))
    transit_time = (exit_frame - entry_frame) / fps      # same frames the groups use

    G_eff = cfg["group_frames"]
    if exit_f - entry_f < 5 * G_eff:                       # short (fast) transit: smaller groups
        G_eff = max(4, int(round((exit_f - entry_f) / 5.0)))
    groups = make_groups(entry_f, exit_f, [r for r in recs if r["t"] >= entry_f - 1e-9], G_eff, fps)
    if not groups:
        print("\nError: no usable optical-flow velocities between entry and exit.")
        return None

    sel = (fidx >= entry_f - 5) & (fidx <= exit_f + 5)
    max_height_mm = float(hg[sel].max()) * px_to_mm if sel.any() else float(hg.max()) * px_to_mm

    L_sum = float(sum(g["L"] for g in groups))
    v_avg = L_sum / 1000.0 / transit_time
    v_all = np.array([r["v"] for r in recs if r["t"] >= entry_f and np.isfinite(r["v"])])

    print(f"\n -> ENTRY  : frame {entry_frame}  (optical flow confirms the plug moving through the gate)")
    print(f" -> EXIT   : frame {exit_frame}  (moving height at the gate < {cfg['exit_frac']*100:.0f}% of its max)")
    print(f" -> MAX PLUG HEIGHT (from the pipe bottom): {max_height_mm:.2f} mm")
    print(f" -> flow frame gap: {k_lag} frame(s); {len(v_all)} frame-pair velocities measured between entry and exit")

    gx = next((r["gate_x"] for r in recs if r["gate_x"]), [])
    if gx:
        print("\n -> VELOCITY PER GATE (median over the transit):")
        print(f"    {'gate x (px)':>12} {'v (m/s)':>9} {'pairs':>6}")
        for j, xg in enumerate(gx):
            vj = np.array([r["gate_v"][j] for r in recs
                           if r["t"] >= entry_f and len(r["gate_v"]) > j and np.isfinite(r["gate_v"][j])])
            print(f"    {xg:>12.0f} {(np.median(vj) * to_mps if vj.size else np.nan):>9.3f} {vj.size:>6}")

    print(f"\n -> VELOCITY & LENGTH PER GROUP OF {G_eff} FRAMES")
    print("    " + "-" * 92)
    print(f"    {'grp':>3} {'frames':>12} {'dt (s)':>8} {'v (m/s)':>9} {'length (mm)':>12} {'cum. (mm)':>10} "
          f"{'pairs':>6} {'gates':>6} {'tracks':>7}")
    print("    " + "-" * 92)
    for i, g in enumerate(groups, 1):
        tag = "  (interpolated)" if g["src"] == "interp" else ""
        print(f"    {i:>3} {str(g['a'])+'-'+str(g['b']):>12} {g['dt']:>8.4f} {g['v']:>9.3f} {g['L']:>12.1f} "
              f"{g['cum']:>10.1f} {g['n']:>6} {g['gates']:>6.0f} {g['tracks']:>7.0f}{tag}")
    print("    " + "-" * 92)
    print(f"    TOTAL: time {transit_time:.4f} s | length (sum of groups) {L_sum:.1f} mm | "
          f"average velocity {v_avg:.3f} m/s | median pair velocity {np.median(v_all):.3f} m/s")

    try:
        plot_groups(groups, recs, entry_f, exit_f, "test_results/velocity_groups.jpg")
    except Exception as e:
        print(f" -> Could not save velocity plot: {e}")

    # ---- Pass 2c: front angle, frames AFTER the entry-side edge reaches ~95 % of max height ----
    print("\nPass 2c: front angle...")
    side = cfg["angle_side"]
    entry_side = ("right" if direction < 0 else "left") if side == "entry" else side
    t_hi = find_edge_peak_time(fidx, H, entry_side, cfg)
    angle_frames = []
    if t_hi is not None:
        f_hi = int(np.floor(t_hi))
        angle_frames = [f for f in range(f_hi + 1, f_hi + 1 + cfg["n_angle_frames"]) if f < n]
        print(f" -> {entry_side}-side edge reaches {cfg['angle_hi_frac']*100:.0f}% of max plug height at frame "
              f"{t_hi:.1f}; measuring angle on frames {angle_frames[0]}-{angle_frames[-1]}")
    else:
        print(" -> edge height rise not found.")

    frames = read_frames(video_path, {0, n - 1} | set(angle_frames))
    found = []
    for f in angle_frames:
        if f in frames:
            mask = masker.create_mask(frames[f], roi=roi_c)
            r = ramp_angle(mask, direction, cfg)
            if r is not None:
                found.append((r[0], r[1], mask, f))

    if found:
        angs = np.array([q[0] for q in found])
        final_angle = float(np.median(angs))
        final_slope = float(np.tan(np.radians(final_angle)))
        print(" -> Front angle per frame: " + ", ".join(f"f{q[3]}: {q[0]:.1f}°" for q in found))
        print(f" -> FRONT ANGLE (median of {len(found)} frames) = {final_angle:.2f}°")
        best = found[int(np.argmin(np.abs(angs - final_angle)))]
        try:
            plot_ramp(cv2.cvtColor(frames[best[3]], cv2.COLOR_BGR2RGB), roi_c, best[2], best[1], best[0],
                      f"Front face fit (frame {best[3]}; median of {len(found)} frames = {final_angle:.1f}°)",
                      "test_results/angle_verification.jpg")
        except Exception as e:
            print(f" -> Could not save angle plot: {e}")
    else:
        final_angle, final_slope = 0.0, 0.0
        print(" -> WARNING: no valid front face found; angle left at 0 (you will be asked to enter it).")

    # ---- first / last frame heights (unchanged) ----
    initial_height_mm = final_height_mm = 0.0
    if 0 in frames:
        initial_height_mm = get_robust_median_height(masker.create_mask(frames[0], roi=roi_c), px_to_mm)
    if (n - 1) in frames:
        final_height_mm = get_robust_median_height(masker.create_mask(frames[n - 1], roi=roi_c), px_to_mm)

    return {
        "initial_height_mm": initial_height_mm,
        "final_height_mm": final_height_mm,
        "entry_frame": entry_frame,
        "exit_frame": exit_frame,
        "transit_time_sec": transit_time,
        "velocity_mps": v_avg,
        "length_mm": L_sum,
        "max_height_mm": max_height_mm,
        "front_angle_deg": final_angle,
        "edge_slope": final_slope,
        "groups": groups,
        "flow_frame_gap": k_lag,
    }