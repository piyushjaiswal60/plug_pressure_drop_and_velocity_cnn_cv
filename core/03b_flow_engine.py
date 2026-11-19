"""
03b_flow_engine.py  (v2)

Optical-flow engine for granular plug videos.  The class name RAFTFlowEngine and the old methods
compute_flow() / get_median_velocity() are kept, so 03c / 03e / tests keep working.

What is better than the old Farneback-only engine
-------------------------------------------------
* DIS dense flow (falls back to a gentler Farneback) - no 65 px window that smears the speckle.
* Local-contrast pre-processing (high-pass + shared normalisation): removes illumination gradients and
  frame-to-frame flicker, keeps the potash speckle.
* Automatic down-scaling for fast plugs (phase-correlation shift estimate), so 15-30 px/frame motion is
  tracked as reliably as 2 px/frame.
* Forward-backward consistency check: a pixel/particle is used only if flow(t->t+k) and flow(t+k->t) agree.
  Bad tracks are removed by consistency, NOT by a magnitude threshold, so slow real plug pixels are kept.
* Sparse particle tracking (Lucas-Kanade on up to ~1500 corner features inside the plug region) in addition
  to the dense field.
* plug_velocity(): velocity of the REAL plug.  The caller passes the plug region (from the U-Net plug mask,
  stationary resting layer removed); velocity is evaluated in many vertical gates across the ROI, from the
  dense field and the sparse tracks, and combined robustly.  No "moving pixels only" selection.
"""

import numpy as np
import cv2


def _make_dis():
    maker = getattr(cv2, "DISOpticalFlow_create", None)
    if maker is None and hasattr(cv2, "DISOpticalFlow"):
        maker = getattr(cv2.DISOpticalFlow, "create", None)
    if maker is None:
        return None
    try:
        d = maker(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        d.setFinestScale(1)
        d.setPatchSize(8)
        d.setPatchStride(4)
        d.setGradientDescentIterations(25)
        d.setVariationalRefinementIterations(5)
        return d
    except Exception:
        return None


class RAFTFlowEngine:
    def __init__(self, model_type='small', device=None, use_dis=True, fb_tol=0.75):
        self.device = device
        self.fb_tol = fb_tol
        self.dis = _make_dis() if use_dis else None
        print("Initializing Optical Flow Engine v2 "
              f"({'DIS dense flow' if self.dis is not None else 'Farneback'} + forward-backward check "
              "+ Lucas-Kanade particle tracking)...")

    # ------------------------------------------------------------------ #
    # pre-processing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _gray(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    @staticmethod
    def _prep_pair(g1, g2):
        """High-pass both frames and normalise them with ONE shared scale (no flicker difference)."""
        a = g1.astype(np.float32)
        b = g2.astype(np.float32)
        a -= cv2.GaussianBlur(a, (0, 0), 12)
        b -= cv2.GaussianBlur(b, (0, 0), 12)
        sd = 0.5 * (np.std(a) + np.std(b)) + 1e-6
        k = 40.0 / sd
        a = np.clip(a * k + 128.0, 0, 255).astype(np.uint8)
        b = np.clip(b * k + 128.0, 0, 255).astype(np.uint8)
        return a, b

    @staticmethod
    def _phase_shift(a, b, weight=None):
        """Dominant translation (dx, dy, confidence) of b relative to a via phase correlation (1/4 scale)."""
        f = 0.25
        sa = cv2.resize(a.astype(np.float32), None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        sb = cv2.resize(b.astype(np.float32), None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        if weight is not None:
            w = cv2.resize(weight.astype(np.float32), (sa.shape[1], sa.shape[0]), interpolation=cv2.INTER_AREA)
            sa, sb = (sa - 128.0) * w, (sb - 128.0) * w
        if min(sa.shape) < 8:
            return 0.0, 0.0, 0.0
        win = cv2.createHanningWindow((sa.shape[1], sa.shape[0]), cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(sa, sb, win)
        return dx / f, dy / f, float(resp)

    # ------------------------------------------------------------------ #
    # dense flow
    # ------------------------------------------------------------------ #
    def _flow(self, a, b):
        if self.dis is not None:
            return self.dis.calc(a, b, None)
        return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 6, 31, 5, 7, 1.5, 0)

    def _dense(self, a, b, region=None, fb=True, max_disp=7.0):
        """
        Dense flow a->b with automatic scale selection and optional forward-backward validity.
        Returns (flow HxWx2 float32 in full-res pixels, valid HxW bool, shift_guess dx).
        """
        dx, dy, _ = self._phase_shift(a, b, region)
        big = max(abs(dx), abs(dy))
        s = 1.0 if big <= 9.0 else float(np.clip(max_disp / big, 0.25, 1.0))
        if s < 1.0:
            sa = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            sb = cv2.resize(b, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        else:
            sa, sb = a, b
        fwd = self._flow(sa, sb)
        valid = np.ones(sa.shape[:2], dtype=bool)
        if fb:
            bwd = self._flow(sb, sa)
            h, w = sa.shape[:2]
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            bw = cv2.remap(bwd, gx + fwd[..., 0], gy + fwd[..., 1], cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
            err = np.hypot(fwd[..., 0] + bw[..., 0], fwd[..., 1] + bw[..., 1])
            valid = err < self.fb_tol
        if s < 1.0:
            fwd = cv2.resize(fwd, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR) / s
            valid = cv2.resize(valid.astype(np.uint8), (a.shape[1], a.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
        return fwd.astype(np.float32), valid, dx

    # ------------------------------------------------------------------ #
    # backward-compatible API
    # ------------------------------------------------------------------ #
    def compute_flow(self, frame1, frame2, roi=None, mask=None):
        """Dense flow (px between the two frames) inside roi; zero outside `mask` (if given)."""
        if roi:
            x, y, w, h = roi
            x, y = max(0, x), max(0, y)
            w = max(1, min(w, frame1.shape[1] - x))
            h = max(1, min(h, frame1.shape[0] - y))
            f1, f2 = frame1[y:y + h, x:x + w], frame2[y:y + h, x:x + w]
            if mask is not None and mask.shape == frame1.shape[:2]:
                mask = mask[y:y + h, x:x + w]
        else:
            f1, f2 = frame1, frame2
        if f1.size == 0 or f2.size == 0 or f1.shape[0] == 0 or f1.shape[1] == 0:
            return None

        a, b = self._prep_pair(self._gray(f1), self._gray(f2))
        region = None
        if mask is not None:
            if mask.shape[:2] != a.shape[:2]:
                mask = cv2.resize(mask, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
            region = (mask > 127)
        flow, _, _ = self._dense(a, b, region.astype(np.float32) if region is not None else None, fb=False)

        if region is not None:
            flow[~region] = 0
        else:                                   # no mask: remove global camera shift
            flow[..., 0] -= np.median(flow[..., 0])
            flow[..., 1] -= np.median(flow[..., 1])
        return flow

    def get_median_velocity(self, flow, mask=None):
        u, v = flow[:, :, 0], flow[:, :, 1]
        if mask is not None:
            sel = (mask > 127)
            u, v = u[sel], v[sel]
        else:
            u, v = u.flatten(), v.flatten()
        if len(u) == 0:
            return 0.0, 0.0
        return float(np.median(u)), float(np.median(v))

    # ------------------------------------------------------------------ #
    # sparse particle tracking
    # ------------------------------------------------------------------ #
    def track_particles(self, a, b, region, init_flow=None, max_pts=1200, fb_tol=0.6):
        """
        Lucas-Kanade tracks of corner features inside `region`, forward-backward checked.
        If init_flow is given, tracking is seeded with it, so large displacements cannot alias.
        """
        pts = cv2.goodFeaturesToTrack(a, maxCorners=max_pts, qualityLevel=0.004, minDistance=3,
                                      blockSize=5, mask=region.astype(np.uint8) * 255)
        if pts is None or len(pts) < 10:
            return np.empty((0, 2)), np.empty((0, 2))
        lk = dict(winSize=(31, 31), maxLevel=4,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if init_flow is not None:
            xi = np.clip(np.round(pts[:, 0, 0]).astype(int), 0, init_flow.shape[1] - 1)
            yi = np.clip(np.round(pts[:, 0, 1]).astype(int), 0, init_flow.shape[0] - 1)
            guess = (pts + init_flow[yi, xi][:, None, :]).astype(np.float32)
            p1, st1, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, guess, flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
            p0, st0, _ = cv2.calcOpticalFlowPyrLK(b, a, p1, pts.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
        else:
            p1, st1, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, **lk)
            p0, st0, _ = cv2.calcOpticalFlowPyrLK(b, a, p1, None, **lk)
        err = np.linalg.norm((p0 - pts).reshape(-1, 2), axis=1)
        ok = (st1.ravel() == 1) & (st0.ravel() == 1) & (err < fb_tol)
        return pts.reshape(-1, 2)[ok], (p1 - pts).reshape(-1, 2)[ok]

    # ------------------------------------------------------------------ #
    # velocity of the real plug
    # ------------------------------------------------------------------ #
    def plug_velocity(self, crop1, crop2, region, lag=1, direction=1, n_gates=9,
                      min_gate_px=400, min_tracks=8, margin=0.06):
        """
        crop1, crop2 : ROI crops of two frames `lag` frames apart (BGR or gray)
        region       : bool mask (crop sized) of the REAL plug (U-Net plug mask, resting layer removed)
        direction    : +1 if the plug moves toward +x, -1 otherwise (result is positive along motion)
        Returns dict: v_px (px/frame along motion, NaN if not enough gates), gate_x, gate_v (px/frame),
                      n_valid_gates, n_dense, n_tracks, v_dense, v_lk, v_phase.
        """
        out = {"v_px": np.nan, "gate_x": [], "gate_v": [], "n_valid_gates": 0, "n_dense": 0,
               "n_tracks": 0, "v_dense": np.nan, "v_lk": np.nan, "v_phase": np.nan}
        g1, g2 = self._gray(crop1), self._gray(crop2)
        H, W = g1.shape[:2]
        region = region.astype(bool)
        if region.sum() < min_gate_px:
            return out
        er = cv2.erode(region.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        if er.sum() < min_gate_px:
            er = region

        a, b = self._prep_pair(g1, g2)
        flow, valid, dxp = self._dense(a, b, er.astype(np.float32), fb=True)
        pts, disp = self.track_particles(a, b, er, init_flow=flow)

        sgn = 1.0 if direction >= 0 else -1.0
        out["v_phase"] = sgn * dxp / lag

        sel = er & valid
        out["n_dense"] = int(sel.sum())
        out["n_tracks"] = int(len(pts))
        if sel.sum() >= min_gate_px:
            out["v_dense"] = sgn * float(np.mean(np.percentile(flow[..., 0][sel], [10, 25, 50, 75, 90]))) / lag
        if len(pts) >= min_tracks:
            out["v_lk"] = sgn * float(np.median(disp[:, 0])) / lag

        xs = np.linspace(margin, 1 - margin, n_gates) * W
        half = 0.5 * W / (n_gates + 1)
        cols = np.arange(W)
        gv, gx = [], []
        for xc in xs:
            strip = (cols >= xc - half) & (cols <= xc + half)
            m = sel & strip[None, :]
            d_val = l_val = None
            if m.sum() >= min_gate_px:
                u = flow[..., 0][m]
                lo, hi = np.percentile(u, [10, 90])
                d_val = sgn * float(np.mean(u[(u >= lo) & (u <= hi)])) / lag     # trimmed mean over ALL plug pixels
            if len(pts):
                ps = (pts[:, 0] >= xc - half) & (pts[:, 0] <= xc + half)
                if ps.sum() >= min_tracks:
                    l_val = sgn * float(np.median(disp[ps, 0])) / lag
            if d_val is not None and l_val is not None:
                tol = max(1.5 / lag, 0.3 * abs(d_val))
                val = 0.5 * (d_val + l_val) if abs(d_val - l_val) <= tol else d_val
            else:
                val = d_val if d_val is not None else l_val
            gx.append(float(xc))
            gv.append(np.nan if val is None else float(val))
        out["gate_x"], out["gate_v"] = gx, gv
        good = np.array([v for v in gv if np.isfinite(v)])
        out["n_valid_gates"] = int(len(good))
        if len(good) >= 2:
            out["v_px"] = float(np.median(good))
        return out