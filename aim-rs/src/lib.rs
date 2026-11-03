//! Rust port of `aim.py` (PROJECT AA) for differential testing.
//!
//! - [`select_target`] and [`smooth_move`] mirror the Python semantics exactly,
//!   including banker's rounding (`round_half_even`) to match CPython `round()`.
//! - Jitter randomness intentionally differs from Python (Mersenne Twister):
//!   the deterministic core is covered by differential vectors (`jitter = 0`),
//!   jitter itself by property tests (bounds + determinism per seed).

use serde::Deserialize;

/// One detection: x1, y1, x2, y2, conf, cls in pixels (mirrors the `(N, 6)` rows).
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Det {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
    pub conf: f64,
    pub cls: f64,
}

impl Det {
    pub fn mid_x(&self) -> f64 {
        (self.x1 + self.x2) / 2.0
    }
    pub fn mid_y(&self) -> f64 {
        (self.y1 + self.y2) / 2.0
    }
    pub fn height(&self) -> f64 {
        self.y2 - self.y1
    }
}

fn argmin_first(values: &[f64]) -> usize {
    let mut best = 0usize;
    for (i, &v) in values.iter().enumerate().skip(1) {
        if v < values[best] {
            best = i;
        }
    }
    best
}

/// Pick one target. Returns `(x_mid, y_mid, box_height)` in pixels, or `None`.
///
/// Mirrors `aim.select_target`: FOV box filter, sticky lock on `last`, then
/// nearest-to-center (or nearest-to-last when `center` is false).
#[allow(clippy::too_many_arguments)]
pub fn select_target(
    dets: &[Det],
    cw: f64,
    ch: f64,
    center: bool,
    last: Option<(f64, f64)>,
    sticky_px: f64,
    fov: Option<(f64, f64)>,
    max_box_h: f64,
) -> Option<(f64, f64, f64)> {
    if dets.is_empty() {
        return None;
    }
    let mut idx: Vec<usize> = (0..dets.len()).collect();
    if let Some((fx, fy)) = fov {
        if fx != 0.0 && fy != 0.0 {
            idx.retain(|&i| {
                (dets[i].mid_x() - cw).abs() <= fx && (dets[i].mid_y() - ch).abs() <= fy
            });
            if idx.is_empty() {
                return None;
            }
        }
    }
    if max_box_h != 0.0 {
        idx.retain(|&i| dets[i].height() <= max_box_h);
        if idx.is_empty() {
            return None;
        }
    }
    let dist_to = |p: (f64, f64), i: usize| {
        let dx = dets[i].mid_x() - p.0;
        let dy = dets[i].mid_y() - p.1;
        dx.hypot(dy)
    };
    if let Some(lp) = last {
        if sticky_px != 0.0 {
            let ds: Vec<f64> = idx.iter().map(|&i| dist_to(lp, i)).collect();
            let j = argmin_first(&ds);
            if ds[j] <= sticky_px {
                let i = idx[j];
                return Some((dets[i].mid_x(), dets[i].mid_y(), dets[i].height()));
            }
        }
    }
    let j = if center {
        let ds: Vec<f64> = idx.iter().map(|&i| dist_to((cw, ch), i)).collect();
        argmin_first(&ds)
    } else if let Some(lp) = last {
        let ds: Vec<f64> = idx.iter().map(|&i| dist_to(lp, i)).collect();
        argmin_first(&ds)
    } else {
        0
    };
    let i = idx[j];
    Some((dets[i].mid_x(), dets[i].mid_y(), dets[i].height()))
}

/// Vertical offset from box center toward the head, by box height.
/// Mirrors `aim.headshot_offset` exactly (bin edges included).
pub fn headshot_offset(h: f64, headshot_mode: bool) -> f64 {
    if !headshot_mode {
        return h * 0.2;
    }
    let k = if h < 80.0 {
        0.42
    } else if h <= 150.0 {
        0.38
    } else {
        0.33
    };
    h * k
}

/// ID-stable selection over parallel detection/id slices.
/// Mirrors `aim.select_locked`: keep `locked` when present with
/// `conf >= min_conf` (highest conf on ties); with `recover_px > 0` and a
/// last position, spatial continuity wins over the id on disagreement
/// (tracker swaps on crossings); else fall back to [`select_target`] and
/// adopt the picked row's id (-1/None = unknown).
/// Returns `(target, new_lock)` with target `(x_mid, y_mid, h, tid)`.
#[allow(clippy::too_many_arguments)]
pub fn select_locked(
    dets: &[Det],
    ids: &[i64],
    locked: Option<i64>,
    cw: f64,
    ch: f64,
    center: bool,
    last: Option<(f64, f64)>,
    sticky_px: f64,
    fov: Option<(f64, f64)>,
    max_box_h: f64,
    min_conf: f64,
    recover_px: f64,
) -> (Option<(f64, f64, f64, Option<i64>)>, Option<i64>) {
    let norm = |t: i64| if t < 0 { None } else { Some(t) };
    if let Some(lk) = locked {
        if !dets.is_empty() {
            if let Some((lx, ly)) = last {
                if recover_px != 0.0 {
                    // ID match first: only override on clear disagreement,
                    // else the lock would flap between nearby targets.
                    let mut best_id: Option<usize> = None;
                    let mut best_c = -1.0f64;
                    for (i, d) in dets.iter().enumerate() {
                        let tid = ids.get(i).copied().and_then(norm);
                        if tid == Some(lk) && d.conf >= min_conf && d.conf > best_c {
                            best_id = Some(i);
                            best_c = d.conf;
                        }
                    }
                    if let Some(i) = best_id {
                        let d = &dets[i];
                        let dd = (d.mid_x() - lx).hypot(d.mid_y() - ly);
                        if dd <= recover_px {
                            return (
                                Some((d.mid_x(), d.mid_y(), d.height(), Some(lk))),
                                Some(lk),
                            );
                        }
                    }
                    let mut di: Option<usize> = None;
                    let mut dd = recover_px;
                    for (i, d) in dets.iter().enumerate() {
                        if d.conf < min_conf {
                            continue;
                        }
                        let dist = (d.mid_x() - lx).hypot(d.mid_y() - ly);
                        if dist < dd || (di.is_none() && dist <= dd) {
                            di = Some(i);
                            dd = dist;
                        }
                    }
                    if let Some(i) = di {
                        let d = &dets[i];
                        let tid = ids.get(i).copied().and_then(norm);
                        return (Some((d.mid_x(), d.mid_y(), d.height(), tid)), tid);
                    }
                }
            }
        }
        let mut best: Option<usize> = None;
        let mut best_c = -1.0f64;
        for (i, d) in dets.iter().enumerate() {
            let tid = ids.get(i).copied().and_then(norm);
            if tid == Some(lk) && d.conf >= min_conf && d.conf > best_c {
                best = Some(i);
                best_c = d.conf;
            }
        }
        if let Some(i) = best {
            let d = &dets[i];
            return (
                Some((d.mid_x(), d.mid_y(), d.height(), Some(lk))),
                Some(lk),
            );
        }
    }
    match select_target(dets, cw, ch, center, last, sticky_px, fov, max_box_h) {
        None => (None, None),
        Some((xm, ym, h)) => {
            let mut tid = None;
            for (i, d) in dets.iter().enumerate() {
                if (d.mid_x() - xm).abs() < 1e-9
                    && (d.mid_y() - ym).abs() < 1e-9
                    && (d.height() - h).abs() < 1e-9
                {
                    tid = ids.get(i).copied().and_then(norm);
                    break;
                }
            }
            (Some((xm, ym, h, tid)), tid)
        }
    }
}
/// SunOne-style pixels -> mouse-counts conversion, fixed for center-crop.
///
/// Mirrors `aim.fov_to_counts` bit-exactly (same op order): full game-window
/// size in the denominator, not the capture size. Returns `(0.0, 0.0)` when
/// disabled/invalid.
pub fn fov_to_counts(
    dx_px: f64,
    dy_px: f64,
    screen_w: f64,
    screen_h: f64,
    fov_x: f64,
    fov_y: f64,
    sens: f64,
    yaw: f64,
    pitch: f64,
    speed: f64,
    base_fov: f64,
    fov_scaled: bool,
) -> (f64, f64) {
    if !dx_px.is_finite() || !dy_px.is_finite() || !screen_w.is_finite() || !screen_h.is_finite() {
        return (0.0, 0.0);
    }
    if fov_x <= 0.0 || fov_y <= 0.0 || sens == 0.0 || yaw == 0.0 || pitch == 0.0 {
        return (0.0, 0.0);
    }
    let scale = if fov_scaled && base_fov > 1.0 { fov_x / base_fov } else { 1.0 };
    let den_x = sens * yaw * scale;
    let den_y = sens * pitch * scale;
    if den_x == 0.0 || den_y == 0.0 {
        return (0.0, 0.0);
    }
    let dx = dx_px * (fov_x / screen_w.max(1.0)) / den_x * speed;
    let dy = dy_px * (fov_y / screen_h.max(1.0)) / den_y * speed;
    if !dx.is_finite() || !dy.is_finite() {
        return (0.0, 0.0);
    }
    (dx, dy)
}

/// SunOne-style speed factor by true-pixel distance (continuous ramp).
/// Mirrors `aim.speed_curve` bit-exactly: ramp starts at `snap_px` with
/// `smin`, reaches `smax` at `near_px`; flat `smax` beyond.
pub fn speed_curve(
    dist_px: f64,
    snap_px: f64,
    near_px: f64,
    smin: f64,
    smax: f64,
    exp: f64,
) -> f64 {
    if !dist_px.is_finite() || dist_px < 0.0 {
        return smin;
    }
    let top = snap_px.max(0.0);
    if !near_px.is_finite() || near_px <= top || dist_px >= near_px {
        return smax;
    }
    let span = near_px - top;
    let t = if span > 0.0 {
        ((dist_px - top) / span).clamp(0.0, 1.0)
    } else {
        1.0
    };
    smin + (smax - smin) * (1.0 - (1.0 - t).powf(exp))
}

/// IoU of two `(x, y, w, h)` boxes. Mirrors `aim.mini_iou`.
pub fn mini_iou(a: [f64; 4], b: [f64; 4]) -> f64 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = (a[0] + a[2]).min(b[0] + b[2]);
    let y2 = (a[1] + a[3]).min(b[1] + b[3]);
    let w = (x2 - x1).max(0.0);
    let h = (y2 - y1).max(0.0);
    let inter = w * h;
    let ua = a[2] * a[3] + b[2] * b[3] - inter;
    if ua <= 1e-6 {
        return 0.0;
    }
    inter / ua
}

/// One MiniTrack row: `x1, y1, x2, y2, conf, cls` in pixels.
#[derive(Debug, Clone, Copy)]
pub struct MiniDet {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
    pub conf: f64,
    pub cls: f64,
}

#[derive(Debug, Clone)]
struct MiniState {
    id: i64,
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    px: f64,
    py: f64,
    vx: f64,
    vy: f64,
    cls: i64,
    conf: f64,
    hits: i64,
    missed: i64,
    last: f64,
}

/// SunOne-style multi-target tracker. Mirrors `aim.MiniTrack` field for
/// field (velocity-predicted greedy association, locked privileges).
/// Time `t` is explicit (no wall clock): deterministic and vector-tested.
#[derive(Debug, Clone)]
pub struct MiniTracker {
    tracks: Vec<MiniState>,
    next_id: i64,
    locked: Option<i64>,
    frame_w: f64,
    frame_h: f64,
    max_missed: i64,
    class_player: i64,
    last_ids: Vec<i64>,
}

impl MiniTracker {
    pub fn new(class_player: i64, max_missed: i64) -> Self {
        Self {
            tracks: Vec::new(),
            next_id: 1,
            locked: None,
            frame_w: 320.0,
            frame_h: 320.0,
            max_missed,
            class_player,
            last_ids: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.tracks.clear();
        self.next_id = 1;
        self.locked = None;
        self.last_ids.clear();
    }

    pub fn locked_id(&self) -> Option<i64> {
        self.locked
    }

    pub fn last_ids(&self) -> Vec<i64> {
        self.last_ids.clone()
    }

    fn allowed(&self, id: i64) -> i64 {
        self.max_missed + if Some(id) == self.locked { 8 } else { 0 }
    }

    fn match_cost(&self, t: &MiniState, bx: f64, by: f64, bw: f64, bh: f64,
                  cls: i64, px: f64, py: f64, now: f64) -> Option<f64> {
        if cls != t.cls {
            return None;
        }
        let dt = (now - t.last).clamp(1e-4, 0.25);
        let pred_cx = t.x + t.w * 0.5 + t.vx * dt;
        let pred_cy = t.y + t.h * 0.5 + t.vy * dt;
        let pred = [pred_cx - t.w * 0.5, pred_cy - t.h * 0.5, t.w, t.h];
        let det_cx = bx + bw * 0.5;
        let det_cy = by + bh * 0.5;
        let box_dist = (det_cx - pred_cx).hypot(det_cy - pred_cy);
        let pred_px = t.px + t.vx * dt;
        let pred_py = t.py + t.vy * dt;
        let pivot_dist = (px - pred_px).hypot(py - pred_py);
        let dist = box_dist.min(pivot_dist);
        let diag = t.w.hypot(t.h);
        let speed = t.vx.hypot(t.vy);
        let mut max_dist = 24.0f64.max(diag * 1.15 + 10.0)
            + speed * dt * (1.8 + t.missed as f64 * 0.35)
            + t.missed as f64 * 14.0f64.max(diag * 0.18);
        if Some(t.id) == self.locked {
            max_dist *= 1.6;
        }
        if dist > max_dist {
            return None;
        }
        let overlap = mini_iou(pred, [bx, by, bw, bh]);
        let cost = dist / max_dist + (1.0 - overlap) * 0.30
            + t.missed as f64 * 0.025 - t.hits.min(6) as f64 * 0.01
            - if Some(t.id) == self.locked { 0.10 } else { 0.0 };
        Some(cost)
    }

    /// Associate detections at time `t`. `last_ids` ends input-aligned.
    pub fn update(&mut self, dets: &[MiniDet], t: f64, keep_lock: bool,
                  frame_w: f64, frame_h: f64) {
        self.frame_w = frame_w;
        self.frame_h = frame_h;
        struct Filt {
            input: usize,
            x: f64, y: f64, w: f64, h: f64,
            cls: i64, conf: f64, px: f64, py: f64,
        }
        let mut filts: Vec<Filt> = Vec::new();
        for (ii, d) in dets.iter().enumerate() {
            if d.cls as i64 != self.class_player || d.x2 <= d.x1 || d.y2 <= d.y1 {
                continue;
            }
            let w = d.x2 - d.x1;
            let h = d.y2 - d.y1;
            filts.push(Filt {
                input: ii, x: d.x1, y: d.y1, w, h,
                cls: d.cls as i64, conf: d.conf,
                px: d.x1 + w * 0.5, py: d.y1 + h * 0.5,
            });
        }
        // Greedy assignment by (cost, track, det).
        let mut cands: Vec<(f64, usize, usize)> = Vec::new();
        for (ti, tr) in self.tracks.iter().enumerate() {
            for (di, f) in filts.iter().enumerate() {
                if let Some(c) = self.match_cost(tr, f.x, f.y, f.w, f.h,
                                                 f.cls, f.px, f.py, t) {
                    if c < 2.0 {
                        cands.push((c, ti, di));
                    }
                }
            }
        }
        cands.sort_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap()
                .then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2))
        });
        let mut track_hit = vec![-1i64; self.tracks.len()];
        let mut det_hit = vec![-1i64; filts.len()];
        for (_, ti, di) in &cands {
            if track_hit[*ti] < 0 && det_hit[*di] < 0 {
                track_hit[*ti] = *di as i64;
                det_hit[*di] = *ti as i64;
            }
        }
        let vmax = self.frame_w.max(self.frame_h) * 3.5;
        for (ti, tr) in self.tracks.iter_mut().enumerate() {
            let di = track_hit[ti];
            if di >= 0 {
                let f = &filts[di as usize];
                let dt = (t - tr.last).clamp(1e-4, 0.2);
                let mut rvx = (f.px - tr.px) / dt;
                let mut rvy = (f.py - tr.py) / dt;
                let spd = rvx.hypot(rvy);
                if spd > vmax && spd > 1e-4 {
                    let k = vmax / spd;
                    rvx *= k;
                    rvy *= k;
                }
                let blend = if Some(tr.id) == self.locked { 0.45 } else { 0.35 };
                tr.vx = tr.vx * (1.0 - blend) + rvx * blend;
                tr.vy = tr.vy * (1.0 - blend) + rvy * blend;
                tr.x = f.x;
                tr.y = f.y;
                tr.w = f.w;
                tr.h = f.h;
                tr.px = f.px;
                tr.py = f.py;
                tr.cls = f.cls;
                tr.conf = f.conf;
                tr.hits += 1;
                tr.missed = 0;
                tr.last = t;
            } else {
                let dt = (t - tr.last).clamp(0.0, 0.2);
                tr.x += tr.vx * dt;
                tr.y += tr.vy * dt;
                tr.px += tr.vx * dt;
                tr.py += tr.vy * dt;
                let decay = if Some(tr.id) == self.locked { 0.90 } else { 0.84 };
                tr.vx *= decay;
                tr.vy *= decay;
                tr.missed += 1;
                tr.last = t;
            }
        }
        self.last_ids = vec![-1; dets.len()];
        for (di, f) in filts.iter().enumerate() {
            let ti = det_hit[di];
            let fid = if ti >= 0 {
                self.tracks[ti as usize].id
            } else {
                let nid = self.next_id;
                self.next_id += 1;
                self.tracks.push(MiniState {
                    id: nid, x: f.x, y: f.y, w: f.w, h: f.h,
                    px: f.px, py: f.py, vx: 0.0, vy: 0.0,
                    cls: f.cls, conf: f.conf, hits: 1, missed: 0, last: t,
                });
                nid
            };
            self.last_ids[f.input] = fid;
        }
        let allowed: Vec<i64> = self.tracks.iter().map(|tr| self.allowed(tr.id)).collect();
        let mut kept = Vec::new();
        for (tr, a) in self.tracks.drain(..).zip(allowed.iter()) {
            if tr.missed <= *a {
                kept.push(tr);
            }
        }
        self.tracks = kept;
        if !self.tracks.iter().any(|tr| Some(tr.id) == self.locked) {
            self.locked = None;
        }
        if !keep_lock || self.locked.is_none() {
            let cx = self.frame_w * 0.5;
            let cy = self.frame_h * 0.5;
            let mut best: Option<usize> = None;
            let mut best_s = f64::INFINITY;
            for (ti, tr) in self.tracks.iter().enumerate() {
                if tr.missed > self.allowed(tr.id) {
                    continue;
                }
                let s = (tr.px - cx).hypot(tr.py - cy)
                    + tr.missed as f64 * 50.0 - tr.hits.min(5) as f64 * 4.0;
                if s < best_s {
                    best_s = s;
                    best = Some(ti);
                }
            }
            self.locked = best.map(|ti| self.tracks[ti].id);
        }
    }

    /// Returns `(x_mid, y_mid, h, tid, observed, missed, vx, vy)` or `None`.
    pub fn get_locked(&self) -> Option<(f64, f64, f64, i64, bool, i64, f64, f64)> {
        for tr in &self.tracks {
            if Some(tr.id) == self.locked {
                if tr.missed > self.allowed(tr.id) {
                    return None;
                }
                return Some((tr.x + tr.w * 0.5, tr.y + tr.h * 0.5, tr.h,
                             tr.id, tr.missed == 0, tr.missed, tr.vx, tr.vy));
            }
        }
        None
    }
}

/// CPython `round()` semantics: round-half-to-even.
fn round_half_even(x: f64) -> f64 {
    let f = x.floor();
    let d = x - f;
    if d < 0.5 {
        f
    } else if d > 0.5 {
        f + 1.0
    } else if (f as i64) % 2 == 0 {
        f
    } else {
        f + 1.0
    }
}

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Uniform float in `[lo, hi)` from an explicit xorshift64 state.
fn uniform(state: &mut u64, lo: f64, hi: f64) -> f64 {
    let u = (xorshift64(state) >> 11) as f64 / (1u64 << 53) as f64;
    lo + (hi - lo) * u
}

/// Humanize a raw `(dx, dy)` delta. Returns integer mouse move.
///
/// Mirrors `aim.smooth_move` with `jitter = 0` bit-exactly. When `jitter > 0`,
/// two noise draws in `[-jitter, jitter]` are added (dx first, then dy) using
/// a caller-seeded xorshift64 stream — same call order as Python, but a
/// different stream than CPython's Mersenne Twister by design.
pub fn smooth_move(
    dx: f64,
    dy: f64,
    smooth: f64,
    max_step: i64,
    deadzone: f64,
    jitter: f64,
    seed: u64,
) -> (i64, i64) {
    let mag = dx.hypot(dy);
    if mag < deadzone.max(0.0) {
        return (0, 0);
    }
    let s = smooth.max(1.0);
    let (mut ox, mut oy) = (dx / s, dy / s);
    let mag2 = ox.hypot(oy);
    let ms = max_step.max(1);
    if mag2 > ms as f64 {
        let k = ms as f64 / mag2;
        ox *= k;
        oy *= k;
    }
    if jitter != 0.0 {
        let mut st = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        ox += uniform(&mut st, -jitter, jitter);
        oy += uniform(&mut st, -jitter, jitter);
    }
    (round_half_even(ox) as i64, round_half_even(oy) as i64)
}

/// Full per-tick aim chain, mirroring `main_v8.py`:
/// select -> headshot offset -> `aaMovementAmp` -> [`smooth_move`].
/// Returns `(move, new_last)` where `move` is `None` when no target.
#[allow(clippy::too_many_arguments)]
pub fn aim_step(
    dets: &[Det],
    cw: f64,
    ch: f64,
    center: bool,
    last: Option<(f64, f64)>,
    sticky_px: f64,
    fov: Option<(f64, f64)>,
    max_box_h: f64,
    headshot_mode: bool,
    amp: f64,
    smooth: f64,
    max_step: i64,
    deadzone: f64,
) -> (Option<(i64, i64)>, Option<(f64, f64)>) {
    match select_target(dets, cw, ch, center, last, sticky_px, fov, max_box_h) {
        None => (None, None),
        Some((xm, ym, h)) => {
            let offset = headshot_offset(h, headshot_mode);
            let raw_dx = (xm - cw) * amp;
            let raw_dy = (ym - offset - ch) * amp;
            let mv = smooth_move(raw_dx, raw_dy, smooth, max_step, deadzone, 0.0, 0);
            (Some(mv), Some((xm, ym)))
        }
    }
}

/// 1D constant-velocity Kalman filter. Mirrors `aim.KalmanCV` bit-exactly:
/// plain f64 ops in the same order, no libm extras.
#[derive(Debug, Clone)]
pub struct KalmanCV {
    pub x: f64,
    pub v: f64,
    p00: f64,
    p01: f64,
    p11: f64,
    q: f64,
    damping: f64,
}

impl KalmanCV {
    pub fn new(x0: f64, p_pos: f64, p_vel: f64, q: f64, damping: f64) -> Self {
        Self { x: x0, v: 0.0, p00: p_pos, p01: 0.0, p11: p_vel, q, damping }
    }

    pub fn predict(&mut self, dt: f64) {
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        let dt4 = dt2 * dt2;
        self.v = self.v * (1.0 - self.damping);
        self.x = self.x + self.v * dt;
        let np00 =
            self.p00 + dt * (self.p01 + self.p01) + dt2 * self.p11 + self.q * dt4 / 4.0;
        let np01 = self.p01 + dt * self.p11 + self.q * dt3 / 2.0;
        let np11 = self.p11 + self.q * dt2;
        self.p00 = np00;
        self.p01 = np01;
        self.p11 = np11;
    }

    /// Returns false when gated as an outlier (state untouched).
    pub fn update(&mut self, z: f64, r: f64, gate: f64) -> bool {
        if gate != 0.0 && (z - self.x).abs() > gate {
            return false;
        }
        let s = self.p00 + r;
        let k0 = self.p00 / s;
        let k1 = self.p01 / s;
        let y = z - self.x;
        self.x = self.x + k0 * y;
        self.v = self.v + k1 * y;
        let (p00, p01) = (self.p00, self.p01);
        self.p00 = p00 - k0 * p00;
        self.p01 = p01 - k0 * p01;
        self.p11 = self.p11 - k1 * p01;
        true
    }

    pub fn ahead(&self, t: f64, vmax: f64) -> f64 {
        let v = self.v.clamp(-vmax, vmax);
        self.x + v * t
    }
}

/// Two independent [`KalmanCV`] axes. Mirrors `aim.Kalman2D`.
#[derive(Debug, Clone)]
pub struct Kalman2D {
    pub kx: KalmanCV,
    pub ky: KalmanCV,
}

impl Kalman2D {
    pub fn new(x0: f64, y0: f64, p_pos: f64, p_vel: f64, qx: f64, qy: f64, damping: f64) -> Self {
        Self {
            kx: KalmanCV::new(x0, p_pos, p_vel, qx, damping),
            ky: KalmanCV::new(y0, p_pos, p_vel, qy, damping),
        }
    }

    pub fn predict(&mut self, dt: f64) {
        self.kx.predict(dt);
        self.ky.predict(dt);
    }

    pub fn update(&mut self, zx: f64, zy: f64, r: f64, gate: f64) -> (bool, bool) {
        (self.kx.update(zx, r, gate), self.ky.update(zy, r, gate))
    }

    pub fn ahead(&self, t: f64, vmax: f64) -> (f64, f64) {
        (self.kx.ahead(t, vmax), self.ky.ahead(t, vmax))
    }
}

/// Lunar-style dynamic own-player rejection for third-person shoulder views.
/// Drops detections touching a shoulder-side edge or sitting in the bottom
/// shoulder corner. Mirrors `aim.reject_own`.
pub fn reject_own(
    dets: &[Det],
    w: f64,
    h: f64,
    edge_px: f64,
    corner: bool,
    left: bool,
    right: bool,
) -> Vec<bool> {
    dets.iter()
        .map(|d| {
            let own_left = left
                && (d.x1 < edge_px
                    || (corner && d.x1 < w * 0.2 && d.y2 > h * 0.8));
            let own_right = right
                && ((w - d.x2) < edge_px
                    || (corner && (w - d.x2) < w * 0.2 && d.y2 > h * 0.8));
            !(own_left || own_right)
        })
        .collect()
}
