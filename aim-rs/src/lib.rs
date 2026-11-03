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
