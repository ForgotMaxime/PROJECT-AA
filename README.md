# AI Aimbot (fork)

> Upstream `RootKit-Org/AI-Aimbot` was archived by the owner on 2025-10-09 and is read-only.
> This copy (`PROJECT AA`) is a local fork for maintenance. No upstream PR will be merged.

![banner](imgs/banner.png)

Visual-only AI aimbot for games with humanoid characters, based on YOLOv5.
Educational purpose: demonstrate vulnerability of games to AI-driven cheats.

**Use at your own risk. Bans and manual reports are possible.**
Known reports: Splitgate, EQU8 (win32 mouse movement detection).

## Versions

| Script | Backend | Notes |
|---|---|---|
| `legacy/main.py` | PyTorch (`torch.hub yolov5s`, legacy) | Archived upstream code, run as `python legacy/main.py` |
| `legacy/main_onnx.py` | ONNX Runtime (legacy v5) | See `config.py:onnxChoice` (1=CPU, 2=AMD/NVIDIA, 3=NVIDIA) |
| `legacy/main_tensorrt.py` | TensorRT (legacy v5) | NVIDIA only, hardest setup |
| `main_v8.py` | Ultralytics YOLO `.pt` / `.onnx` / `.engine` | Default path, CUDA12. See below |

Entry point legacy: `legacy/main.py:17 main()` loads `yolov5s` via `torch.hub`, captures with `gameSelection.py`, moves mouse via `win32api`.

## Requirements (upstream pin)

- Python 3.11 (upstream target; 3.14 installs but untested long-run)
- NVIDIA CUDA Toolkit 11.8 (NVIDIA path only)
- PyTorch NVIDIA: `pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118`
- PyTorch AMD/CPU: `pip install torch torchvision torchaudio`
- `pip install -r requirements.txt` (see `requirements.txt`, `config.py`)

Conda variant: see `Conda/setup-conda-env.md`, `Conda/setup-requirements.md`.
Extras (Streamlit demos only): `Conda/additionalRequirements.txt`.

TensorRT path (condensed, NVIDIA only): CUDA 11.8 + cuDNN 8.9.6 + TensorRT 8.6 GA,
`pip install <cuda>/python/tensorrt-8.6.1-cp311-none-win_amd64.whl`,
set `CUDA\v11.8\lib`, `libnvvp`, `bin` in PATH, then e.g.
`python legacy/export.py --weights ./yolov5s.pt --include engine --half --imgsz 320 320 --device 0`.
Full original 12-step guide removed as outdated; expect adaptation for CUDA 12 / Python >3.11.

## Run

```powershell
cd "PROJECT AA"
python main_v8.py        # live (TensorRT engine + ByteTrack + Kalman)
python main_v8.py --window Fortnite   # select by title, no index race
python main_v8.py --window Fortnite --gui  # + live DearPyGui tuning panel
```

- `CAPS_LOCK` toggles aim (starts OFF).
- `Q` (`config.py:aaQuitKey`) quits. `visuals=True` shows OpenCV feed, `q` in feed window exits.

## YOLOv8 path (`main_v8.py`, `inference_v8.py`)

- Model auto-downloads `weights/yolo11s.pt` on first run (git-ignored, 18MB). Default
  runtime is `weights/yolo11s.engine` (git-ignored, ~47MB, TRT FP16 ~8ms end-to-end vs
  ORT ~18ms vs torch ~27ms, track mode 320px RTX 3060).
- TensorRT: `python export_v8.py weights/yolo11s.pt 320 engine` (needs `tensorrt-cu12` +
  build deps `nvidia-modelopt onnx-graphsurgeon onnxscript lief polygraphy`;
  ~4 min build, GPU-specific file). Fallbacks: `weights/yolo11s.onnx` then `weights/yolo11s.pt`.
- TensorRT: `python export_v8.py weights/yolo11s.pt 320 engine` needs the TensorRT 10.x python package (not installed here); `inference_v8.py`/`main_v8.py` already accept `v8Model = "weights/yolo11s.engine"`.
- Tracking: `v8Track` (ByteTrack IDs via `trackers/bytetrack-aim.yaml`, `persist=True`) + `select_locked` ID lock in `aim.py` (`aaTrackMinConf`, `aaTrackBuffer`, `aaRecoverPx` spatial continuity over ID on disagreement); falls back to detect-only if the tracker fails. State: `locked_id` + miss counter, unlock past buffer. `aaLockHold` ticks of coast-aim on the locked prediction before switching targets. `aaMonoTarget`: one active target while CAPS is held (hold, then committed switch — never idle); release + press to re-acquire.
- Config (`config.py`): `v8Model` (`weights/yolo11s.engine`), `v8Imgsz` (320), `v8Conf` (0.25), `v8Iou` (0.5), `v8Device` (0=cuda:0, `"cpu"`=CPU, `"auto"`), `aaMouseBackend` (`makcu` live, `sendinput`, `win32` legacy).
- Mouse (`mouse.py`): `sendinput` avoids the `mouse_event` hook path flagged by EQU8; still user-mode, not a driver.
- MAKCU: `config.py:aaMouseBackend = "makcu"`, `aaMakcuPort` (`"COM4"` here), `aaMakcuBaud` (4000000 here; 115200 stock, toggle: press its button 4x in 2s). Sends `km.move(dx,dy)\r` per tick, zero moves skipped, handle closed on exit. Needs CH343 driver + AIO/test tool closed. Verified live (square test + in-game smooth).
- Capture (`gameSelection.py`): region clamped to monitor, `bettercam` with `dxcam` fallback.
- Measured RTX 3060, 320px track mode: `yolo11s.engine` TRT FP16 ~8ms end-to-end
  (infer 1.3ms), `yolo11s.onnx` ORT CUDA ~18ms, `yolo11s.pt` torch ~27ms.
  ORT needs the CUDA13/cuDNN9 pip DLLs or it falls back to ~240ms CPU.

## Config (`config.py`)

- `screenShotHeight/Width` (default 320): capture box around screen center
- `useMask`, `maskSide` (`left`/`right`), `maskWidth`, `maskHeight`
- `aaMovementAmp` (default 0.7): legacy scalar px→counts; trims FOV mode as speed when enabled
- `aaGameFovX/Y` (Fortnite: 80/50.5) : conversion physique px→counts **en dernier étage** (`aim.fov_to_counts`, SunOne-style mais corrigée fenêtre réelle, pas capture — deadzone/smooth/maxstep restent en px) ; `aaSens` en % natifs du jeu (axe X/Y, ici 2.1), `aaYaw`/`aaPitch` Fortnite 0.005555, `aaScreenW/H` (0 = auto fenêtre), `aaFovSpeed` 0.25 (gain de boucle stable ~19%/tick ; 1.0 = 100%/tick instable avec le retard boucle). 0 = legacy amp. GUI-live : sliders fov x/y + sens + fov speed.
- `confidence` (default 0.4)
- `headshot_mode` (default True: 0.38 box height offset, else 0.2), or direct `aaAimOffset` fraction (0.0 = exact box center, takes precedence; `None` = use `headshot_mode`)
- `centerOfScreen` (default True): pick closest-to-center target
- `cpsDisplay`, `visuals`, `aaQuitKey`, `onnxChoice`
- `aaMouseBackend` (`sendinput`/`win32`/`makcu`), `aaMakcuPort`, `aaMakcuBaud`
- Own-character rejection (3rd person): `aaIgnoreRects` (static zones), `aaMaxBoxH` (size), Lunar-style `aaOwnReject`/`aaOwnEdgePx`/`aaOwnCorner`/`aaOwnSides` (dynamic edge/corner rule, rotation-proof) + `aaShiftX/aaShiftY` + `aaCropLeft/Right/Top/Bottom`
- v8 aim feel (`aim.py`, `main_v8.py`): `aaSmooth` (divisor, 1=instant), `aaMaxStep` (px/tick clamp), `aaDeadzone`, `aaJitter`, `aaStickyPx` (target lock radius), `aaFovX/aaFovY` (0=off), `aaHoldKey` (default CAPS_LOCK), `aaMouseBackend`, `aaDzEnter/aaDzExit` (deadzone hysteresis), headshot offset by box-height bins + EMA, Kalman CV prediction + lead (`aaPredict`, `aaLeadT`, `aaLeadMax`, `aaGatePx`, `aaKalmanQX/QY/R`, `aaCoastMax`, `aaLockHold`, velocity `aaKalmanDamping` 0.08 + `aaKalmanWarmup` 2 + drop after `aaGateReset` 3 double-gated updates; yellow diamond in visuals)

## Perf reference (upstream)

100-150 CPS on Ryzen 7 2700 / 64GB DDR4 / RTX 3080. Reboot on repeated CUDA errors.

## Custom content

- `customModels/*/README.md`: community weights (Fortnite, Rust, Roblox/Phantom Forces, yolov5n160, fal3). Check per-model setup note before use.
- `customScripts/*/README.md`: variants (AimAssist AMD perf, Tector101 DXCam follow, Villageslayer keybinds, yolov8_live_overlay). Each has its own requirements; not all maintained.

## Rust port + differential testing (`aim-rs/`)

- `aim-rs/src/lib.rs`: port of `aim.py` (`select_target`, `smooth_move`, full-tick `aim_step`). Banker's rounding replicated to match CPython `round()`; jitter uses an explicit seeded xorshift stream (different from CPython's MT by design — deterministic core tested differentially, jitter via contract test).
- Harness 1 (unit): `tools/dump_vectors.py` generates 81 vectors from `aim.py` into `aim-rs/tests/vectors/vectors.json`; `cargo test` replays them with exact comparison.
- Harness 2 (integration): `chain/*` vectors cover the full `dets -> dx/dy` tick (select + headshot offset + amp + smooth), mirroring `main_v8.py`.
- Regen + test: `python tools/dump_vectors.py` then `cargo test --manifest-path aim-rs/Cargo.toml`.

## Docs removed in this fork

- `Conda/00`, `01`, `03`: generic Conda/git tutorials (not project-specific).
- `utils/*/README.md`, `ultralytics1/utils/*/README.md`: vendored YOLOv5 logger/API docs (ClearML, Comet, WandB, Flask) duplicated in both trees.
- Lowercase `readme.md` normalized to `README.md`.

Videos (archived, outdated): tutorial `https://www.youtube.com/watch?v=TCJHLbbeLhg`, stream `https://www.youtube.com/watch?v=uniL5yR7y0M`.

License: GPL-3.0, see `LICENSE`.
