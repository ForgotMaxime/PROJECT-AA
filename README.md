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
| `main.py` | PyTorch (`torch.hub yolov5s`) | Easiest, any machine |
| `main_onnx.py` | ONNX Runtime | See `config.py:onnxChoice` (1=CPU, 2=AMD/NVIDIA, 3=NVIDIA) |
| `main_tensorrt.py` | TensorRT | NVIDIA only, hardest setup |

Entry point: `main.py:17 main()` loads `yolov5s` via `torch.hub`, captures with `gameSelection.py`, moves mouse via `win32api`.

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
`python ./export.py --weights ./yolov5s.pt --include engine --half --imgsz 320 320 --device 0`.
Full original 12-step guide removed as outdated; expect adaptation for CUDA 12 / Python >3.11.

## Run

```powershell
cd "PROJECT AA"
python main.py          # Fast
python main_onnx.py     # Faster
python main_tensorrt.py # Fastest (NVIDIA)
```

- `CAPS_LOCK` toggles aim (starts OFF).
- `Q` (`config.py:aaQuitKey`) quits. `visuals=True` shows OpenCV feed, `q` in feed window exits.

## Config (`config.py`)

- `screenShotHeight/Width` (default 320): capture box around screen center
- `useMask`, `maskSide` (`left`/`right`), `maskWidth`, `maskHeight`
- `aaMovementAmp` (default 0.4, try 0.5-2): mouse amplification
- `confidence` (default 0.4)
- `headshot_mode` (default True: 0.38 box height offset, else 0.2)
- `centerOfScreen` (default True): pick closest-to-center target
- `cpsDisplay`, `visuals`, `aaQuitKey`, `onnxChoice`

## Perf reference (upstream)

100-150 CPS on Ryzen 7 2700 / 64GB DDR4 / RTX 3080. Reboot on repeated CUDA errors.

## Custom content

- `customModels/*/README.md`: community weights (Fortnite, Rust, Roblox/Phantom Forces, yolov5n160, fal3). Check per-model setup note before use.
- `customScripts/*/README.md`: variants (AimAssist AMD perf, Tector101 DXCam follow, Villageslayer keybinds, yolov8_live_overlay). Each has its own requirements; not all maintained.

## Docs removed in this fork

- `Conda/00`, `01`, `03`: generic Conda/git tutorials (not project-specific).
- `utils/*/README.md`, `ultralytics1/utils/*/README.md`: vendored YOLOv5 logger/API docs (ClearML, Comet, WandB, Flask) duplicated in both trees.
- Lowercase `readme.md` normalized to `README.md`.

Videos (archived, outdated): tutorial `https://www.youtube.com/watch?v=TCJHLbbeLhg`, stream `https://www.youtube.com/watch?v=uniL5yR7y0M`.

License: GPL-3.0, see `LICENSE`.
