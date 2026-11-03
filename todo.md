# TODO — PROJECT AA

## Fait
- [x] Fork + cleanup docs (README réécrit, Conda condensé, `readme.md` normalisés)
- [x] Inférence YOLOv8 → yolo11s, CUDA12, FP32 (FP16 mesuré sans gain)
- [x] Export ONNX / code-ready TensorRT (`export_v8.py`)
- [x] Backends souris : MAKCU série (live, défaut), sendinput, win32 legacy
- [x] Capture : clamp virtual-screen multi-moniteurs, fallback dxcam, `--window`
- [x] Aim feel : smooth, maxstep, deadzone + hystérésis, sticky, FOV, hold key, auto device
- [x] Anti-own-char : ignore rects, max box height, shift X/Y (ex-aaRightShift), crop asymétrique
- [x] ByteTrack + verrou ID + hold d'engagement + mode mono-cible
- [x] Kalman CV + lead + gating + coast (qx/qy séparés)
- [x] Port Rust `aim-rs` + harnesses différentiels (60 vecteurs verts)
- [x] Diag live : `tools/diag_capture.py`, `tools/diag_shots.py`
- [x] Testé live Fortnite : ~35 CPS, infer ~25ms torch puis ~13ms ORT (RTX 3060)

## À faire
- [x] Threading capture / inférence / souris (files drop-stale) — smoke test `tools/smoke_threads.py` vert
- [x] ORT CUDA (`onnxruntime-gpu[cuda,cudnn]` + `preload_dlls`, provider vérifié) — `yolo11s.onnx` ~18ms vs `.pt` ~27ms
- [x] Engine TensorRT FP16 320 (`yolo11s.engine`, TRT 11.4 + modelopt chain) — ~8ms end-to-end, défaut actuel
- [ ] Mesure specialistion : ID switches/min en crossing, RMSE strafe vs saut (frames enregistrées)
- [ ] OC-SORT (`inertia` 0.3) si les sauts dominent l'erreur restante
- [ ] Filtre 1€ sur la mesure si jitter résiduel à l'arrêt (seuil : std > 2px)
- [ ] ReID BoT-SORT seulement si swaps en multi-cible dominent
- [x] Test live MAKCU (COM4 @4M, carré OK) — backend `makcu` actif par défaut

## Écarté
- [ ] ~~SAHI temps réel, RT-DETR, `best.onnx` YOLOv5~~ — trop lent / mort
- [ ] ~~INT8, `torch.compile`~~ — avant P2, gains faibles vs risque
