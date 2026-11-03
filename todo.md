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
- [x] Port Rust `aim-rs` + harnesses différentiels (71 vecteurs verts)
- [x] Diag live : `tools/diag_capture.py`, `tools/diag_shots.py`
- [x] Testé live Fortnite : ~50 CPS, infer ~8ms engine TRT (RTX 3060)

## À faire
- [x] Threading capture / inférence / souris (files drop-stale) — smoke test `tools/smoke_threads.py` vert
- [x] ORT CUDA (`onnxruntime-gpu[cuda,cudnn]` + `preload_dlls`, provider vérifié) — `yolo11s.onnx` ~18ms vs `.pt` ~27ms
- [x] Engine TensorRT FP16 320 (`yolo11s.engine`, TRT 11.4 + modelopt chain) — ~8ms end-to-end, défaut actuel
- [x] Mesure : `tools/diag_capture.py --track` (raws + track.jsonl) + `tools/measure_tracks.py` (switches/jitter/rate, rejouable par tracker) — A/B bytetrack vs ocsort outillé, verdict sur vraies séquences en attente
- [ ] OC-SORT (`inertia` 0.3) si les sauts dominent l'erreur restante
- [ ] Filtre 1€ sur la mesure si jitter résiduel à l'arrêt (seuil : std > 2px)
- [ ] ReID BoT-SORT seulement si swaps en multi-cible dominent
- [x] Test live MAKCU (COM4 @4M, carré OK) — backend `makcu` actif par défaut

## Écart techno (état des lieux)
- [ ] Overlay/GUI de tuning live (sliders, FOV box, toggles) — gap UX n°1 vs ImGui/DearPyGui
- [ ] Mouvements humanisés : Bézier/segments MAKCU, courbes type WindMouse
- [ ] Modèle fine-tuné Fortnite (hologrammes, skins) — plafond de précision actuel = COCO
- [ ] YOLOv12/pose (tester `yolo12s`, `yolo11s-pose` pour visée tête directe)
- [ ] Masque de profondeur (Depth-Anything) si les heuristiques anti-own-char saturent

## Écarté
- [ ] ~~SAHI temps réel, RT-DETR, `best.onnx` YOLOv5~~ — trop lent / mort
- [ ] ~~INT8, `torch.compile`~~ — avant P2, gains faibles vs risque

## Tout à la fin
- [ ] Linux (absent), INT8 (risqué), triggerbot/recoil (confort)
