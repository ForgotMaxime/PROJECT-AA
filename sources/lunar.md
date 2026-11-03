# Lunar (zeyad-mansour) — Fortnite, YOLOv5, archivé

- https://github.com/zeyad-mansour/lunar (~440 stars, GPL-3.0, archivé)
- `lib/aimbot.py` : FOV box centrée, nearest-to-crosshair, SendInput
  pixel-par-pixel interpolé, collecte de dataset en jeu.

## Pris
- **Rejet own-player géométrique** : `x1 < 15 or (x1 < W/5 and y2 > H/1.2)`
  → notre `aim.reject_own` (`aaOwnReject`, côtés configurables).
- Offset tête `h/2.7` ≈ nos bins 0.38 — nos chiffres tiennent.

## Idées lues (non reprises en l'état)
- **Mouvements interpolés pixel-par-pixel** (`interpolate_coordinates_from_center`,
  diffusion d'erreur + busy-sleep précis) : l'équivalent moderne côté MAKCU,
  c'est les segments/Bézier natifs `km.move(dx,dy,seg,...)` → voir todo.
- **Calibration sensibilité** (`setup()` : `xy_scale = 10/xy_sens`) : notre
  `aaMovementAmp` joue déjà ce rôle empiriquement, pas de knob en plus.
- **Mode collecte dataset** (`collect_data`, exclut les frames own-player) :
  équivalent à nos raws `diag_capture.py` pour un futur fine-tune.
- État `LOCKED` ±5px : indicateur, pas de logique — nos visuals suffisent.

## Dépassé, à ne pas reprendre
- Stack 2022 : torch 1.11+cu113, `torch.hub` YOLOv5, capture `mss`,
  `best.pt` Fortnite d'époque (même famille que notre `best.onnx` mort).
- `pygame`/`pynput`/`termcolor` : remplacés chez nous (cv2, win32api, DearPyGui).
