# Lunar (zeyad-mansour) — Fortnite, YOLOv5, archivé

- https://github.com/zeyad-mansour/lunar (~440 stars, GPL-3.0, archivé)
- `lib/aimbot.py` : FOV box centrée, nearest-to-crosshair, SendInput
  pixel-par-pixel interpolé, collecte de dataset en jeu.

## Pris
- **Rejet own-player géométrique** : `x1 < 15 or (x1 < W/5 and y2 > H/1.2)`
  → notre `aim.reject_own` (`aaOwnReject`, côtés configurables).
- Offset tête `h/2.7` ≈ nos bins 0.38 — nos chiffres tiennent.

## À prendre
- Rien d'autre (PyTorch Hub YOLOv5, mss, Windows-only : dépassé).
- Idée process : collecte auto de frames en jeu pour dataset (ils le font).
