# RN_AI (ReksarGames) — FPS and TPS, Python + DearPyGui

- https://github.com/ReksarGames/RN_AI (~50 stars, MIT)
- Kalman, deadzone, capture multi-sources (écran/OBS/carte), classes avec
  aim positions par classe, support Makcu avec masquage, PID + courbes humaines.

## Pris (idées)
- **Lock par IDs quantifiés grille 10px** (`_make_lock_id`) : robuste aux swaps
  de tracker. Foldé dans notre `select_locked` via continuité spatiale
  (`aaRecoverPx`) plutôt que grille — même effet, sans nouvel état.
- **Aim positions par classe** : offset vertical différent par classe de cible.
  À garder pour le fine-tune (modèle multi-classes).

## À prendre
- Capture multi-sources (OBS/carte) pour setup 2PC.
- Smoothing temporel des petites cibles (notre EMA/Kalman couvre déjà).
