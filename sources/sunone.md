# SunOne — sunone_aimbot (Python) + sunone_aimbot_2 (C++)

- https://github.com/SunOner/sunone_aimbot (Python, ~1k stars, MIT)
- https://github.com/SunOner/sunone_aimbot_2 (C++, ~730 stars, "FPS and TPS")
- YOLOv8/v10/v11/v12, TensorRT + DirectML, Kalman, WindMouse, overlay ImGui,
  KMBOX + MAKCU. V2 conseillée par l'auteur pour Windows au quotidien.

## Pris
- Rien en code (C++ / archi différente), mais la direction est validée :
  TensorRT FP16 + Kalman + MAKCU = notre stack actuelle.

## À prendre
- **Masque de profondeur** (réf `depth-anything-tensorrt` dans leur README) :
  ton corps = toujours le plus proche, ennemis plus loin, angle caméra
  indifférent. Vraie réponse multi-angles quand nos heuristiques satureront.
- **WindMouse** (réf `ben.land` dolmans) : mouvements courbes humanisés.
  Alternative/complément à nos segments MAKCU.
- Overlay ImGui de tuning (nous : DearPyGui `gui_live.py`, équivalent Python).
