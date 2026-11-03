# Portion of screen to be captured (This forms a square/rectangle around the center of screen)
screenShotHeight = 320
screenShotWidth = 320

# Use "left" or "right" for the mask side depending on where the interfering object is, useful for 3rd player models or large guns
useMask = False
maskSide = "left"
maskWidth = 80
maskHeight = 200

# Autoaim mouse movement amplifier
aaMovementAmp = .7

# Person Class Confidence
confidence = 0.4

# What key to press to quit and shutdown the autoaim
aaQuitKey = "P"

# If you want to main slightly upwards towards the head
headshot_mode = True

# Displays the Corrections per second in the terminal
cpsDisplay = True

# Set to True if you want to get the visuals
visuals = True

# Smarter selection of people
centerOfScreen = True

# ONNX ONLY - Choose 1 of the 3 below
# 1 - CPU
# 2 - AMD
# 3 - NVIDIA
onnxChoice = 1

# YOLOv8 (main_v8.py) - .pt, .onnx or .engine, device 0 = cuda:0, "cpu" = CPU
v8Model = "yolo11s.pt"
v8Imgsz = 320
v8Conf = 0.25
v8Iou = 0.5
v8Device = 0

# ByteTrack ID lock (main_v8.py). Off = sticky heuristic only.
v8Track = True
v8Tracker = "trackers/bytetrack-aim.yaml"
aaTrackMinConf = 0.3
aaTrackBuffer = 10

# Mouse backend for main_v8.py: "sendinput" (default), "win32" (legacy) or "makcu"
aaMouseBackend = "sendinput"

# MAKCU serial bridge (only used when aaMouseBackend = "makcu")
aaMakcuPort = "COM3"
aaMakcuBaud = 115200

# Aim feel (main_v8.py): smooth divisor, per-tick clamp, deadzone, jitter
aaSmooth = 1.3
aaMaxStep = 60
aaDeadzone = 1
aaJitter = 0

# Deadzone hysteresis (px): start moving above enter, stop below exit.
aaDzEnter = 2.5
aaDzExit = 1.0

# Target prediction (Kalman CV + lead). Off = aim at raw midpoint.
aaPredict = True
aaLeadT = 0.05
aaLeadMax = 30.0
aaGatePx = 60.0
aaKalmanQ = 4e6
aaKalmanR = 50.0
aaCoastMax = 5

# Target lock: keep last target within this radius, FOV box half-extents (0 = off)
aaStickyPx = 80
aaFovX = 0
aaFovY = 0

# Key held to aim (0x14 = CAPS_LOCK)
aaHoldKey = 0x14

# Pixel rects (x, y, w, h) blacked out before inference: own 3rd-person model, HUD.
# Tune with tools/diag_capture.py frames (magenta outline). [] = off.
aaIgnoreRects = [(0, 210, 50, 110)]

# Crop the capture box edges (px): trims HUD/edge junk instead of blacking out.
aaCropLeft = 20
aaCropRight = 40
aaCropTop = 0
aaCropBottom = 0

# Ignore detections taller than this (own close-up body/gun). 0 = off.
aaMaxBoxH = 160

# Aim shift (ex-aaRightShift, Fortnite 3rd person): shifts the capture region
# AND the aim point by the same amount, so aim math stays exact while the
# selection center moves away from your own shoulder model. 0 = off.
# Positive X moves selection right (own char on the left).
aaShiftX = 60
aaShiftY = 0