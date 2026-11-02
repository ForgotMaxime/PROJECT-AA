import pygetwindow
import time
import bettercam
from typing import Union

# Could be do with
# from config import *
# But we are writing it out for clarity for new devs
from config import screenShotHeight, screenShotWidth

def gameSelection(title_filter=None) -> (bettercam.BetterCam, int, Union[int, None]):
    # Selecting the correct game window
    try:
        videoGameWindows = pygetwindow.getAllWindows()
        if title_filter:
            needle = str(title_filter).lower()
            matches = [w for w in videoGameWindows if needle in (w.title or "").lower()]
            if not matches:
                print(f'No window matching "{title_filter}" found:')
                for w in videoGameWindows:
                    if (w.title or "").strip():
                        print(f"  - {w.title}")
                return None
            videoGameWindow = matches[0]
            print(f'Selected window: "{videoGameWindow.title}" (filter "{title_filter}")')
        else:
            print("=== All Windows ===")
            for index, window in enumerate(videoGameWindows):
                # only output the window if it has a meaningful title
                if window.title != "":
                    print("[{}]: {}".format(index, window.title))
            # have the user select the window they want
            try:
                userInput = int(input(
                    "Please enter the number corresponding to the window you'd like to select: "))
            except ValueError:
                print("You didn't enter a valid number. Please try again.")
                return
            # "save" that window as the chosen window for the rest of the script
            videoGameWindow = videoGameWindows[userInput]
    except Exception as e:
        print("Failed to select game window: {}".format(e))
        return None

    # Activate that Window
    activationRetries = 30
    activationSuccess = False
    while (activationRetries > 0):
        try:
            videoGameWindow.activate()
            activationSuccess = True
            break
        except pygetwindow.PyGetWindowException as we:
            print("Failed to activate game window: {}".format(str(we)))
            print("Trying again... (you should switch to the game now)")
        except Exception as e:
            print("Failed to activate game window: {}".format(str(e)))
            print("Read the relevant restrictions here: https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setforegroundwindow")
            activationSuccess = False
            activationRetries = 0
            break
        # wait a little bit before the next try
        time.sleep(3.0)
        activationRetries = activationRetries - 1
    # if we failed to activate the window then we'll be unable to send input to it
    # so just exit the script now
    if activationSuccess == False:
        return None
    print("Successfully activated the game window...")

    # Starting screenshoting engine
    left = ((videoGameWindow.left + videoGameWindow.right) // 2) - (screenShotWidth // 2)
    top = videoGameWindow.top + \
        (videoGameWindow.height - screenShotHeight) // 2
    right, bottom = left + screenShotWidth, top + screenShotHeight

    # Clamp region to the virtual screen (multi-monitor aware) so
    # negative/off-screen coords fail fast instead of breaking capture.
    try:
        import win32api as _w32
        vx, vy = _w32.GetSystemMetrics(76), _w32.GetSystemMetrics(77)
        vw, vh = _w32.GetSystemMetrics(78), _w32.GetSystemMetrics(79)
        left, top = max(vx, left), max(vy, top)
        right, bottom = min(vx + vw, right), min(vy + vh, bottom)
        if right - left < 64 or bottom - top < 64:
            print(f"Capture region too small after clamping: {(left, top, right, bottom)}")
            return None
    except Exception:
        pass

    region: tuple = (left, top, right, bottom)

    # Calculating the center Autoaim box
    cWidth: int = screenShotWidth // 2
    cHeight: int = screenShotHeight // 2

    print(region)

    camera = bettercam.create(region=region, output_color="BGRA", max_buffer_len=512)
    if camera is None:
        print("bettercam failed, trying dxcam fallback...")
        try:
            import dxcam
            camera = dxcam.create(region=region, output_color="BGRA", max_buffer_len=512)
        except Exception as e:
            print(f"dxcam fallback failed: {e}")
            camera = None
    if camera is None:
        print("Your Camera Failed! Check the region and GPU screen-capture support.")
        return
    camera.start(target_fps=120, video_mode=True)

    return camera, cWidth, cHeight