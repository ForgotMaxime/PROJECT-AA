import pygetwindow
import time
import bettercam
from typing import Union

# Could be do with
# from config import *
# But we are writing it out for clarity for new devs
from config import screenShotHeight, screenShotWidth

def capture_region(win, shot_w, shot_h, shift_x=0, shift_y=0,
                   crop_left=0, crop_right=0, crop_top=0, crop_bottom=0):
    """Compute the capture region and its center, multi-monitor aware.

    Shift moves the whole box (aim stays exact, see aaShiftX); crop shaves
    pixels off the edges (left/right/top/bottom). Returns ((l,t,r,b), cw, ch)
    or (None, 0, 0) when the region collapses.
    """
    left = ((win.left + win.right) // 2) - (shot_w // 2) + int(shift_x) + int(crop_left)
    top = win.top + (win.height - shot_h) // 2 + int(shift_y) + int(crop_top)
    right = left + shot_w - int(crop_left) - int(crop_right)
    bottom = top + shot_h - int(crop_top) - int(crop_bottom)

    # Clamp region to the virtual screen (multi-monitor aware) so
    # negative/off-screen coords fail fast instead of breaking capture.
    try:
        import win32api as _w32
        vx, vy = _w32.GetSystemMetrics(76), _w32.GetSystemMetrics(77)
        vw, vh = _w32.GetSystemMetrics(78), _w32.GetSystemMetrics(79)
        left, top = max(vx, left), max(vy, top)
        right, bottom = min(vx + vw, right), min(vy + vh, bottom)
    except Exception:
        pass
    if right - left < 64 or bottom - top < 64:
        return None, 0, 0
    return (left, top, right, bottom), (right - left) // 2, (bottom - top) // 2


def gameSelection(title_filter=None, shift_x=0, shift_y=0,
                  crop_left=0, crop_right=0, crop_top=0, crop_bottom=0) -> (bettercam.BetterCam, int, Union[int, None]):
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
    region, cWidth, cHeight = capture_region(
        videoGameWindow, screenShotWidth, screenShotHeight,
        shift_x, shift_y, crop_left, crop_right, crop_top, crop_bottom)
    if region is None:
        print("Capture region too small after clamping, exiting.")
        return None

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

    try:
        win_w = int(videoGameWindow.right - videoGameWindow.left)
        win_h = int(videoGameWindow.bottom - videoGameWindow.top)
    except Exception:
        win_w, win_h = 0, 0
    return camera, cWidth, cHeight, win_w, win_h