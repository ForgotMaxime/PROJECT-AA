"""Probe MAKCU serial ports: send km.move and read the `>>>` reply.

Usage: python tools/makcu_probe.py [--baud 115200] [--move 10]
Tries COM3/COM4 (override with --ports COM5,COM6).
"""
import argparse
import sys
import time


def probe(port, baud, move=10, timeout=2.0):
    import serial
    try:
        ser = serial.Serial(port, baudrate=baud, timeout=timeout, write_timeout=2)
    except Exception as e:
        return f"{port}@{baud}: OPEN FAIL ({e})"
    try:
        ser.reset_input_buffer()
        cmd = f"km.move({move},0)\r".encode("ascii")
        ser.write(cmd)
        time.sleep(0.3)
        reply = ser.read(64)
        extra = ser.read(ser.in_waiting) if ser.in_waiting else b""
        return f"{port}@{baud}: wrote {cmd!r} reply={reply + extra!r}"
    except Exception as e:
        return f"{port}@{baud}: IO FAIL ({e})"
    finally:
        try:
            ser.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", default="COM3,COM4")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--move", type=int, default=10)
    args = ap.parse_args()
    for port in [p.strip() for p in args.ports.split(",")]:
        print(probe(port, args.baud, args.move), flush=True)


if __name__ == "__main__":
    sys.exit(main())
