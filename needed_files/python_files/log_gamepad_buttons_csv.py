#!/usr/bin/env python3
"""Log pygame gamepad button/axis state to CSV.

Use this on the robot machine to discover actual button ids for a controller.
Stop with Ctrl+C, or pass --duration seconds.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import pygame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record pygame gamepad state to CSV.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to gamepad_button_trace_<timestamp>.csv",
    )
    parser.add_argument("--hz", type=float, default=20.0, help="Polling frequency.")
    parser.add_argument("--duration", type=float, default=None, help="Optional duration in seconds.")
    parser.add_argument("--joystick", type=int, default=0, help="Joystick index.")
    return parser.parse_args()


def event_name(event_type: int) -> str:
    names = {
        pygame.JOYBUTTONDOWN: "JOYBUTTONDOWN",
        pygame.JOYBUTTONUP: "JOYBUTTONUP",
        pygame.JOYAXISMOTION: "JOYAXISMOTION",
        pygame.JOYHATMOTION: "JOYHATMOTION",
        pygame.JOYDEVICEADDED: "JOYDEVICEADDED",
        pygame.JOYDEVICEREMOVED: "JOYDEVICEREMOVED",
    }
    return names.get(event_type, str(event_type))


def main() -> None:
    args = parse_args()
    out_path = args.out
    if out_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(f"gamepad_button_trace_{stamp}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pygame.init()
    pygame.joystick.init()
    joystick_count = pygame.joystick.get_count()
    if joystick_count <= args.joystick:
        raise SystemExit(f"No joystick index {args.joystick}. Detected count: {joystick_count}")

    js = pygame.joystick.Joystick(args.joystick)
    js.init()

    print(f"Joystick: {js.get_name()}")
    print(f"Buttons: {js.get_numbuttons()} | Axes: {js.get_numaxes()} | Hats: {js.get_numhats()}")
    print(f"Writing CSV: {out_path}")
    print("Press buttons/triggers now. Stop with Ctrl+C.")

    fieldnames = [
        "wall_time",
        "elapsed_s",
        "event_type",
        "event_button",
        "event_axis",
        "event_value",
        "event_hat",
        "pressed_buttons",
        "axes",
        "hats",
    ]

    start = time.perf_counter()
    period = 1.0 / max(args.hz, 1e-6)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        try:
            while True:
                now = time.perf_counter()
                elapsed = now - start
                if args.duration is not None and elapsed >= args.duration:
                    break

                events = pygame.event.get()
                if not events:
                    events = [None]

                pressed_buttons = [i for i in range(js.get_numbuttons()) if js.get_button(i)]
                axes = [round(float(js.get_axis(i)), 4) for i in range(js.get_numaxes())]
                hats = [js.get_hat(i) for i in range(js.get_numhats())]

                for event in events:
                    row = {
                        "wall_time": datetime.now().isoformat(timespec="milliseconds"),
                        "elapsed_s": round(elapsed, 4),
                        "event_type": "",
                        "event_button": "",
                        "event_axis": "",
                        "event_value": "",
                        "event_hat": "",
                        "pressed_buttons": json.dumps(pressed_buttons),
                        "axes": json.dumps(axes),
                        "hats": json.dumps(hats),
                    }
                    if event is not None:
                        row["event_type"] = event_name(event.type)
                        row["event_button"] = getattr(event, "button", "")
                        row["event_axis"] = getattr(event, "axis", "")
                        row["event_value"] = round(float(getattr(event, "value", 0.0)), 4) if hasattr(event, "value") else ""
                        row["event_hat"] = getattr(event, "hat", "")

                    writer.writerow(row)

                print(f"buttons={pressed_buttons} axes={axes} hats={hats}", end="\r", flush=True)
                time.sleep(period)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            js.quit()
            pygame.joystick.quit()
            pygame.quit()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
