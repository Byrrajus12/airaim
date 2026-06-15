"""AirAim entry point.

`--sandbox` runs the webcam hand-tracking loop and prints the detected
gesture state. `--cursor` moves a fake on-screen circle with your fingertip.
"""

import argparse

import cursor
import tracker


def main():
    parser = argparse.ArgumentParser(description="AirAim — webcam mouse")
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="run hand tracking and print the gesture state, no mouse control",
    )
    parser.add_argument(
        "--cursor",
        action="store_true",
        help="move a fake circle on a blank window with your fingertip",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="camera index (default 0)",
    )
    args = parser.parse_args()

    if args.sandbox:
        tracker.run_sandbox(camera_index=args.camera)
    elif args.cursor:
        cursor.run_cursor_sandbox(camera_index=args.camera)
    else:
        parser.error("no mode selected — try: python main.py --sandbox or --cursor")


if __name__ == "__main__":
    main()
