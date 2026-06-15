"""AirAim entry point.

`--sandbox` runs the webcam hand-tracking loop and prints the detected
gesture state.
"""

import argparse

import tracker


def main():
    parser = argparse.ArgumentParser(description="AirAim — webcam mouse")
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="run hand tracking and print the gesture state, no mouse control",
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
    else:
        parser.error("no mode selected — try: python main.py --sandbox")


if __name__ == "__main__":
    main()
