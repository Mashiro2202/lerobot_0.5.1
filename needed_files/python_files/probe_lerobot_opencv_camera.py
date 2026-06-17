import argparse
import time

import cv2

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--backend", type=int, default=200)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()

    cfg = OpenCVCameraConfig(
        index_or_path=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        fourcc=args.fourcc,
        backend=args.backend,
    )
    camera = OpenCVCamera(cfg)
    camera.connect(warmup=False)

    start = time.perf_counter()
    frames = 0
    last_shape = None

    try:
        while time.perf_counter() - start < args.seconds:
            frame = camera.read()
            frames += 1
            last_shape = frame.shape
            if args.show:
                cv2.imshow(args.device, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.disconnect()
        if args.show:
            cv2.destroyAllWindows()

    elapsed_s = time.perf_counter() - start
    measured_fps = frames / elapsed_s if elapsed_s > 0 else 0.0
    print(f"device={args.device}")
    print(f"requested={args.width}x{args.height} fps={args.fps} fourcc={args.fourcc} backend={args.backend}")
    print(f"frames={frames} elapsed_s={elapsed_s:.3f} measured_fps={measured_fps:.2f} shape={last_shape}")


if __name__ == "__main__":
    main()
