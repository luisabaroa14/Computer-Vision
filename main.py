#!/usr/bin/env python3
"""Main entry point for computer vision zone monitoring system."""

import argparse
import os
import sys
from typing import Optional, Union
import cv2

from src import MonitorConfig, ZoneMotionDetector, FPS, VideoStream


def parse_source(source_str: str) -> Union[int, str]:
    """Convert integer source string (like '0') to int for webcam index."""
    if source_str.isdigit():
        return int(source_str)
    return source_str


def run_monitoring(
    source: Union[int, str],
    record: bool = False,
    output_path: str = "output_videos/monitoring_output.avi",
    headless: bool = False,
    enable_pose: bool = True,
    width: int = 854,
    height: int = 480,
) -> None:
    """Run real-time zone monitoring pipeline."""
    # Build configuration
    cfg = MonitorConfig(
        frame_width=width,
        frame_height=height,
        enable_pose=enable_pose,
        record_output=record,
        output_video_path=output_path,
    )

    print(f"[INFO] Initializing video stream from: {source}")
    stream = VideoStream(src=source).start()

    # Video writer setup if recording is enabled
    writer: Optional[cv2.VideoWriter] = None
    if record:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path, fourcc, cfg.output_fps, cfg.frame_size)
        print(f"[INFO] Recording active -> saving to {output_path}")

    detector = ZoneMotionDetector(cfg)
    fps = FPS().start()

    print("[INFO] Monitoring started. Press 'q' in video window or Ctrl+C to terminate.")

    try:
        while True:
            grabbed, frame = stream.read()
            if not grabbed or frame is None:
                print("[INFO] Stream ended or connection lost.")
                break

            # Resize frame to target processing dimensions
            frame_resized = cv2.resize(frame, cfg.frame_size)

            # Analyze zones for movement and human occupancy
            zone_statuses = detector.analyze_frame(frame_resized)

            # Generate overlay visualization
            annotated_frame = detector.draw_overlays(
                frame_resized,
                zone_statuses,
                fps_value=fps.fps
            )

            # Write to output file if recording
            if writer is not None:
                writer.write(annotated_frame)

            # Display GUI window unless headless
            if not headless:
                cv2.imshow("Zone Safety Monitoring", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            fps.update()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupt received, shutting down gracefully...")
    finally:
        fps.stop()
        stream.release()
        if writer is not None:
            writer.release()
        detector.release()
        if not headless:
            cv2.destroyAllWindows()

        print("=" * 45)
        print(f"[INFO] Execution summary:")
        print(f"       Duration : {fps.elapsed():.2f} seconds")
        print(f"       Frames   : {fps._num_frames}")
        print(f"       Mean FPS : {fps.mean_fps():.2f}")
        print("=" * 45)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Computer Vision Zone & Movement Monitoring System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s", "--source",
        type=str,
        default=os.getenv("VIDEO_SOURCE", "0"),
        help="Video source: camera index (e.g. 0), video file path, or RTSP stream URL.",
    )
    parser.add_argument(
        "-r", "--record",
        action="store_true",
        default=os.getenv("RECORD_OUTPUT", "false").lower() in ("true", "1", "yes"),
        help="Record annotated output video to disk.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=os.getenv("OUTPUT_VIDEO_PATH", "output_videos/result.avi"),
        help="Output file path when --record is specified.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying GUI window (ideal for Docker and background services).",
    )
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="Disable MediaPipe pose detection to maximize throughput.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=int(os.getenv("FRAME_WIDTH", "854")),
        help="Frame processing width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=int(os.getenv("FRAME_HEIGHT", "480")),
        help="Frame processing height.",
    )

    args = parser.parse_args()

    run_monitoring(
        source=parse_source(args.source),
        record=args.record,
        output_path=args.output,
        headless=args.headless,
        enable_pose=not args.no_pose,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
