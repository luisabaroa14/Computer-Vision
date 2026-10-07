#!/usr/bin/env python3
"""Mixer Area Detection - Legacy entry point forwarded to the modular monitoring engine."""

import os
from main import run_monitoring, parse_source


def main():
    # Retrieve video source from environment or fallback to local sample video
    default_source = os.getenv("VIDEO_SOURCE", "Mezcladora2.MOV")
    source = parse_source(default_source)

    run_monitoring(
        source=source,
        record=False,
        headless=False,
        enable_pose=True,
    )


if __name__ == "__main__":
    main()
