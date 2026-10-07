# Computer Vision Zone Safety & Motion Monitoring

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-0097A7.svg)](https://developers.google.com/mediapipe)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent, real-time computer vision system for industrial machinery safety and zone monitoring. The application monitors designated polygonal regions of interest (ROIs), detects motion via background subtraction and morphological filtering, and performs human pose estimation to identify intrusions into hazardous zones.

---

## Architecture Overview

```mermaid
flowchart LR
    A[Video Source\nWebcam / RTSP / File] --> B[Threaded VideoStream]
    B --> C[Frame Resizing & Blur]
    C --> D[Zone ROI Masking]
    D --> E[Background Subtraction\nGMG / MOG2]
    E --> F[Morphological Filter\n& Contour Analysis]
    C --> G[MediaPipe Pose\nLandmark Estimation]
    F --> H[Intrusion & Motion Logic\nPoint-in-Polygon]
    G --> H
    H --> I[HUD Overlay & Telemetry]
    I --> J[Display GUI / File Writer]
```

---

## Features

- **Polygonal Zone Tracking (ROI)**: Define arbitrary multi-point zones (e.g. *Key Area*, *Secondary Area*) using normalized coordinates that automatically scale to any resolution.
- **Motion Detection**: Employs Gaussian Mixture Models / GMG background subtraction coupled with morphological opening and dilation to filter out noise.
- **Human Pose Estimation**: Integrates Google MediaPipe Pose to estimate operator body landmarks (center of hip mass) and accurately detect physical intrusions into safety boundaries.
- **Thread-Safe Low-Latency Capture**: Threaded video ingest isolates OpenCV frame fetching to prevent RTSP buffer delay and frame queuing.
- **HUD Telemetry & FPS Tracking**: Real-time rolling FPS counter and status banner displaying zone occupancy and alerts.
- **Headless & Production-Ready**: Supports headless execution for background servers and Docker containers, with video recording (`--record`).

---

## Repository Structure

```
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules for media, venvs, and cache
├── .dockerignore             # Docker build exclusion rules
├── Dockerfile                # Production-ready slim container definition
├── LICENSE                   # MIT License
├── README.md                 # Project documentation
├── requirements.txt          # Minimal, pinned Python dependencies
├── config.py                 # Central configuration (zones, thresholds, dimensions)
├── detector.py               # Motion detection & MediaPipe pose pipeline
├── fps.py                    # Rolling FPS measurement utility
├── video_stream.py           # Thread-safe RTSP/camera ingest handler
├── main.py                   # Unified CLI application entry point
├── area_detection.py         # Backward-compatible legacy launcher
└── mixer_area_detection.py   # Backward-compatible legacy launcher
```

---

## Quickstart

### 1. Prerequisites

- Python 3.9+
- Recommended: A dedicated virtual environment

```bash
# Clone the repository
git clone https://github.com/luisabaroa14/Computer-Vision.git
cd Computer-Vision

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` to configure your default video stream (RTSP URL, camera index, or video path).

---

## Usage

### Run with Local Webcam
```bash
python main.py --source 0
```

### Run with an RTSP IP Camera
```bash
python main.py --source "rtsp://username:password@192.168.1.100:554/live"
```

### Run with a Recorded Video File and Save Output
```bash
python main.py --source path/to/video.mp4 --record --output output_videos/result.avi
```

### Headless Mode (No GUI Window)
```bash
python main.py --source 0 --headless --record
```

### High-Performance Mode (Disable Pose Estimation)
```bash
python main.py --source 0 --no-pose
```

---

## CLI Options

| Argument | Shorthand | Default | Description |
| :--- | :--- | :--- | :--- |
| `--source` | `-s` | `0` (or `VIDEO_SOURCE`) | Video source: device index, video path, or RTSP URL |
| `--record` | `-r` | `false` | Enable output video recording |
| `--output` | `-o` | `output_videos/result.avi` | Path where recorded video will be saved |
| `--headless` | | `false` | Run without opening an OpenCV GUI window |
| `--no-pose` | | `false` | Disable MediaPipe pose estimation |
| `--width` | | `854` | Frame processing width |
| `--height` | | `480` | Frame processing height |

---

## Customizing Zones

Zones are defined in `config.py` using relative fractional coordinates `(x, y)` between `0.0` and `1.0`. This ensures zones remain accurate regardless of input stream resolution:

```python
ZoneDefinition(
    name="Hazard Zone",
    points_relative=[
        (0.80, 1.00),
        (0.25, 1.00),
        (0.25, 0.70),
        (0.80, 0.70),
    ],
    color_normal=(150, 200, 0),  # Green/Cyan when safe
    color_alert=(0, 0, 255),     # Red when movement/human detected
    min_contour_area=3000,
)
```

---

## Docker Deployment

Build the Docker image:

```bash
docker build -t cv-zone-monitor .
```

Run monitoring headlessly inside Docker and record outputs to host machine:

```bash
docker run --rm \
  -e VIDEO_SOURCE="rtsp://user:pass@192.168.1.100:554/live" \
  -v "$(pwd)/output_videos:/app/output_videos" \
  cv-zone-monitor --record --output output_videos/docker_run.avi
```

---

## License

This project is licensed under the [MIT License](LICENSE).
