# Legacy Entry Points

This directory contains legacy entry scripts (`area_detection.py` and `mixer_area_detection.py`) preserved for historical reference and backward compatibility.

For all modern usage and production deployments, please use `main.py` in the project root:

```bash
# Modern CLI usage
python main.py --source 0
python main.py --source "rtsp://<username>:<password>@<camera-ip>:554/live"
```
