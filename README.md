# ARGUS — Cup Defect Detection System

Automated defect logging for a cup production line. Two USB cameras watch each cup as it passes through the stamping machine. A PatchCore anomaly detector flags defects in real time, saves photos, and serves a web dashboard with shift statistics and Excel export.

**No hardware trigger required.** Motion detection via MOG2 background subtraction detects when the drum stops — no modifications to the machine.

---

## Features

- **Motion-triggered burst capture** — 5 frames per camera while the cup spins from inertia, covering ~180° of surface
- **Anomaly detection** — PatchCore (anomalib) trained only on good cups, no defect labeling needed
- **Automatic threshold tuning** — ROC curve on validation set, threshold selected at max F1
- **Web dashboard** — shift statistics, hourly chart, defect table with thumbnails
- **Per-cup detail page** — both camera frames + anomaly heatmap overlay
- **Excel export** — one-click report with defective rows highlighted red
- **Offline replay** — run the full pipeline on recorded video without cameras

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│ Docker container                                │
│                                                 │
│  Review UI  (FastAPI + Jinja2, localhost:8000)  │
│       │                                         │
│  Storage (SQLite WAL + JPEG files on disk)      │
│       │                                         │
│  Orchestrator (asyncio)                         │
│   ├── CameraWorker × 2  (OpenCV + MOG2 FSM)     │
│   └── DefectDetector    (anomalib PatchCore)    │
└─────────────────────────────────────────────────┘
         │ USB passthrough
      Cam1   Cam2
```

| Module | Responsibility |
|---|---|
| `argus/camera.py` | Capture loop, MOG2 motion FSM, burst collection |
| `argus/pipeline.py` | asyncio orchestrator, pairs events from two cameras |
| `argus/detector.py` | PatchCore inference, returns score + anomaly map |
| `argus/storage.py` | SQLite writes + JPEG save |
| `argus/calibrate.py` | Train PatchCore, build ROC, save threshold |
| `argus/motion_fsm.py` | FSM: `WAITING → DRUM_STOP → BURST → COOLDOWN` |
| `argus/ui/app.py` | FastAPI routes: `/`, `/cup/{id}`, `/export`, `/image` |

---

## Quick Start (Docker)

**Requirements:** Docker, docker-compose, two USB cameras at `/dev/video0` and `/dev/video1`.

```bash
git clone https://github.com/Revenant121/Argus.git
cd Argus
docker compose up --build
```

UI available at [http://localhost:8000](http://localhost:8000)

---

## CLI Commands

All commands run inside the container:

```bash
docker compose run --rm argus <command>
```

| Command | Description |
|---|---|
| `argus run` | Start live detection (requires trained model) |
| `argus calibrate` | Train PatchCore on normal frames, tune threshold |
| `argus ui` | Launch web dashboard |
| `argus extract-video VIDEO` | Extract burst frames from recorded video |
| `argus replay VIDEO_CAM1 [VIDEO_CAM2]` | Run pipeline on pre-recorded video |
| `argus diagnose-motion VIDEO` | Analyse motion scores, suggest FSM thresholds |
| `argus doctor` | Self-check: cameras, paths, dependencies |

### Examples

```bash
# Extract normal frames from recorded video for training
argus extract-video line_cam1.mp4 --cam cam1 --out /data/raw/good/cam1

# Analyse motion to calibrate FSM thresholds
argus diagnose-motion line_cam1.mp4 --plot motion.png

# Train the detector
argus calibrate --normal /data/raw/good/cam1 --abnormal /data/raw/defect/cam1

# Run live
argus run
```

---

## Workflow

### Step 1 — Collect data on-site

Record 5–10 min of normal production video per camera:

```bash
ffmpeg -i /dev/video0 -t 600 line_cam1.mp4
```

Then extract burst frames:

```bash
argus extract-video line_cam1.mp4 --cam cam1 --out /data/raw/good/cam1
argus extract-video line_cam2.mp4 --cam cam2 --out /data/raw/good/cam2
```

### Step 2 — Diagnose motion thresholds

```bash
argus diagnose-motion line_cam1.mp4 --plot motion.png
```

Copy the suggested values into `config.yaml → motion.drum_stop_threshold / drum_move_threshold`.

### Step 3 — Train and calibrate

```bash
argus calibrate \
  --normal /data/raw/good/cam1 \
  --abnormal /data/raw/defect/cam1   # optional, improves threshold tuning
```

### Step 4 — Run

```bash
argus run
# open http://localhost:8000
```

---

## Configuration

Edit `config.yaml` before building the container:

```yaml
cameras:
  - id: cam1
    device: /dev/video0
    width: 1280
    height: 720
    fps: 30
    roi: [160, 60, 960, 600]   # [x, y, w, h] — tune with argus doctor

motion:
  drum_stop_threshold: 0.015   # tune with: argus diagnose-motion
  drum_move_threshold: 0.060

burst:
  count: 5
  interval_ms: 150

detector:
  threshold: 0.5               # auto-tuned by argus calibrate
  model_path: /models/patchcore

storage:
  db_path: /data/argus.db
  images_dir: /data/images

ui:
  host: 127.0.0.1
  port: 8000
```

---

## Database Schema

```sql
cups(id, ts, verdict, score, cam1_path, cam2_path, notes)
defect_regions(id, cup_id, cam, bbox_json, score)
```

---

## Stack

- **Python 3.11** · **OpenCV** · **anomalib** (PatchCore) · **PyTorch CPU**
- **FastAPI** · **Jinja2** · **htmx** · **Chart.js**
- **SQLite WAL** · **openpyxl** · **asyncio**
- **Docker** + **docker-compose** · **uv**

---

## License

MIT
