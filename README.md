[🇷🇺 Русский](README.ru.md) | 🇬🇧 English

# ARGUS — Cup Defect Detection System

Automated quality inspection for a cup production line. Two USB cameras capture each cup while the drum pauses. A PatchCore anomaly detector and EfficientNet classifier flag defects in real time, save photos, and serve a web dashboard with shift statistics and Excel export.

**No hardware trigger required.** The moment the drum stops is detected via MOG2 motion analysis — no modifications to the machine needed.

---

## Features

- **Motion-triggered burst capture** — 5 frames per camera while the cup spins from inertia, covering ~180° of surface
- **Anomaly detection** — PatchCore (anomalib) trained only on good cups, no defect labeling needed
- **Defect classifier** — EfficientNet-B0 with Grad-CAM, identifies defect type (dent, scratch, seam defect)
- **Automatic threshold tuning** — ROC curve on validation set, threshold selected at max F1
- **Web dashboard** — shift statistics, hourly chart, defect table with thumbnails
- **Per-cup detail page** — both camera frames + anomaly heatmap overlay
- **Manual photo inspection** — upload any images via browser and get the model's verdict
- **Excel export** — one-click report with defective rows highlighted red
- **Data cleanup** — delete records and photos for a selected day to free up disk space
- **Offline replay** — run the full pipeline on recorded video without cameras

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│ Docker container                                 │
│                                                  │
│  Web UI  (FastAPI + Jinja2, port 8000)           │
│       │                                          │
│  Storage (SQLite WAL + JPEG files on disk)       │
│       │                                          │
│  Orchestrator                                    │
│   ├── CameraWorker × 2  (OpenCV + MOG2 FSM)      │
│   ├── DefectDetector    (PatchCore / anomalib)   │
│   └── DefectClassifier  (EfficientNet-B0)        │
└──────────────────────────────────────────────────┘
          │ USB passthrough
       Cam1   Cam2
```

| Module | Responsibility |
|---|---|
| `argus/motion_fsm.py` | FSM: `WAITING → DRUM_STOP → BURST → COOLDOWN` |
| `argus/camera.py` | Capture loop, MOG2 motion detection, burst collection |
| `argus/pipeline.py` | Orchestrator, syncs events from two cameras by timestamp |
| `argus/detector.py` | PatchCore inference, returns score + anomaly map |
| `argus/classifier.py` | EfficientNet-B0 with Grad-CAM, binary classification |
| `argus/storage.py` | SQLite writes + JPEG save |
| `argus/calibrate.py` | Train PatchCore, build ROC, save threshold |
| `argus/ui/app.py` | FastAPI routes: dashboard, cup detail, export, inspect, cleanup |

---

## Quick Start (Docker)

**Requirements:** Docker, docker-compose, two USB cameras at `/dev/video0` and `/dev/video1`.

```bash
git clone https://github.com/mcSHROUD/ARGUS.git
cd ARGUS
docker compose up --build
```

UI available at [http://localhost:8000](http://localhost:8000)

> **Note:** On first launch Docker downloads backbone weights (~300 MB). For air-gapped environments set `TRANSFORMERS_OFFLINE=1` in `docker-compose.yml` after the initial download.

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
| `argus replay VIDEO_CAM1` | Run pipeline on pre-recorded video |
| `argus diagnose-motion VIDEO` | Analyse motion scores, suggest FSM thresholds |
| `argus doctor` | Self-check: cameras, paths, dependencies |

### Examples

```bash
# Extract normal frames from video for training
argus extract-video line_cam1.mp4 --cam cam1 --out /data/raw/good/cam1

# Analyse motion to calibrate FSM thresholds
argus diagnose-motion line_cam1.mp4

# Train the detector
argus calibrate --normal /data/raw/good/cam1

# Start live detection
argus run
```

---

## Workflow (new production line)

### Step 1 — Record video on-site

```bash
ffmpeg -i /dev/video0 -t 600 line_cam1.mp4
ffmpeg -i /dev/video1 -t 600 line_cam2.mp4
```

### Step 2 — Extract burst frames

```bash
argus extract-video line_cam1.mp4 --cam cam1 --out /data/raw/good/cam1
argus extract-video line_cam2.mp4 --cam cam2 --out /data/raw/good/cam2
```

### Step 3 — Diagnose motion thresholds

```bash
argus diagnose-motion line_cam1.mp4
```

Copy the suggested values into `config.yaml` → `motion.drum_stop_threshold` / `drum_move_threshold`.

### Step 4 — Train and calibrate

```bash
argus calibrate --normal /data/raw/good/cam1
```

### Step 5 — Run

```bash
docker compose up -d
# open http://localhost:8000
```

---

## Configuration

All parameters in `config.yaml`:

```yaml
cameras:
  - id: cam1
    device: /dev/video0
    width: 1920
    height: 1080
    fps: 60
    roi: [700, 0, 600, 700]   # [x, y, w, h] — inspection zone on the drum

motion:
  drum_stop_threshold: 0.004   # <= means drum is stopped (tune with diagnose-motion)
  drum_move_threshold: 0.019   # >= means drum is spinning

burst:
  frames: 5
  max_duration_ms: 800

detector:
  model_path: models_new/patchcore.ckpt
  threshold: 0.5               # overwritten by argus calibrate

storage:
  db_path: data/argus.db
  images_dir: data/images
  save_worst_only: true        # save only the frame with the highest score

ui:
  host: 0.0.0.0
  port: 8000
```

---

## Web Interface

| Page | URL | Description |
|---|---|---|
| Dashboard | `/` | Shift statistics, hourly chart, defect table |
| Cup detail | `/cup/{id}` | Photos from both cameras, heatmap, defect type |
| Inspection | `/inspect` | Upload photos manually and get the model's verdict |
| Export | `/export?date=YYYY-MM-DD` | Download Excel report for a selected day |
| Cleanup | `/cleanup` | Delete data for a selected day to free disk space |

---

## Models

Two algorithms work in tandem:

| Model | File | Description |
|---|---|---|
| **PatchCore** | `models_new/patchcore.ckpt` | Anomaly detector, trained on good cups only |
| **EfficientNet-B0** | `models_new/classifier.pt` | Binary classifier with Grad-CAM visualization |

Threshold values are stored in `*.threshold` files alongside the models and automatically override the value in `config.yaml`.

---

## Database Schema

```sql
cups(id, ts, verdict, score, cam1_path, cam2_path, cam1_heatmap_path, cam2_heatmap_path, notes)
defect_regions(id, cup_id, cam, bbox, score)
feedback(id, cup_id, user_verdict, ts, comment)
system_events(id, ts, kind, data)
```

---

## Deploying to a New Device

1. Install Docker: `sudo apt install docker.io docker-compose-plugin`
2. Clone the repo: `git clone https://github.com/mcSHROUD/ARGUS.git`
3. Connect cameras, verify devices: `ls /dev/video*`
4. Edit `config.yaml` (camera devices, ROI)
5. `docker compose up --build -d`
6. When moving to a new line — recalibrate: `argus calibrate --normal /data/raw/good/cam1`

---

## Stack

- **Python 3.11** · **OpenCV** · **anomalib** (PatchCore) · **PyTorch CPU**
- **FastAPI** · **Jinja2** · **Chart.js**
- **SQLite WAL** · **openpyxl**
- **Docker** + **docker-compose** · **Git LFS** (models and dataset)

---

## License

MIT
