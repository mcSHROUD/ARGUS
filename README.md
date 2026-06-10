# ARGUS — Система контроля дефектов стаканчиков

Автоматическая инспекция стаканчиков на производственной линии. Две USB-камеры снимают каждый стаканчик в момент паузы барабана. PatchCore-детектор аномалий и EfficientNet-классификатор в реальном времени выявляют дефекты, сохраняют фотографии и отображают результаты в веб-дашборде со статистикой смены и экспортом в Excel.

**Без аппаратного триггера.** Обнаружение момента остановки барабана происходит через детекцию движения (MOG2) — никаких изменений в оборудование вносить не нужно.

---

## Возможности

- **Захват по движению** — 5 кадров с каждой камеры пока стаканчик вращается по инерции, покрывая ~180° поверхности
- **Обнаружение аномалий** — PatchCore (anomalib), обучается только на хороших стаканчиках, разметка дефектов не нужна
- **Классификатор дефектов** — EfficientNet-B0 с Grad-CAM, определяет тип дефекта (вмятина, царапина, дефект шва)
- **Автоматическая настройка порога** — ROC-кривая на валидационной выборке, порог выбирается по максимуму F1
- **Веб-дашборд** — статистика смены, почасовой график, таблица дефектов с миниатюрами
- **Детальная страница стаканчика** — кадры с обеих камер + наложение тепловой карты аномалий
- **Ручная проверка фото** — загрузите любые снимки через браузер и получите вердикт модели
- **Экспорт в Excel** — отчёт в один клик, дефектные строки выделены красным
- **Очистка данных** — удаление записей и фотографий за выбранный день для освобождения места на диске
- **Воспроизведение оффлайн** — запуск полного пайплайна на записанном видео без камер

---

## Архитектура

```
┌──────────────────────────────────────────────────┐
│ Docker-контейнер                                 │
│                                                  │
│  Веб-интерфейс  (FastAPI + Jinja2, порт 8000)    │
│       │                                          │
│  Хранилище (SQLite WAL + JPEG на диске)          │
│       │                                          │
│  Оркестратор                                     │
│   ├── CameraWorker × 2  (OpenCV + MOG2 FSM)      │
│   ├── DefectDetector    (PatchCore / anomalib)   │
│   └── DefectClassifier  (EfficientNet-B0)        │
└──────────────────────────────────────────────────┘
          │ USB passthrough
       Cam1   Cam2
```

| Модуль | Роль |
|---|---|
| `argus/motion_fsm.py` | FSM: `WAITING → DRUM_STOP → BURST → COOLDOWN` |
| `argus/camera.py` | Поток захвата, MOG2, сборка burst-события |
| `argus/pipeline.py` | Оркестратор, синхронизация двух камер по времени |
| `argus/detector.py` | Инференс PatchCore, возвращает score + карту аномалий |
| `argus/classifier.py` | EfficientNet-B0 с Grad-CAM, бинарная классификация |
| `argus/storage.py` | Запись в SQLite + сохранение JPEG |
| `argus/calibrate.py` | Обучение PatchCore, построение ROC, запись порога |
| `argus/ui/app.py` | FastAPI: дашборд, детали стаканчика, экспорт, инспекция, очистка |

---

## Быстрый старт (Docker)

**Требования:** Docker, docker-compose, две USB-камеры на `/dev/video0` и `/dev/video1`.

```bash
git clone https://github.com/mcSHROUD/ARGUS.git
cd ARGUS
docker compose up --build
```

Интерфейс доступен по адресу [http://localhost:8000](http://localhost:8000)

> **Примечание:** При первом запуске Docker скачивает веса backbone (~300 МБ). Для работы без интернета установите `TRANSFORMERS_OFFLINE=1` в `docker-compose.yml` после первоначальной загрузки.

---

## Команды CLI

Все команды выполняются внутри контейнера:

```bash
docker compose run --rm argus <команда>
```

| Команда | Описание |
|---|---|
| `argus run` | Запуск живого детектирования (требует обученную модель) |
| `argus calibrate` | Обучение PatchCore на нормальных кадрах, настройка порога |
| `argus ui` | Запуск веб-дашборда |
| `argus extract-video VIDEO` | Извлечение burst-кадров из записанного видео |
| `argus replay VIDEO_CAM1` | Запуск пайплайна на записанном видео |
| `argus diagnose-motion VIDEO` | Анализ motion-scores, рекомендации по порогам FSM |
| `argus doctor` | Самодиагностика: камеры, пути, зависимости |

### Примеры

```bash
# Извлечь нормальные кадры из видео для обучения
argus extract-video line_cam1.mp4 --cam cam1 --out /data/raw/good/cam1

# Проанализировать движение для подбора порогов FSM
argus diagnose-motion line_cam1.mp4

# Обучить детектор
argus calibrate --normal /data/raw/good/cam1

# Запустить живое детектирование
argus run
```

---

## Рабочий процесс (новая линия)

### Шаг 1 — Запись видео на производстве

```bash
ffmpeg -i /dev/video0 -t 600 line_cam1.mp4
ffmpeg -i /dev/video1 -t 600 line_cam2.mp4
```

### Шаг 2 — Извлечение кадров

```bash
argus extract-video line_cam1.mp4 --cam cam1 --out /data/raw/good/cam1
argus extract-video line_cam2.mp4 --cam cam2 --out /data/raw/good/cam2
```

### Шаг 3 — Подбор порогов движения

```bash
argus diagnose-motion line_cam1.mp4
```

Скопируйте предложенные значения в `config.yaml` → `motion.drum_stop_threshold` / `drum_move_threshold`.

### Шаг 4 — Обучение и калибровка

```bash
argus calibrate --normal /data/raw/good/cam1
```

### Шаг 5 — Запуск

```bash
docker compose up -d
# открыть http://localhost:8000
```

---

## Конфигурация

Все параметры в `config.yaml`:

```yaml
cameras:
  - id: cam1
    device: /dev/video0
    width: 1920
    height: 1080
    fps: 60
    roi: [700, 0, 600, 700]   # [x, y, w, h] — зона инспекции барабана

motion:
  drum_stop_threshold: 0.004   # <= = барабан стоит (подобрать через diagnose-motion)
  drum_move_threshold: 0.019   # >= = барабан крутится

burst:
  frames: 5
  max_duration_ms: 800

detector:
  model_path: models_new/patchcore.ckpt
  threshold: 0.5               # перезаписывается командой calibrate

storage:
  db_path: data/argus.db
  images_dir: data/images
  save_worst_only: true        # сохранять только кадр с максимальным score

ui:
  host: 0.0.0.0
  port: 8000
```

---

## Веб-интерфейс

| Страница | URL | Описание |
|---|---|---|
| Дашборд | `/` | Статистика смены, почасовой график, таблица дефектов |
| Стаканчик | `/cup/{id}` | Фото с обеих камер, тепловая карта, тип дефекта |
| Инспекция | `/inspect` | Ручная проверка — загрузите фото и получите вердикт |
| Экспорт | `/export?date=ГГГГ-ММ-ДД` | Скачать Excel-отчёт за выбранный день |
| Очистка | `/cleanup` | Удаление данных за день для освобождения диска |

---

## Модели

Два алгоритма работают совместно:

| Модель | Файл | Описание |
|---|---|---|
| **PatchCore** | `models_new/patchcore.ckpt` | Детектор аномалий, обучается только на хороших стаканчиках |
| **EfficientNet-B0** | `models_new/classifier.pt` | Бинарный классификатор с Grad-CAM для визуализации |

Пороговые значения хранятся в `*.threshold`-файлах рядом с моделями и автоматически перезаписывают значение из `config.yaml`.

---

## Схема базы данных

```sql
cups(id, ts, verdict, score, cam1_path, cam2_path, cam1_heatmap_path, cam2_heatmap_path, notes)
defect_regions(id, cup_id, cam, bbox, score)
feedback(id, cup_id, user_verdict, ts, comment)
system_events(id, ts, kind, data)
```

---

## Деплой на новое устройство

1. Установить Docker: `sudo apt install docker.io docker-compose-plugin`
2. Клонировать репозиторий: `git clone https://github.com/mcSHROUD/ARGUS.git`
3. Подключить камеры, проверить устройства: `ls /dev/video*`
4. Отредактировать `config.yaml` (устройства камер, ROI)
5. `docker compose up --build -d`
6. При смене линии — перекалибровать: `argus calibrate --normal /data/raw/good/cam1`

---

## Стек технологий

- **Python 3.11** · **OpenCV** · **anomalib** (PatchCore) · **PyTorch CPU**
- **FastAPI** · **Jinja2** · **Chart.js**
- **SQLite WAL** · **openpyxl**
- **Docker** + **docker-compose** · **Git LFS** (модели и датасет)

---

## Лицензия

MIT
