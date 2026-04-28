from __future__ import annotations

import io
import json
from datetime import date as date_type
from pathlib import Path

import openpyxl
from openpyxl.styles import Font, PatternFill
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from argus.config import load_config
from argus.storage import Storage

config = load_config()
storage = Storage(config.storage.db_path, config.storage.images_dir)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app = FastAPI(title="ARGUS review")


@app.get("/", response_class=HTMLResponse)
def index(request: Request, date: str | None = None) -> HTMLResponse:
    if date is None:
        date = date_type.today().isoformat()
    stats = storage.get_daily_stats(date)
    hourly = storage.get_hourly_counts(date)
    hourly_by_hour = {r["hour"]: r for r in hourly}
    hourly_full = [
        hourly_by_hour.get(h, {"hour": h, "defects": 0, "good": 0})
        for h in range(24)
    ]
    defect_cups = storage.list_cups(limit=50, verdict="defect", date=date)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "stats": stats,
            "hourly_json": json.dumps(hourly_full),
            "cups": defect_cups,
            "date": date,
        },
    )


@app.get("/cup/{cup_id}", response_class=HTMLResponse)
def cup_detail(request: Request, cup_id: int) -> HTMLResponse:
    cup = storage.get_cup(cup_id)
    if cup is None:
        raise HTTPException(404, "cup not found")
    return templates.TemplateResponse(request, "cup.html", {"cup": cup})


@app.get("/image")
def image(path: str) -> FileResponse:
    p = Path(path)
    images_root = Path(config.storage.images_dir).resolve()
    try:
        p.resolve().relative_to(images_root)
    except ValueError as e:
        raise HTTPException(403, "path outside images_dir") from e
    if not p.exists():
        raise HTTPException(404, "image not found")
    return FileResponse(p)


@app.get("/export")
def export_excel(date: str | None = None) -> StreamingResponse:
    if date is None:
        date = date_type.today().isoformat()
    cups = storage.list_cups(limit=10000, date=date)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = f"ARGUS {date}"

    headers = ["ID", "Время", "Вердикт", "Score", "Камера 1", "Камера 2", "Примечание"]
    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True)

    red_fill = PatternFill(start_color="FFCCCC", end_color="FFCCCC", fill_type="solid")
    for cup in cups:
        ws.append([
            cup.id,
            cup.ts.strftime("%Y-%m-%d %H:%M:%S"),
            cup.verdict,
            round(cup.score, 3),
            cup.cam1_path or "",
            cup.cam2_path or "",
            cup.notes or "",
        ])
        if cup.verdict == "defect":
            for cell in ws[ws.max_row]:
                cell.fill = red_fill

    ws.column_dimensions["A"].width = 8
    ws.column_dimensions["B"].width = 20
    ws.column_dimensions["C"].width = 10
    ws.column_dimensions["D"].width = 10
    ws.column_dimensions["E"].width = 45
    ws.column_dimensions["F"].width = 45
    ws.column_dimensions["G"].width = 15

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    filename = f"argus_{date}.xlsx"
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
