from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from argus.config import load_config
from argus.storage import Storage

config = load_config()
storage = Storage(config.storage.db_path, config.storage.images_dir)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app = FastAPI(title="ARGUS review")


@app.get("/", response_class=HTMLResponse)
def index(request: Request, verdict: str | None = None, limit: int = 50) -> HTMLResponse:
    cups = storage.list_cups(limit=limit, verdict=verdict)
    return templates.TemplateResponse(
        request,
        "index.html",
        {"cups": cups, "verdict": verdict},
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


@app.post("/feedback")
def feedback(
    cup_id: int = Form(...),
    verdict: str = Form(...),
    comment: str = Form(""),
) -> dict:
    if verdict not in ("confirmed", "false_positive", "missed"):
        raise HTTPException(400, "invalid verdict")
    storage.add_feedback(cup_id, verdict, comment or None)
    return {"ok": True}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
