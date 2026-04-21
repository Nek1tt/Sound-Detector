"""
FastAPI сервер для инференса в реальном времени (микрофон).

Выбор бэкенда через переменную окружения AUDIO_BACKEND:
  AUDIO_BACKEND=pt      python server.py   # по умолчанию
  AUDIO_BACKEND=onnx    python server.py
  AUDIO_BACKEND=tflite  python server.py

Дополнительно:
  ONNX_PATH=exports/mn04_as.onnx python server.py
  TFLITE_PATH=exports/mn04_as.tflite python server.py
"""
import os
import threading
import uvicorn
import time
import datetime
import shutil
from pathlib import Path
from collections import deque

from fastapi import FastAPI, UploadFile, File

from src.config import AppConfig
from src.model import load_model, get_device
from src.daemon import AudioDaemon

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

events_queue = deque(maxlen=100)
latest_event = {"timestamp": "--:--:--", "message": "Система инициализирована"}
event_lock   = threading.Lock()


def update_event_callback(result: dict):
    global latest_event

    if result["top_predictions"]:
        event_type = result["top_predictions"][0][0]
        prob       = result["top_predictions"][0][1]
        current_time = datetime.datetime.now().strftime("%H:%M:%S")

        with event_lock:
            events_queue.append({"timestamp": current_time, "message": event_type})
            latest_event = {"timestamp": current_time, "message": event_type}

        print(
            f"ПЛАТА: [{current_time}]  Инференс: {result['elapsed_ms']:4.0f}мс | "
            f"🔊 {event_type:<30} {prob:.3f}"
        )


# ── Эндпоинты ─────────────────────────────────────────────────────────────

@app.get("/logs")
async def get_logs():
    with event_lock:
        events = list(events_queue)
    return {"events": events}


@app.post("/add_sound")
async def add_sound(file: UploadFile = File(...), name: str = "custom"):
    try:
        clean_name = "".join(x for x in name if x.isalnum() or x in "._- ")
        file_name  = f"{clean_name}.ogg"
        file_path  = os.path.join(BASE_DIR, file_name)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"✅ Файл сохранен: {file_name}")
        return {"status": "success", "filename": file_name}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/")
async def root():
    return {"status": "Board & API Active"}


# ── Запуск ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = AppConfig()
    cfg.model.name = "mn04_as"

    # Выбор бэкенда через переменную окружения
    backend = os.environ.get("AUDIO_BACKEND", "pt").lower()
    if backend not in ("pt", "onnx", "tflite"):
        print(f"[warn] Неизвестный AUDIO_BACKEND='{backend}', используем 'pt'")
        backend = "pt"
    cfg.model.backend = backend

    # Явные пути к файлам (если заданы)
    if os.environ.get("ONNX_PATH"):
        cfg.model.onnx_path = Path(os.environ["ONNX_PATH"])
    if os.environ.get("TFLITE_PATH"):
        cfg.model.tflite_path = Path(os.environ["TFLITE_PATH"])

    print(f"[system] Бэкенд: {backend.upper()}")
    print("[system] Загрузка модели...")
    device = get_device("cpu")
    model  = load_model(cfg.model, cfg.paths, device)

    daemon = AudioDaemon(model, cfg.daemon, update_event_callback)
    daemon.start_mic()

    uvicorn.run(app, host="0.0.0.0", port=8085)