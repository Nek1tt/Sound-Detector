"""
FastAPI сервер для инференса в реальном времени.
"""
import os
import threading
import uvicorn
import time
import datetime
import shutil
import asyncio
import httpx
import numpy as np
from pathlib import Path
from collections import deque
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from src.config import AppConfig
from src.model import load_model, get_device, load_audio
from src.daemon import AudioDaemon

# Ограничение потоков для защиты Raspberry Pi от перегрева
os.environ["OMP_NUM_THREADS"] = "2"

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_SOUNDS_DIR = os.path.join(BASE_DIR, "custom_sounds")
os.makedirs(CUSTOM_SOUNDS_DIR, exist_ok=True)

events_queue = deque(maxlen=100)
latest_event = {"timestamp": "--:--:--", "message": "Система инициализирована"}
event_lock   = threading.Lock()

cfg = AppConfig()
model = None
daemon = None
BOT_TOKEN = os.getenv("BOT_TOKEN")


def update_event_callback(result: dict):
    global latest_event

    # 1. Логи для стрима
    if result.get("event_type") == "stream_log":
        label = result["label"]
        prob = result["prob"]
        current_time = datetime.datetime.now().strftime("%H:%M:%S")

        with event_lock:
            events_queue.append({"timestamp": current_time, "message": label})
            latest_event = {"timestamp": current_time, "message": label}
        print(f"[{current_time}] Лог: {label:<15} {prob:.3f}")

    # 2. Завершенное событие (Отправка файла в Telegram)
    elif result.get("event_type") == "alert":
        audio_file = result["audio_file"]
        chat_id = result["chat_id"]
        label = result["label"]

        # --- НОВОЕ: Добавили проверки и обычный поток ---
        if not BOT_TOKEN:
            print("❌ ОШИБКА: BOT_TOKEN не задан на плате! Отправка невозможна.")
            return

        if not chat_id:
            print("❌ ОШИБКА: chat_id пустой. Бот не прислал ID пользователя.")
            return

        print(f"⏳ Начинаю отправку {os.path.basename(audio_file)} в TG...")

        # Запускаем синхронную отправку в отдельном потоке (чтобы не тормозить микрофон)
        threading.Thread(
            target=send_alert_to_telegram_sync,
            args=(chat_id, label, audio_file),
            daemon=True
        ).start()


# --- НОВОЕ: Синхронная функция отправки ---
def send_alert_to_telegram_sync(chat_id: int, label: str, file_path: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendVoice"
    try:
        with open(file_path, "rb") as audio:
            files = {"voice": (os.path.basename(file_path), audio, "audio/wav")}
            data = {
                "chat_id": chat_id,
                "caption": f"🚨 Обнаружен звук: **{label}**",
                "parse_mode": "Markdown"
            }

            # Обычный блокирующий запрос (безопасно, так как мы в отдельном потоке)
            response = requests.post(url, data=data, files=files, timeout=20)

            if response.status_code == 200:
                print(f"✈️ Файл {os.path.basename(file_path)} успешно отправлен в TG")
                try:
                    os.remove(file_path)  # Удаляем файл
                except OSError:
                    pass
            else:
                print(f"❌ Ошибка API TG: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"💥 Ошибка сети при отправке в TG: {e}")


class SyncConfig(BaseModel):
    chat_id: int
    filters: list[str]

@app.get("/logs")
async def get_logs():
    with event_lock:
        return {"events": list(events_queue)}

@app.post("/add_sound")
async def add_sound(file: UploadFile = File(...), name: str = "custom"):
    try:
        clean_name = "".join(x for x in name if x.isalnum() or x in "._- ")
        ogg_path = os.path.join(CUSTOM_SOUNDS_DIR, f"{clean_name}.ogg")
        npy_path = os.path.join(CUSTOM_SOUNDS_DIR, f"{clean_name}.npy")

        with open(ogg_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if model is not None:
            waveform = load_audio(ogg_path, sr=cfg.model.sample_rate)
            probs, features = model.infer_waveform(waveform)
            np.save(npy_path, features)

        return {"status": "success", "filename": f"{clean_name}.ogg"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/set_active_sounds")
async def set_active_sounds(config: SyncConfig):
    try:
        if daemon is None: return {"status": "error"}

        daemon.active_filters = set(config.filters)
        daemon.chat_id = config.chat_id
        daemon.custom_embeddings.clear()

        loaded_count = 0
        for name in config.filters:
            npy_path = os.path.join(CUSTOM_SOUNDS_DIR, f"{name}.npy")
            if os.path.exists(npy_path):
                embedding = np.load(npy_path)
                daemon.add_custom_sound(name, embedding)
                loaded_count += 1

        print(f"🔄 Синхронизация: {len(config.filters)} фильтров. Chat ID: {config.chat_id}")
        return {"status": "success", "loaded_custom": loaded_count}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
async def root():
    return {"status": "Active"}

if __name__ == "__main__":
    cfg.model.name = "mn04_as"
    backend = os.environ.get("AUDIO_BACKEND", "onnx").lower()
    if backend not in ("pt", "onnx", "tflite"): backend = "pt"
    cfg.model.backend = backend

    if os.environ.get("ONNX_PATH"): cfg.model.onnx_path = Path(os.environ["ONNX_PATH"])

    print(f"[system] Бэкенд: {backend.upper()}")
    device = get_device("cpu")
    model  = load_model(cfg.model, cfg.paths, device)

    daemon = AudioDaemon(model, cfg.daemon, update_event_callback)
    daemon.start_mic()

    uvicorn.run(app, host="0.0.0.0", port=8085)