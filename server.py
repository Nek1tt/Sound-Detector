import threading
import uvicorn
import time
import datetime
import os
import shutil
from fastapi import FastAPI, UploadFile, File

# Импорты из вашей ML-части
from src.config import AppConfig
from src.model import AudioModel, get_device
from src.daemon import AudioDaemon

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_SOUNDS_DIR = os.path.join(BASE_DIR, "custom_sounds")
os.makedirs(CUSTOM_SOUNDS_DIR, exist_ok=True)

# Глобальные переменные
latest_event = {"timestamp": "--:--:--", "message": "Система инициализирована"}
daemon = None  # Ссылка на запущенный демон


def on_audio_event(result: dict):
    """Callback-функция, которая вызывается демоном при анализе каждого окна"""
    global latest_event

    # Берем только те, что превысили порог confidence_threshold
    above = result.get("above_threshold", [])

    if above:
        # Берем самый вероятный звук (включая кастомный, если он сработал)
        top_label, prob = above[0]
        current_time = datetime.datetime.now().strftime("%H:%M:%S")

        # Лог в консоль сервера
        print(f"АУДИО: [{current_time}] {top_label} (Уверенность: {prob:.2f})")

        latest_event = {"timestamp": current_time, "message": top_label}


def start_ml_daemon():
    """Инициализация и запуск реальной модели"""
    global daemon
    print("Загрузка модели...")
    cfg = AppConfig()
    device = get_device("auto")
    model = AudioModel.load(cfg.model, cfg.paths, device)

    daemon = AudioDaemon(model, cfg.daemon, on_audio_event)

    # Загружаем уже существующие кастомные звуки из папки при старте
    for f in os.listdir(CUSTOM_SOUNDS_DIR):
        if f.endswith(".ogg"):
            name = f.replace(".ogg", "")
            daemon.load_custom_sound(name, os.path.join(CUSTOM_SOUNDS_DIR, f))

    # Запуск микрофона (потребуется sounddevice на сервере/Raspberry)
    daemon.start_mic()
    # Если микрофона нет и вы тестируете файлом:
    # daemon.start_mock("data/test.wav", loop=True)


@app.get("/logs")
async def get_logs():
    return {"events": [latest_event]}  # Обернул в events список, как ожидает ваш бот


@app.post("/add_sound")
async def add_sound(file: UploadFile = File(...), name: str = "custom"):
    try:
        clean_name = "".join(x for x in name if x.isalnum() or x in "._- ")
        file_name = f"{clean_name}.ogg"
        file_path = os.path.join(CUSTOM_SOUNDS_DIR, file_name)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"✅ Файл сохранен: {file_name}")

        # Сразу передаем новый файл в модель для расчета эмбеддинга
        if daemon:
            daemon.load_custom_sound(clean_name, file_path)

        return {"status": "success", "filename": file_name}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/")
async def root():
    return {"status": "ML Board & API Active"}


if __name__ == "__main__":
    # Запуск ML демона в отдельном потоке
    threading.Thread(target=start_ml_daemon, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8085)