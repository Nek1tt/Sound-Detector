import threading
import uvicorn
import time
import datetime
import random
import os
import shutil
from fastapi import FastAPI, UploadFile, File

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Глобальная переменная для хранения последнего события с платы
latest_event = {"timestamp": "--:--:--", "message": "Система инициализирована"}


# --- ЛОГИКА ПЛАТЫ (Эмуляция или твой код) ---
def board_listener():
    global latest_event
    sounds_pool = ["dog bark", "siren", "speech", "music", "glass break", "silence"]
    while True:
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        # Здесь должен быть твой реальный код захвата и классификации
        event_type = random.choice(sounds_pool)

        # Лог в консоль
        print(f"ПЛАТА: [{current_time}] {event_type}")

        # Обновляем состояние для API
        latest_event = {"timestamp": current_time, "message": event_type}
        time.sleep(2)


# --- ЭНДПОИНТЫ API ---
@app.get("/logs")
async def get_logs():
    return latest_event


@app.post("/add_sound")
async def add_sound(file: UploadFile = File(...), name: str = "custom"):
    try:
        clean_name = "".join(x for x in name if x.isalnum() or x in "._- ")
        file_name = f"{clean_name}.ogg"
        file_path = os.path.join(BASE_DIR, file_name)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"✅ Файл сохранен: {file_name}")
        return {"status": "success", "filename": file_name}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/")
async def root():
    return {"status": "Board & API Active"}


if __name__ == "__main__":
    # Запуск микрофона в отдельном потоке
    threading.Thread(target=board_listener, daemon=True).start()
    # Запуск сервера на порту 8085
    uvicorn.run(app, host="0.0.0.0", port=8085)