import threading
import uvicorn
import time
import datetime
import os
import shutil
from fastapi import FastAPI, UploadFile, File
from collections import deque
events_queue = deque(maxlen=100)
# ИЗМЕНЕНИЕ: Импорты для работы реальной модели
from src.config import AppConfig
from src.model import AudioModel, get_device
from src.daemon import AudioDaemon

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Глобальная переменная для хранения последнего события с платы
latest_event = {"timestamp": "--:--:--", "message": "Система инициализирована"}

# ИЗМЕНЕНИЕ: Блокировка для безопасного доступа к latest_event из разных потоков
event_lock = threading.Lock()

def update_event_callback(result: dict):
    """
    Коллбек, который вызывает AudioDaemon после каждого инференса.
    """
    global latest_event
     
    if result["top_predictions"]:
        event_type = result["top_predictions"][0][0]
        prob = result["top_predictions"][0][1]
        #second_event_type = result["top_predictions"][1][0]
        #second_prob = result["top_predictions"][1][1]
        
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Обновляем состояние для API потокобезопасно
        with event_lock:
            events_queue.append({
                "timestamp": current_time,
                "message": event_type
            })
        with event_lock:
            latest_event = {"timestamp": current_time, "message": event_type}

        # Логгирование в консоль с метриками производительности
        print(f"ПЛАТА: [{current_time}]  Инференс: {result['elapsed_ms']:4.0f}мс | 🔊 {event_type:<30} {prob:.3f} ")


# --- ЭНДПОИНТЫ API ---
@app.get("/logs")
async def get_logs():
    with event_lock:
        events = list(events_queue)
    return {"events": events}

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
    # Инициализация конфигурации
    cfg = AppConfig()
    
    # ПРИНУДИТЕЛЬНО ставим легкую модель для Raspberry Pi
    cfg.model.name = "mn04_as" 
    
    # Инициализация модели
    print("[system] Загрузка модели...")
    device = get_device("cpu") # На Pi гарантированно CPU
    model = AudioModel.load(cfg.model, cfg.paths, device)
    
    # Инициализация и запуск демона
    daemon = AudioDaemon(model, cfg.daemon, update_event_callback)
    daemon.start_mic()
    
    # Запуск сервера
    uvicorn.run(app, host="0.0.0.0", port=8085)
