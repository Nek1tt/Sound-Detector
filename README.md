# EfficientAT Audio Classifier

Классификатор звуков на базе [EfficientAT](https://github.com/fschmid56/EfficientAT)  
(MobileNet / DyMN, предобучены на AudioSet 527 классов).

Поддерживает оценку на **ESC-50**, инференс одного файла и фоновый режим  
для непрерывного анализа (mock-файл или микрофон).

Совместимо с: Linux x86_64, macOS, **Raspberry Pi OS Lite (Bookworm, 64-bit)**.

---

## Структура

```
efficientat-audio/
├── main.py                    # Точка входа CLI
├── requirements.txt           # Зависимости
│
├── src/
│   ├── config.py              # Конфигурация: модель, пути, daemon
│   ├── model.py               # Загрузка EfficientAT, инференс
│   ├── dataset.py             # ESC-50 + маппинг AudioSet
│   ├── evaluate.py            # Метрики: Accuracy, F1, mAP, linear probe
│   └── daemon.py              # Одиночный инференс + фоновый режим
│
├── third_party/               # Сюда клонируется EfficientAT (git clone)
├── data/                      # Датасеты (не в git, скачать скриптом)
├── outputs/                   # results.json после evaluate
│
├── scripts/
│   ├── download_data.py       # Скачать ESC-50
│   └── setup_pi.sh            # Полная установка на Raspberry Pi
│

```

---

## Установка

### Обычный компьютер (Linux / macOS / Windows)

```bash
# 1. Клонировать репозиторий
git clone <url> Sound-Detector && cd Sound-Detector

# 2. Виртуальное окружение и зависимости
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Клонировать EfficientAT (веса скачаются при первом запуске)
git clone --depth=1 https://github.com/fschmid56/EfficientAT.git third_party/EfficientAT

# 4. Скачать ESC-50 (~600 MB)
python scripts/download_data.py
```

### Raspberry Pi OS Lite (Bookworm, 64-bit)

```bash
chmod +x scripts/setup_pi.sh
./scripts/setup_pi.sh
```

Скрипт сам установит системные пакеты, создаст venv, поставит PyTorch  
с правильным индексом для aarch64 и клонирует EfficientAT.

---

## CLI

### evaluate — оценка на ESC-50

```bash
# Полная оценка, все 2000 файлов, 5-fold CV
python main.py evaluate

# Конкретная модель
python main.py evaluate --model mn04_as          # Pi 4, быстро
python main.py evaluate --model mn10_as          # баланс
python main.py evaluate --model mn20_as          # максимальное качество

# Быстрая проверка — только fold 1 (400 файлов)
python main.py evaluate --fold 1

# Все опции
python main.py evaluate --model mn10_as \
    --esc50-dir data/ESC-50-master \
    --output-dir outputs \
    --device cpu \
    --threads 4

python main.py evaluate --help
```

Результат сохраняется в `outputs/results.json`:

```json
{
  "model": "mn10_as",
  "zeroshot": {
    "accuracy": 0.712,
    "f1_macro": 0.689,
    "mAP": 0.731,
    "ap_per_class": { "dog": 0.95, "cat": 0.88, ... }
  },
  "linear_probe": {
    "accuracy_mean": 0.891,
    "accuracy_std": 0.012,
    ...
  }
}
```

### infer — инференс одного файла

```bash
python main.py infer sounds/dog.wav
python main.py infer sounds/siren.mp3 --top-k 5
python main.py infer sounds/rain.wav --model mn04_as --top-k 10
```

Вывод:
```
[infer] Файл     : sounds/siren.mp3
[infer] Время    : 312 мс

Топ-5 предсказаний:
   1. 0.923  ██████████████████████████████  Siren
   2. 0.412  ████████████                    Emergency vehicle
   3. 0.201  ██████                          Car alarm
   4. 0.098  ██                              Vehicle
   5. 0.043  █                               Traffic noise
```

### daemon — фоновый режим

**Mock-режим** (разработка, без микрофона):

```bash
python main.py daemon --mode mock --source sounds/test.wav --loop
python main.py daemon --mode mock --source sounds/test.wav --loop \
    --threshold 0.4 --window 5.0 --hop 0.5
```

**Mic-режим** (реальный микрофон):

```bash
# Сначала установить sounddevice
pip install sounddevice==0.4.7
# На Pi: sudo apt install -y portaudio19-dev && pip install sounddevice==0.4.7

# Запуск
python main.py daemon --mode mic
python main.py daemon --mode mic --threshold 0.4 --model mn04_as

# Список доступных устройств
python -c "import sounddevice; print(sounddevice.query_devices())"
python main.py daemon --mode mic --mic-device 2
```

Вывод daemon:
```
[14:32:01]   298мс  0.731  Dog                                    ← АЛЕРТ (1 кл.)
[14:32:01]   301мс  0.412  Dog
[14:32:02]   295мс  0.089  Silence
```

---

## Выбор модели

| Модель      | Параметры | Рекомендуется        | Скорость на Pi 4 CPU |
|-------------|-----------|----------------------|----------------------|
| `mn04_as`   | ~1.2M     | Pi 4, IoT, батарейки | ~200 мс/файл         |
| `mn05_as`   | ~1.8M     | Pi 4 / Pi 5          | ~280 мс/файл         |
| `mn10_as`   | ~4.9M     | Ноутбук, сервер      | ~450 мс/файл         |
| `mn20_as`   | ~9.9M     | Сервер               | ~800 мс/файл         |
| `dymn10_as` | ~10M      | GPU                  | —                    |

---