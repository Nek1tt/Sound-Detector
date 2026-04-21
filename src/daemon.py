"""
Два режима работы с аудио:

  1. infer_single_file()  — анализирует один .wav файл и возвращает результат
  2. AudioDaemon          — крутится в фоне, анализирует поток (mock-файл или микрофон)
"""
import os
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from collections import deque

# ОПТИМИЗАЦИЯ ДЛЯ RASPBERRY PI: Ограничиваем количество потоков, 
# чтобы избежать перегрузки ARM-процессора переключением контекста
os.environ["OMP_NUM_THREADS"] = "4"
torch.set_num_threads(4)

from src.config import DaemonConfig, ModelConfig, PathsConfig, InferenceConfig
from src.model import AudioModel, get_device, load_audio
import scipy.signal
CUSTOM_CLASSES = {
    "Тишина": [
        "Silence"
    ],
    "Музыка": [
        "Music", "Musical instrument", "Singing", "Music of Latin America",
        "Pop music", "Rock music", "Electronic music", "Hip hop music",
        "Classical music", "Jazz", "Ambient music"
    ],

    "Собака": [
        "Dog", "Bark", "Bow-wow", "Canidae, dogs, wolves"
    ],

    "Кошка": [
        "Cat", "Meow", "Purr"
    ],

    "Стекло": [
        "Glass", "Shatter", "Breaking", "Crash"
    ],

    "Птица": [
        "Bird", "Chirp, tweet", "Caw", "Crow", "Bird vocalization, bird call, bird song"
    ],

    "Детский плач": [
        "Baby cry, infant cry", "Crying, sobbing"
    ],

    "Сирена": [
        "Siren", "Emergency vehicle", "Fire alarm", "Civil defense siren",
        "Ambulance (siren)", "Police car (siren)"
    ],

    "Жарка еды": [
        "Frying (food)", "Sizzle"
    ],

    "Речь": [
        "Speech", "Conversation", "Narration, monologue",
        "Female speech, woman speaking", "Male speech, man speaking"
    ],

    "Щелчок": [
        "Finger snapping"
    ],

    "Хлопок": [
        "Clapping", "Applause", "Hands"
    ],

    "Транспорт": [
        "Vehicle", "Car", "Truck", "Bus", "Motorcycle",
        "Train", "Rail transport", "Aircraft", "Helicopter",
        "Traffic noise, roadway noise", "Engine"
    ],
}
# ── Режим 1: одиночный инференс ───────────────────────────────────────────
def aggregate_probs(probs, labels):
    result = {}

    for cls, class_labels in CUSTOM_CLASSES.items():
        indices = [i for i, l in enumerate(labels) if l in class_labels]
        if indices:
            result[cls] = float(np.max(probs[indices]))  # или sum
        else:
            result[cls] = 0.0

    return result

def infer_single_file(
    filepath,
    model_cfg: ModelConfig,
    paths_cfg: PathsConfig,
    inf_cfg: InferenceConfig,
    top_k: int = 10,
) -> dict:
    device = get_device(inf_cfg.device)
    model  = AudioModel.load(model_cfg, paths_cfg, device)
    labels = model.get_audioset_labels()

    t0 = time.time()
    waveform = load_audio(
        filepath,
        sr=model_cfg.sample_rate,
        clip_samples=model_cfg.clip_samples,
    )
    probs, features = model.infer_waveform(waveform)
    elapsed_ms = (time.time() - t0) * 1000

   # top_idx = np.argsort(probs)[::-1][:top_k]
   # top_predictions = [(labels[i], float(probs[i])) for i in top_idx]
    agg = aggregate_probs(probs, self.labels)

    top_predictions = sorted(
    agg.items(),
    key=lambda x: x[1],
    reverse=True
    )
    return {
        "file":            str(filepath),
        "top_predictions": top_predictions,
        "probs":           probs,
        "features":        features,
        "elapsed_ms":      elapsed_ms,
    }


# ── Режим 2: фоновый daemon ───────────────────────────────────────────────

class AudioDaemon:
    def __init__(
        self,
        model: AudioModel,
        cfg: DaemonConfig,
        callback: Callable[[dict], None],
    ):
        self.model    = model
        self.cfg      = cfg
        self.callback = callback
        self.labels   = model.get_audioset_labels()
        self.smooth_buffer = deque(maxlen=5)
        
        self.stable_frames = 2
        self.spike_threshold = 0.05
        self.ema_alpha = 0.3
        self.current_label = None
        self.stable_count = 0
        self.ema = {}
        # ИЗМЕНЕНИЕ: Очередь размером 1 для сброса старых кадров
        self._queue      : queue.Queue = queue.Queue(maxsize=1)
        self._stop_event : threading.Event = threading.Event()
        self._threads    : list = []
        
        # ИЗМЕНЕНИЕ: Мьютекс для безопасного запуска/остановки
        self._state_lock : threading.Lock = threading.Lock()

        for cls in self.labels:
            self.ema[cls] = 0.0

    # ── Запуск ──────────────────────────────────────────────────────────

    def start_mock(self, source_file, loop: bool = True) -> None:
        with self._state_lock:
            if self._threads and any(t.is_alive() for t in self._threads):
                print("[daemon] Предупреждение: Потоки уже запущены.")
                return

            print(f"[daemon] Mock-режим: {source_file}  loop={loop}")
            self._stop_event.clear()
            t_prod = threading.Thread(
                target=self._mock_producer, args=(source_file, loop), daemon=True
            )
            t_cons = threading.Thread(target=self._consumer, daemon=True)
            self._threads = [t_prod, t_cons]
            for t in self._threads:
                t.start()

    def start_mic(self, device_index: Optional[int] = None) -> None:
        try:
            import sounddevice  # noqa
        except ImportError:
            raise ImportError(
                "sounddevice не установлен.\n"
                "Установите: pip install sounddevice\n"
                "На Pi сначала: sudo apt install -y portaudio19-dev"
            )
            
        with self._state_lock:
            if self._threads and any(t.is_alive() for t in self._threads):
                print("[daemon] Предупреждение: Потоки уже запущены.")
                return

            print(f"[daemon] Mic-режим: device={device_index if device_index is not None else 'default'}")
            self._stop_event.clear()
            t_prod = threading.Thread(
                target=self._mic_producer, args=(device_index,), daemon=True
            )
            t_cons = threading.Thread(target=self._consumer, daemon=True)
            self._threads = [t_prod, t_cons]
            for t in self._threads:
                t.start()

    def stop(self) -> None:
        with self._state_lock:
            self._stop_event.set()
            for t in self._threads:
                if t.is_alive():
                    t.join(timeout=5)
            self._threads.clear()
            print("[daemon] Остановлен")

    # ── Внутренние методы ────────────────────────────────────────────────

    def _mock_producer(self, source_file, loop: bool) -> None:
        cfg   = self.cfg
        sr    = self.model.cfg.sample_rate
        wf    = load_audio(source_file, sr=sr)

        win_samples = int(cfg.window_seconds * sr)
        hop_samples = int(cfg.hop_seconds    * sr)

        while not self._stop_event.is_set():
            pos = 0
            while pos + win_samples <= len(wf):
                if self._stop_event.is_set():
                    return
                chunk = wf[pos : pos + win_samples]
                
                # ИЗМЕНЕНИЕ: Неблокирующее добавление с вытеснением
                try:
                    self._queue.put(chunk.copy(), block=False)
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._queue.put(chunk.copy(), block=False)
                    
                pos += hop_samples
                self._stop_event.wait(timeout=cfg.hop_seconds)
            if not loop:
                break

    def _mic_producer(self, device_index: Optional[int]) -> None:
        import sounddevice as sd
        import scipy.signal

        cfg    = self.cfg
        target_sr   = self.model.cfg.sample_rate
        win_samples = int(cfg.window_seconds * target_sr)
        mic_hop_samples = int(cfg.hop_seconds * cfg.mic_sample_rate)

        ring = np.zeros(win_samples, dtype=np.float32)
        
        # ИСПРАВЛЕНИЕ 1: Теперь это список не чисел, а массивов (чанков)
        mic_buf_chunks: list = [] 
        lock = threading.Lock()

        def _cb(indata, frames, time_info, status):
            with lock:
                # ИСПРАВЛЕНИЕ 2: Просто копируем готовый numpy-массив целиком. 
                # Никаких .tolist() и .astype()! (Он и так float32 благодаря dtype='float32')
                mic_buf_chunks.append(indata[:, 0].copy())

        with sd.InputStream(
            samplerate=cfg.mic_sample_rate,
            channels=1,
            blocksize=cfg.chunk_size,
            device=device_index,
            callback=_cb,
            dtype='float32',
        ):
            while not self._stop_event.is_set():
                self._stop_event.wait(timeout=cfg.hop_seconds)

                with lock:
                    # Быстро считаем общую длину данных без их склеивания
                    total_len = sum(len(chunk) for chunk in mic_buf_chunks)
                    if total_len < mic_hop_samples:
                        continue
                    
                    # ИСПРАВЛЕНИЕ 3: Мгновенно склеиваем чанки средствами С (numpy)
                    flat_buf = np.concatenate(mic_buf_chunks)
                    
                    if len(flat_buf) > mic_hop_samples * 2:
                        flat_buf = flat_buf[-mic_hop_samples:]
                        
                    new_raw = flat_buf[:mic_hop_samples]
                    
                    # Сохраняем остаток как единственный элемент списка
                    remainder = flat_buf[mic_hop_samples:]
                    mic_buf_chunks = [remainder] if len(remainder) > 0 else []

                # Быстрый ресемплинг через scipy
                if cfg.mic_sample_rate != target_sr:
                    new_raw = scipy.signal.resample_poly(
                        new_raw,
                        up=target_sr,
                        down=cfg.mic_sample_rate
                    ).astype(np.float32)

                # Сдвигаем кольцевой буфер
                shift = min(len(new_raw), win_samples)
                ring  = np.roll(ring, -shift)
                ring[-shift:] = new_raw[-shift:]

                # Отправляем в очередь с вытеснением старых кадров
                try:
                    self._queue.put(ring.copy(), block=False)
                except queue.Full:
                    try:
                        self._queue.get_nowait() 
                    except queue.Empty:
                        pass
                    self._queue.put(ring.copy(), block=False)

    def _consumer(self) -> None:
        SPIKE_CLASSES = {"Щелчок", "Хлопок", "Стекло"}
        alpha = 0.3                     # коэффициент EMA
        spike_threshold = 0.05          # порог для спайка

        while not self._stop_event.is_set():
            try:
                chunk = self._queue.get(timeout=1)
            except queue.Empty:
                continue

            t0 = time.time()
            probs, features = self.model.infer_waveform(chunk)
            elapsed_ms = (time.time() - t0) * 1000

            agg = aggregate_probs(probs, self.labels)   # русские классы

        # 1. Проверяем спайки (мгновенная реакция)
            spike_detected = False
            spike_label = None
            spike_prob = 0.0
            for cls in SPIKE_CLASSES:
                p = agg.get(cls, 0.0)
                if p > spike_threshold and p > spike_prob:
                    spike_detected = True
                    spike_label = cls
                    spike_prob = p

            if spike_detected:
                label = spike_label
                prob = spike_prob
            else:
            # 2. Обновляем EMA для всех классов
                for cls, raw_prob in agg.items():
                    old = self.ema.get(cls, 0.0)
                    self.ema[cls] = alpha * raw_prob + (1 - alpha) * old

            # 3. Выбираем класс с максимальной EMA
                if self.ema:
                    label, prob = max(self.ema.items(), key=lambda x: x[1])
                else:
                    label, prob = "Тишина", 0.0

            top_predictions = [(label, prob)]
            above_threshold = [(lbl, p) for lbl, p in top_predictions if p >= self.cfg.confidence_threshold]

            self.callback({
                "timestamp": time.time(),
                "top_predictions": top_predictions,
                "above_threshold": above_threshold,
                "features": features,
                "elapsed_ms": elapsed_ms,
                "queue_size": self._queue.qsize(),
            })

        # Для отладки (можно убрать)
            print(f"[{time.strftime('%H:%M:%S')}] {label:10} {prob:.3f}  spike={spike_detected}")

