"""
Фоновый daemon для анализа микрофона с поддержкой Few-Shot Learning и записи событий.
"""
import os
import queue
import threading
import time
import soundfile as sf
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from collections import deque

from src.config import DaemonConfig, ModelConfig, PathsConfig, InferenceConfig
from src.model import load_model, AudioModel, get_device, load_audio

CUSTOM_CLASSES = {
    "Тишина": ["Silence"],
    "Музыка": ["Music", "Musical instrument", "Singing", "Music of Latin America", "Pop music", "Rock music", "Electronic music", "Hip hop music", "Classical music", "Jazz", "Ambient music"],
    "Собака": ["Dog", "Bark", "Bow-wow", "Canidae, dogs, wolves"],
    "Кошка": ["Cat", "Meow", "Purr"],
    "Стекло": ["Glass", "Shatter", "Breaking", "Crash"],
    "Птица": ["Bird", "Chirp, tweet", "Caw", "Crow", "Bird vocalization, bird call, bird song"],
    "Детский плач": ["Baby cry, infant cry", "Crying, sobbing"],
    "Сирена": ["Siren", "Emergency vehicle", "Fire alarm", "Civil defense siren", "Ambulance (siren)", "Police car (siren)"],
    "Жарка еды": ["Frying (food)", "Sizzle"],
    "Речь": ["Speech", "Conversation", "Narration, monologue", "Female speech, woman speaking", "Male speech, man speaking"],
    "Щелчок": ["Finger snapping"],
    "Хлопок": ["Clapping", "Applause", "Hands"],
    "Транспорт": ["Vehicle", "Car", "Truck", "Bus", "Motorcycle", "Train", "Rail transport", "Aircraft", "Helicopter", "Traffic noise, roadway noise"],
    "Механизмы": ["Engine", "Clock", "Tick", "Mechanisms", "Mechanical fan"],
}

def aggregate_probs(probs: np.ndarray, labels: list) -> dict:
    result = {}
    for cls, class_labels in CUSTOM_CLASSES.items():
        indices = [i for i, l in enumerate(labels) if l in class_labels]
        result[cls] = float(np.max(probs[indices])) if indices else 0.0
    return result

class AudioDaemon:
    def __init__(self, model, cfg: DaemonConfig, callback: Callable[[dict], None]):
        self.model    = model
        self.cfg      = cfg
        self.callback = callback
        self.labels   = model.get_audioset_labels()

        self.ema: dict = {}
        for cls in CUSTOM_CLASSES: self.ema[cls] = 0.0

        self._queue      : queue.Queue  = queue.Queue(maxsize=1)
        self._stop_event : threading.Event = threading.Event()
        self._threads    : list = []
        self._state_lock : threading.Lock = threading.Lock()

        # Custom Sounds
        self.custom_embeddings = {}
        self.custom_lock = threading.Lock()
        self.custom_threshold = 0.70

        # Recording State
        self.active_filters = set()
        self.chat_id = None
        self.recording_buffer = []
        self.current_recording_label = None
        self.max_recording_chunks = int(10.0 / self.cfg.hop_seconds)

        self.alerts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "alerts")
        os.makedirs(self.alerts_dir, exist_ok=True)

    def add_custom_sound(self, name: str, features: np.ndarray):
        with self.custom_lock:
            self.custom_embeddings[name] = features

    def _flush_recording(self):
        if not self.recording_buffer or not self.current_recording_label: return
        full_audio = np.concatenate(self.recording_buffer)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(self.alerts_dir, f"{self.current_recording_label}_{timestamp}.wav")
        sf.write(filepath, full_audio, self.model.cfg.sample_rate)

        self.callback({
            "timestamp": time.time(),
            "label": self.current_recording_label,
            "chat_id": self.chat_id,
            "audio_file": filepath,
            "event_type": "alert"
        })
        self.recording_buffer = []
        self.current_recording_label = None

    def start_mock(self, source_file, loop: bool = True) -> None:
        with self._state_lock:
            if self._threads and any(t.is_alive() for t in self._threads): return
            self._stop_event.clear()
            t_prod = threading.Thread(target=self._mock_producer, args=(source_file, loop), daemon=True)
            t_cons = threading.Thread(target=self._consumer, daemon=True)
            self._threads = [t_prod, t_cons]
            for t in self._threads: t.start()

    def start_mic(self, device_index: Optional[int] = None) -> None:
        import sounddevice as sd # noqa
        with self._state_lock:
            if self._threads and any(t.is_alive() for t in self._threads): return
            self._stop_event.clear()
            t_prod = threading.Thread(target=self._mic_producer, args=(device_index,), daemon=True)
            t_cons = threading.Thread(target=self._consumer, daemon=True)
            self._threads = [t_prod, t_cons]
            for t in self._threads: t.start()

    def stop(self) -> None:
        with self._state_lock:
            self._stop_event.set()
            for t in self._threads:
                if t.is_alive(): t.join(timeout=5)
            self._threads.clear()

    def _mock_producer(self, source_file, loop: bool) -> None:
        cfg, sr = self.cfg, self.model.cfg.sample_rate
        wf = load_audio(source_file, sr=sr)
        win_samples, hop_samples = int(cfg.window_seconds * sr), int(cfg.hop_seconds * sr)
        while not self._stop_event.is_set():
            pos = 0
            while pos + win_samples <= len(wf):
                if self._stop_event.is_set(): return
                self._enqueue(wf[pos : pos + win_samples].copy())
                pos += hop_samples
                self._stop_event.wait(timeout=cfg.hop_seconds)
            if not loop: break

    def _mic_producer(self, device_index: Optional[int]) -> None:
        import sounddevice as sd
        import scipy.signal
        cfg, target_sr = self.cfg, self.model.cfg.sample_rate
        win_samples, mic_hop_samples = int(cfg.window_seconds * target_sr), int(cfg.hop_seconds * cfg.mic_sample_rate)
        ring = np.zeros(win_samples, dtype=np.float32)
        mic_buf_chunks, lock = [], threading.Lock()

        def _cb(indata, frames, time_info, status):
            with lock: mic_buf_chunks.append(indata[:, 0].copy())

        with sd.InputStream(samplerate=cfg.mic_sample_rate, channels=1, blocksize=cfg.chunk_size, device=device_index, callback=_cb, dtype="float32"):
            while not self._stop_event.is_set():
                self._stop_event.wait(timeout=cfg.hop_seconds)
                with lock:
                    if sum(len(c) for c in mic_buf_chunks) < mic_hop_samples: continue
                    flat_buf = np.concatenate(mic_buf_chunks)
                    if len(flat_buf) > mic_hop_samples * 2: flat_buf = flat_buf[-mic_hop_samples:]
                    new_raw, remainder = flat_buf[:mic_hop_samples], flat_buf[mic_hop_samples:]
                    mic_buf_chunks[:] = [remainder] if len(remainder) > 0 else []
                if cfg.mic_sample_rate != target_sr:
                    new_raw = scipy.signal.resample_poly(new_raw, up=target_sr, down=cfg.mic_sample_rate).astype(np.float32)
                shift = min(len(new_raw), win_samples)
                ring = np.roll(ring, -shift)
                ring[-shift:] = new_raw[-shift:]
                self._enqueue(ring.copy())

    def _consumer(self) -> None:
        SPIKE_CLASSES = {"Щелчок", "Хлопок", "Стекло"}
        alpha, spike_threshold = 0.3, 0.05

        # --- НОВОЕ: Вычисляем размер сдвига в сэмплах для правильной склейки ---
        target_sr = self.model.cfg.sample_rate
        hop_samples = int(self.cfg.hop_seconds * target_sr)
        # ----------------------------------------------------------------------

        while not self._stop_event.is_set():
            try:
                chunk = self._queue.get(timeout=1)
            except queue.Empty:
                continue

            probs, features = self.model.infer_waveform(chunk)
            agg = aggregate_probs(probs, self.labels)

            # ... (логика spike_detected и custom_detected остается без изменений) ...

            # (Для контекста, я пропускаю кусок кода с custom_detected,
            # он остается точно таким же, как у вас был)

            # ...

            if custom_detected:
                label, prob = custom_label, custom_prob
                spike_detected = True
            elif spike_detected:
                label, prob = spike_label, spike_prob
            else:
                for cls, raw_prob in agg.items():
                    self.ema[cls] = alpha * raw_prob + (1 - alpha) * self.ema.get(cls, 0.0)
                label, prob = max(self.ema.items(), key=lambda x: x[1]) if self.ema else ("Тишина", 0.0)

            # --- ИСПРАВЛЕННАЯ ЛОГИКА ЗАПИСИ (Без дублирования) ---
            is_target = label in self.active_filters and prob >= self.cfg.confidence_threshold

            if is_target:
                if self.current_recording_label == label:
                    if len(self.recording_buffer) < self.max_recording_chunks:
                        # УЖЕ ПИШЕМ: Добавляем ТОЛЬКО новый кусочек аудио (сдвиг окна)
                        self.recording_buffer.append(chunk[-hop_samples:])
                    else:
                        # Достигли лимита 10 сек -> сохраняем
                        self._flush_recording()
                        self.current_recording_label = label
                        # НАЧИНАЕМ ЗАНОВО: Первый чанк берем целиком (полное окно)
                        self.recording_buffer = [chunk]
                else:
                    # КЛАСС СМЕНИЛСЯ (или только начался)
                    self._flush_recording()
                    self.current_recording_label = label
                    # НАЧИНАЕМ ЗАПИСЬ: Первый чанк берем целиком (полное окно)
                    self.recording_buffer = [chunk]
                    print(f"🔴 Начата запись: {label}")
            else:
                # ЗВУК ПРОПАЛ -> сохраняем всё, что успели накопить
                if self.current_recording_label is not None:
                    self._flush_recording()
            # ---------------------------------------------------

            # Коллбек для стрима
            if prob >= self.cfg.confidence_threshold:
                self.callback({"event_type": "stream_log", "label": label, "prob": prob})

    def _enqueue(self, chunk: np.ndarray) -> None:
        try: self._queue.put(chunk, block=False)
        except queue.Full:
            try: self._queue.get_nowait()
            except queue.Empty: pass
            self._queue.put(chunk, block=False)