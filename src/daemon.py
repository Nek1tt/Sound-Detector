"""
Два режима работы с аудио:

  1. infer_single_file()  — анализирует один .wav файл и возвращает результат
  2. AudioDaemon          — крутится в фоне, анализирует поток (mock-файл или микрофон)

Архитектура daemon:
    Producer thread  →  audio_queue  →  Consumer thread  →  callback(result)

Producer реализован в двух вариантах:
    _mock_producer  — читает файл, нарезает на окна, имитирует реальное время
    _mic_producer   — читает с микрофона через sounddevice (требует pip install sounddevice)

Для будущей интеграции с микрофоном — реализуйте свой callback и передайте
его в AudioDaemon. Пример см. в конце файла (if __name__ == "__main__").
"""
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from src.config import DaemonConfig, ModelConfig, PathsConfig, InferenceConfig
from src.model import AudioModel, get_device, load_audio


# ── Режим 1: одиночный инференс ───────────────────────────────────────────

def infer_single_file(
    filepath,
    model_cfg: ModelConfig,
    paths_cfg: PathsConfig,
    inf_cfg: InferenceConfig,
    top_k: int = 10,
) -> dict:
    """
    Загружает модель, анализирует один аудио-файл, возвращает результат.

    Returns:
        {
          "file":            str,
          "top_predictions": [(label, prob), ...],   # top_k штук
          "probs":           np.ndarray (527,),
          "features":        np.ndarray (C,),
          "elapsed_ms":      float,
        }
    """
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

    top_idx = np.argsort(probs)[::-1][:top_k]
    top_predictions = [(labels[i], float(probs[i])) for i in top_idx]

    return {
        "file":            str(filepath),
        "top_predictions": top_predictions,
        "probs":           probs,
        "features":        features,
        "elapsed_ms":      elapsed_ms,
    }


# ── Режим 2: фоновый daemon ───────────────────────────────────────────────

class AudioDaemon:
    """
    Непрерывный анализатор аудио-потока.

    Запуск из файла (разработка без микрофона):
        daemon = AudioDaemon(model, cfg, my_callback)
        daemon.start_mock("test.wav", loop=True)

    Запуск с микрофоном (требует pip install sounddevice):
        daemon = AudioDaemon(model, cfg, my_callback)
        daemon.start_mic()

    Остановка:
        daemon.stop()

    Callback вызывается из consumer-потока и получает dict:
        {
          "timestamp":       float,            # time.time()
          "top_predictions": [(label, prob)],  # топ-10
          "above_threshold": [(label, prob)],  # отфильтрованные по порогу
          "features":        np.ndarray,       # эмбеддинг
          "elapsed_ms":      float,
        }
    """

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

        self._queue      : queue.Queue = queue.Queue(maxsize=10)
        self._stop_event : threading.Event = threading.Event()
        self._threads    : list = []

    # ── Запуск ──────────────────────────────────────────────────────────

    def start_mock(self, source_file, loop: bool = True) -> None:
        """
        Mock-режим: читает файл, нарезает на скользящие окна,
        кладёт в очередь с задержкой hop_seconds.
        Полезно для разработки без физического микрофона.
        """
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
        """
        Mic-режим: читает с микрофона через sounddevice.
        Требует: pip install sounddevice
        На Pi: sudo apt install -y portaudio19-dev && pip install sounddevice
        """
        try:
            import sounddevice  # noqa — только проверка наличия
        except ImportError:
            raise ImportError(
                "sounddevice не установлен.\n"
                "Установите: pip install sounddevice\n"
                "На Pi сначала: sudo apt install -y portaudio19-dev"
            )
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
        """Останавливает все потоки."""
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=5)
        print("[daemon] Остановлен")

    # ── Внутренние методы ────────────────────────────────────────────────

    def _mock_producer(self, source_file, loop: bool) -> None:
        """Читает файл, нарезает на перекрывающиеся окна, кладёт в очередь."""
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
                try:
                    self._queue.put(chunk.copy(), timeout=1)
                except queue.Full:
                    pass  # consumer не успевает — пропускаем окно
                pos += hop_samples
                # имитируем реальный темп
                self._stop_event.wait(timeout=cfg.hop_seconds)
            if not loop:
                break

    def _mic_producer(self, device_index: Optional[int]) -> None:
        """
        Читает с микрофона через sounddevice.
        Накапливает кольцевой буфер размером window_seconds,
        каждые hop_seconds отправляет окно в очередь.
        """
        import sounddevice as sd  # type: ignore

        cfg    = self.cfg
        target_sr   = self.model.cfg.sample_rate
        win_samples = int(cfg.window_seconds * target_sr)
        hop_samples = int(cfg.hop_seconds    * target_sr)

        ring = np.zeros(win_samples, dtype=np.float32)
        mic_buf: list = []
        lock = threading.Lock()

        def _cb(indata, frames, time_info, status):
            with lock:
                mic_buf.extend(indata[:, 0].astype(np.float32).tolist())

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
                    if len(mic_buf) < hop_samples:
                        continue
                    new_raw = np.array(mic_buf[:hop_samples], dtype=np.float32)
                    mic_buf[:] = mic_buf[hop_samples:]

                # Ресемплинг если частота микрофона ≠ 32 kHz
                if cfg.mic_sample_rate != target_sr:
                    import librosa
                    new_raw = librosa.resample(
                        new_raw,
                        orig_sr=cfg.mic_sample_rate,
                        target_sr=target_sr,
                    )

                shift = min(len(new_raw), win_samples)
                ring  = np.roll(ring, -shift)
                ring[-shift:] = new_raw[-shift:]

                try:
                    self._queue.put(ring.copy(), timeout=0.1)
                except queue.Full:
                    pass

    def _consumer(self) -> None:
        """Берёт чанки из очереди, запускает инференс, вызывает callback."""
        while not self._stop_event.is_set():
            try:
                chunk = self._queue.get(timeout=1)
            except queue.Empty:
                continue

            t0 = time.time()
            probs, features = self.model.infer_waveform(chunk)
            elapsed_ms = (time.time() - t0) * 1000

            top_idx = np.argsort(probs)[::-1][:10]
            top_predictions = [(self.labels[i], float(probs[i])) for i in top_idx]
            above_threshold = [
                (lbl, p) for lbl, p in top_predictions
                if p >= self.cfg.confidence_threshold
            ]

            self.callback({
                "timestamp":       time.time(),
                "top_predictions": top_predictions,
                "above_threshold": above_threshold,
                "features":        features,
                "elapsed_ms":      elapsed_ms,
            })


# ── Пример использования ──────────────────────────────────────────────────
if __name__ == "__main__":
    from src.config import AppConfig

    cfg    = AppConfig()
    device = get_device("auto")
    model  = AudioModel.load(cfg.model, cfg.paths, device)

    def my_callback(result):
        ts  = time.strftime("%H:%M:%S", time.localtime(result["timestamp"]))
        lbl, prob = result["top_predictions"][0]
        print(f"[{ts}] {result['elapsed_ms']:.0f}мс | {lbl[:40]:<40} {prob:.3f}")

    daemon = AudioDaemon(model, cfg.daemon, my_callback)
    daemon.start_mock("data/test.wav", loop=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        daemon.stop()
   