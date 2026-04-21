"""
Загрузка модели EfficientAT и инференс.

Поддерживает три бэкенда:
  pt     — оригинальный PyTorch (mn*_as / dymn*_as)
  onnx   — ONNX Runtime (нужен onnxruntime)
  tflite — TFLite interpreter (нужен tensorflow)

Препроцессинг (AugmentMelSTFT) всегда выполняется через PyTorch и
является общим для всех трёх бэкендов. На вход моделям подаётся
готовая мел-спектрограмма (1, 1, 128, T) — float32.
"""
import os
import sys
import time
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import librosa

from src.config import ModelConfig, PathsConfig


# ── Утилиты ───────────────────────────────────────────────────────────────

def _add_efficientat_to_path(repo_path: Path) -> None:
    repo_path = repo_path.resolve()
    if not repo_path.exists():
        raise FileNotFoundError(
            f"EfficientAT не найден: {repo_path}\n"
            f"Выполните: git clone https://github.com/fschmid56/EfficientAT.git {repo_path}"
        )
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))


def get_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _resolve_export_path(
    explicit: Optional[Path],
    exports_dir: Path,
    model_name: str,
    suffix: str,          # ".onnx" или ".tflite"
) -> Path:
    """
    Возвращает путь к файлу экспортированной модели.
    Порядок поиска:
      1. Явно указанный путь (explicit)
      2. exports/<model_name><suffix>  (напр. exports/mn04_as.onnx)
      3. <model_name><suffix>          (в корне проекта)
    """
    if explicit is not None:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"Указанный файл не найден: {p}")

    candidates = [
        Path(exports_dir) / f"{model_name}{suffix}",
        Path(f"{model_name}{suffix}"),
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Не найден файл модели {model_name}{suffix}.\n"
        f"Ожидаемые пути: {[str(c) for c in candidates]}\n"
        f"Сначала выполните конвертацию: python scripts/convert_to_onnx.py --model {model_name}"
    )


# ── Общий препроцессор (мел-спектрограмма) ───────────────────────────────

class MelPreprocessor:
    """
    Тонкая обёртка над AugmentMelSTFT из EfficientAT.
    Принимает waveform (N,) → возвращает spec (1, 1, n_mels, T) numpy float32.
    """

    def __init__(self, cfg: ModelConfig, device: torch.device, paths_cfg: PathsConfig):
        _add_efficientat_to_path(paths_cfg.efficientat_repo)
        from models.preprocess import AugmentMelSTFT  # type: ignore

        self.device = device
        self.mel = AugmentMelSTFT(
            n_mels=cfg.n_mels,
            sr=cfg.sample_rate,
            win_length=cfg.win_size,
            hopsize=cfg.hop_size,
            n_fft=cfg.n_fft,
        ).to(device).eval()

    def __call__(self, waveform_np: np.ndarray) -> np.ndarray:
        """waveform_np (N,) → spec_np (1, 1, n_mels, T)"""
        waveform = torch.from_numpy(waveform_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            spec = self.mel(waveform)      # (1, n_mels, T)
            spec = spec.unsqueeze(1)       # (1, 1, n_mels, T)
        return spec.cpu().numpy()


# ── Бэкенд 1: PyTorch ─────────────────────────────────────────────────────

class AudioModel:
    """
    Оригинальный PyTorch бэкенд.
    Совместим с остальным кодом через тот же интерфейс infer_waveform().
    """

    def __init__(self, model, mel_transform, device: torch.device, cfg: ModelConfig):
        self.model = model
        self.mel   = mel_transform
        self.device = device
        self.cfg   = cfg

    @classmethod
    def load(
        cls,
        model_cfg: ModelConfig,
        paths_cfg: PathsConfig,
        device: torch.device,
    ) -> "AudioModel":
        _add_efficientat_to_path(paths_cfg.efficientat_repo)

        from models.mn.model import get_model as get_mobilenet    # type: ignore
        from models.dymn.model import get_model as get_dymn       # type: ignore
        from models.preprocess import AugmentMelSTFT               # type: ignore
        from helpers.utils import NAME_TO_WIDTH                    # type: ignore

        name = model_cfg.name
        print(f"[model/pt] Загружаем {name}...")
        t0 = time.time()

        if name.startswith("dymn"):
            model = get_dymn(width_mult=NAME_TO_WIDTH(name), pretrained_name=name)
        else:
            model = get_mobilenet(
                width_mult=NAME_TO_WIDTH(name),
                pretrained_name=name,
                head_type="mlp",
            )

        model.to(device).eval()

        mel = AugmentMelSTFT(
            n_mels=model_cfg.n_mels,
            sr=model_cfg.sample_rate,
            win_length=model_cfg.win_size,
            hopsize=model_cfg.hop_size,
            n_fft=model_cfg.n_fft,
        ).to(device).eval()

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(
            f"[model/pt] Готово за {time.time() - t0:.1f}с | "
            f"Параметров: {n_params:.2f}M | Устройство: {device}"
        )
        return cls(model, mel, device, model_cfg)

    def infer_waveform(self, waveform_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        waveform = torch.from_numpy(waveform_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            spec = self.mel(waveform)
            spec = spec.unsqueeze(1)
            logits, features = self.model(spec)
        probs    = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        features = features.squeeze(0).cpu().numpy()
        return probs, features

    def get_audioset_labels(self):
        from helpers.utils import labels as AUDIOSET_LABELS  # type: ignore
        return AUDIOSET_LABELS

    @property
    def n_params_m(self) -> float:
        return sum(p.numel() for p in self.model.parameters()) / 1e6


# ── Бэкенд 2: ONNX Runtime ───────────────────────────────────────────────

class ONNXAudioModel:
    """
    ONNX Runtime бэкенд.

    Граф принимает мел-спектрограмму (1, 1, 128, T) и возвращает logits (1, 527).
    Препроцессинг (AugmentMelSTFT) выполняется на PyTorch и передаётся как numpy.
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        paths_cfg: PathsConfig,
        device: torch.device,
    ):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "ONNX Runtime не установлен.\n"
                "Установите: pip install onnxruntime"
            )

        self.cfg    = model_cfg
        self.device = device

        onnx_path = _resolve_export_path(
            model_cfg.onnx_path,
            paths_cfg.exports_dir,
            model_cfg.name,
            ".onnx",
        )

        print(f"[model/onnx] Загружаем {onnx_path}...")
        t0 = time.time()

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", 4))
        sess_opts.log_severity_level   = 3  # подавляем INFO-спам

        self._sess = ort.InferenceSession(
            str(onnx_path),
            sess_opts,
            providers=["CPUExecutionProvider"],
        )

        self._mel = MelPreprocessor(model_cfg, device, paths_cfg)

        # Получаем метки AudioSet через EfficientAT
        _add_efficientat_to_path(paths_cfg.efficientat_repo)
        from helpers.utils import labels as AUDIOSET_LABELS  # type: ignore
        self._labels = AUDIOSET_LABELS

        print(f"[model/onnx] Готово за {time.time() - t0:.1f}с | {onnx_path.name}")

    def infer_waveform(self, waveform_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        spec = self._mel(waveform_np)          # (1, 1, 128, T)
        logits = self._sess.run(["logits"], {"spec": spec})[0]  # (1, 527)
        probs = 1.0 / (1.0 + np.exp(-logits[0].astype(np.float32)))  # sigmoid
        # ONNX не экспортирует фичи по умолчанию — возвращаем нули-заглушку
        features = np.zeros(960, dtype=np.float32)
        return probs, features

    def get_audioset_labels(self):
        return self._labels

    @property
    def n_params_m(self) -> float:
        return 0.0  # ONNX не хранит параметры в Python


# ── Бэкенд 3: TFLite ─────────────────────────────────────────────────────

class TFLiteAudioModel:
    """
    TFLite бэкенд.

    Важно: onnx2tf транспонирует NCHW → NHWC, поэтому вход может быть
    (1, 128, 1001, 1) вместо (1, 1, 128, T). Класс определяет порядок
    осей автоматически по метаданным интерпретатора.
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        paths_cfg: PathsConfig,
        device: torch.device,
    ):
        try:
            import tensorflow as tf
            self._tf = tf
        except ImportError:
            raise ImportError(
                "TensorFlow не установлен.\n"
                "Установите: pip install tensorflow"
            )

        self.cfg    = model_cfg
        self.device = device

        tflite_path = _resolve_export_path(
            model_cfg.tflite_path,
            paths_cfg.exports_dir,
            model_cfg.name,
            ".tflite",
        )

        print(f"[model/tflite] Загружаем {tflite_path}...")
        t0 = time.time()

        self._interp = tf.lite.Interpreter(model_path=str(tflite_path))
        self._interp.allocate_tensors()

        self._inp_details = self._interp.get_input_details()
        self._out_details = self._interp.get_output_details()

        # Определяем, нужна ли транспозиция NCHW→NHWC
        inp_shape = list(self._inp_details[0]["shape"])
        # NHWC: [1, n_mels, T, 1]  → транспозиция нужна
        # NCHW: [1, 1, n_mels, T]  → транспозиция не нужна
        self._needs_transpose = (
            len(inp_shape) == 4 and inp_shape[-1] == 1 and inp_shape[1] > 1
        )
        fmt = "NHWC (транспозиция включена)" if self._needs_transpose else "NCHW"
        print(f"[model/tflite] Формат входа: {inp_shape} → {fmt}")

        self._mel = MelPreprocessor(model_cfg, device, paths_cfg)

        _add_efficientat_to_path(paths_cfg.efficientat_repo)
        from helpers.utils import labels as AUDIOSET_LABELS  # type: ignore
        self._labels = AUDIOSET_LABELS

        print(f"[model/tflite] Готово за {time.time() - t0:.1f}с | {tflite_path.name}")

    def infer_waveform(self, waveform_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        spec = self._mel(waveform_np)          # (1, 1, 128, T)

        if self._needs_transpose:
            # NCHW (1,1,128,T) → NHWC (1,128,T,1)
            tfl_input = spec.transpose(0, 2, 3, 1)
        else:
            tfl_input = spec

        self._interp.set_tensor(self._inp_details[0]["index"], tfl_input)
        self._interp.invoke()
        logits = self._interp.get_tensor(self._out_details[0]["index"])  # (1,527)

        probs    = 1.0 / (1.0 + np.exp(-logits[0].astype(np.float32)))
        features = np.zeros(960, dtype=np.float32)
        return probs, features

    def get_audioset_labels(self):
        return self._labels

    @property
    def n_params_m(self) -> float:
        return 0.0


# ── Фабрика ───────────────────────────────────────────────────────────────

def load_model(
    model_cfg: ModelConfig,
    paths_cfg: PathsConfig,
    device: torch.device,
):
    """
    Возвращает экземпляр нужного бэкенда в зависимости от model_cfg.backend.

    Все три класса имеют одинаковый интерфейс:
        probs, features = model.infer_waveform(waveform_np)
        labels          = model.get_audioset_labels()
        model.n_params_m
    """
    backend = model_cfg.backend
    if backend == "pt":
        return AudioModel.load(model_cfg, paths_cfg, device)
    elif backend == "onnx":
        return ONNXAudioModel(model_cfg, paths_cfg, device)
    elif backend == "tflite":
        return TFLiteAudioModel(model_cfg, paths_cfg, device)
    else:
        raise ValueError(
            f"Неизвестный бэкенд: '{backend}'. "
            f"Допустимые значения: 'pt', 'onnx', 'tflite'."
        )


# ── Загрузка аудио (общая для всех бэкендов) ─────────────────────────────

def load_audio(
    filepath,
    sr: int = 32000,
    clip_samples: int = None,
) -> np.ndarray:
    """
    Загружает аудио-файл: ресемплинг → mono → обрезка/паддинг.
    Returns: float32 массив (N,)
    """
    waveform, _ = librosa.load(str(filepath), sr=sr, mono=True)
    if clip_samples is not None:
        if len(waveform) < clip_samples:
            waveform = np.pad(waveform, (0, clip_samples - len(waveform)))
        else:
            waveform = waveform[:clip_samples]
    return waveform.astype(np.float32)