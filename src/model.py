"""
Загрузка модели EfficientAT и инференс.
Поддерживает MobileNet (mn*_as) и DyMN (dymn*_as).
"""
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import librosa

from src.config import ModelConfig, PathsConfig


def _add_efficientat_to_path(repo_path: Path) -> None:
    """Добавляет репозиторий EfficientAT в sys.path."""
    repo_path = repo_path.resolve()
    if not repo_path.exists():
        raise FileNotFoundError(
            f"EfficientAT не найден: {repo_path}\n"
            f"Выполните: git clone https://github.com/fschmid56/EfficientAT.git {repo_path}"
        )
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))


def get_device(device_str: str = "auto") -> torch.device:
    """Возвращает torch.device по строке 'auto' / 'cpu' / 'cuda'."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


class AudioModel:
    """
    Обёртка над EfficientAT: загрузка весов, мел-трансформ, инференс.

    Использование:
        model = AudioModel.load(model_cfg, paths_cfg, device)
        probs, features = model.infer_waveform(waveform_np)
    """

    def __init__(
        self,
        model,
        mel_transform,
        device: torch.device,
        cfg: ModelConfig,
    ):
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
        """Загружает предобученную модель EfficientAT с HuggingFace."""
        _add_efficientat_to_path(paths_cfg.efficientat_repo)

        # Импорты из EfficientAT (только после добавления пути)
        from models.mn.model import get_model as get_mobilenet    # type: ignore
        from models.dymn.model import get_model as get_dymn       # type: ignore
        from models.preprocess import AugmentMelSTFT               # type: ignore
        from helpers.utils import NAME_TO_WIDTH                    # type: ignore

        name = model_cfg.name
        print(f"[model] Загружаем {name}...")
        t0 = time.time()

        if name.startswith("dymn"):
            model = get_dymn(
                width_mult=NAME_TO_WIDTH(name),
                pretrained_name=name,
            )
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
        )
        mel.to(device).eval()

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(
            f"[model] Готово за {time.time() - t0:.1f}с | "
            f"Параметров: {n_params:.2f}M | Устройство: {device}"
        )

        return cls(model, mel, device, model_cfg)

    def infer_waveform(
        self, waveform_np: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Инференс одного моно аудио-фрагмента.

        Цепочка размерностей:
            waveform_np : (N,)
            → waveform  : (1, N)     батч=1
            → spec      : (1, 128, T) мел-спектрограмма
            → spec      : (1, 1, 128, T) добавляем канал для conv2d
            → logits    : (1, 527)
            → features  : (1, C)

        Returns:
            probs    : np.ndarray (527,)  sigmoid-вероятности AudioSet классов
            features : np.ndarray (C,)   эмбеддинг последнего слоя
        """
        waveform = torch.from_numpy(waveform_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            spec = self.mel(waveform)     # (1, n_mels, T)
            spec = spec.unsqueeze(1)      # (1, 1, n_mels, T)
            logits, features = self.model(spec)

        probs    = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        features = features.squeeze(0).cpu().numpy()
        return probs, features

    def get_audioset_labels(self):
        """Возвращает список 527 меток AudioSet."""
        from helpers.utils import labels as AUDIOSET_LABELS  # type: ignore
        return AUDIOSET_LABELS

    @property
    def n_params_m(self) -> float:
        return sum(p.numel() for p in self.model.parameters()) / 1e6


def load_audio(
    filepath,
    sr: int = 32000,
    clip_samples: int = None,
) -> np.ndarray:
    """
    Загружает аудио-файл: ресемплинг → mono → обрезка/паддинг.

    Returns:
        float32 массив (N,)
    """
    waveform, _ = librosa.load(str(filepath), sr=sr, mono=True)
    if clip_samples is not None:
        if len(waveform) < clip_samples:
            waveform = np.pad(waveform, (0, clip_samples - len(waveform)))
        else:
            waveform = waveform[:clip_samples]
    return waveform.astype(np.float32)
