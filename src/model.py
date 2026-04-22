"""
Загрузка модели EfficientAT и инференс.
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

def _add_efficientat_to_path(repo_path: Path) -> None:
    repo_path = repo_path.resolve()
    if not repo_path.exists(): raise FileNotFoundError(f"EfficientAT не найден: {repo_path}")
    if str(repo_path) not in sys.path: sys.path.insert(0, str(repo_path))

def get_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)

def _resolve_export_path(explicit: Optional[Path], exports_dir: Path, model_name: str, suffix: str) -> Path:
    if explicit is not None:
        p = Path(explicit)
        if p.exists(): return p
        raise FileNotFoundError(f"Файл не найден: {p}")
    candidates = [Path(exports_dir) / f"{model_name}{suffix}", Path(f"{model_name}{suffix}")]
    for c in candidates:
        if c.exists(): return c
    raise FileNotFoundError(f"Не найден файл модели {model_name}{suffix}.")

class MelPreprocessor:
    def __init__(self, cfg: ModelConfig, device: torch.device, paths_cfg: PathsConfig):
        _add_efficientat_to_path(paths_cfg.efficientat_repo)
        from models.preprocess import AugmentMelSTFT  # type: ignore
        self.device = device
        self.mel = AugmentMelSTFT(n_mels=cfg.n_mels, sr=cfg.sample_rate, win_length=cfg.win_size, hopsize=cfg.hop_size, n_fft=cfg.n_fft).to(device).eval()

    def __call__(self, waveform_np: np.ndarray) -> np.ndarray:
        waveform = torch.from_numpy(waveform_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            spec = self.mel(waveform).unsqueeze(1)
        return spec.cpu().numpy()

class AudioModel:
    def __init__(self, model, mel_transform, device: torch.device, cfg: ModelConfig):
        self.model, self.mel, self.device, self.cfg = model, mel_transform, device, cfg

    @classmethod
    def load(cls, model_cfg: ModelConfig, paths_cfg: PathsConfig, device: torch.device) -> "AudioModel":
        _add_efficientat_to_path(paths_cfg.efficientat_repo)
        from models.mn.model import get_model as get_mobilenet
        from models.dymn.model import get_model as get_dymn
        from models.preprocess import AugmentMelSTFT
        from helpers.utils import NAME_TO_WIDTH

        name = model_cfg.name
        if name.startswith("dymn"): model = get_dymn(width_mult=NAME_TO_WIDTH(name), pretrained_name=name)
        else: model = get_mobilenet(width_mult=NAME_TO_WIDTH(name), pretrained_name=name, head_type="mlp")
        model.to(device).eval()

        mel = AugmentMelSTFT(n_mels=model_cfg.n_mels, sr=model_cfg.sample_rate, win_length=model_cfg.win_size, hopsize=model_cfg.hop_size, n_fft=model_cfg.n_fft).to(device).eval()
        return cls(model, mel, device, model_cfg)

    def infer_waveform(self, waveform_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        waveform = torch.from_numpy(waveform_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            spec = self.mel(waveform).unsqueeze(1)
            logits, features = self.model(spec)

        # Используем .flatten() для 100% гарантии формы (384,)
        return torch.sigmoid(logits).squeeze(0).cpu().numpy(), features.cpu().numpy().flatten()
    def get_audioset_labels(self):
        from helpers.utils import labels as AUDIOSET_LABELS
        return AUDIOSET_LABELS

    @property
    def n_params_m(self) -> float: return sum(p.numel() for p in self.model.parameters()) / 1e6

class ONNXAudioModel:
    def __init__(self, model_cfg: ModelConfig, paths_cfg: PathsConfig, device: torch.device):
        try: import onnxruntime as ort
        except ImportError: raise ImportError("Установите onnxruntime")

        self.cfg, self.device = model_cfg, device
        onnx_path = _resolve_export_path(model_cfg.onnx_path, paths_cfg.exports_dir, model_cfg.name, ".onnx")

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", 2))
        sess_opts.log_severity_level = 3

        self._sess = ort.InferenceSession(str(onnx_path), sess_opts, providers=["CPUExecutionProvider"])
        self._mel = MelPreprocessor(model_cfg, device, paths_cfg)

        _add_efficientat_to_path(paths_cfg.efficientat_repo)
        from helpers.utils import labels as AUDIOSET_LABELS
        self._labels = AUDIOSET_LABELS

    def infer_waveform(self, waveform_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        spec = self._mel(waveform_np)
        ort_outs = self._sess.run(["logits", "features"], {"spec": spec})

        logits, features = ort_outs[0], ort_outs[1]
        probs = 1.0 / (1.0 + np.exp(-logits[0].astype(np.float32)))

        # Используем .flatten() для 100% гарантии формы (384,)
        return probs, features.flatten()

    def get_audioset_labels(self): return self._labels
    @property
    def n_params_m(self) -> float: return 0.0

class TFLiteAudioModel:
    def __init__(self, model_cfg: ModelConfig, paths_cfg: PathsConfig, device: torch.device):
        try: import tensorflow as tf
        except ImportError: raise ImportError("Установите tensorflow")

        self.cfg, self.device, self._tf = model_cfg, device, tf
        tflite_path = _resolve_export_path(model_cfg.tflite_path, paths_cfg.exports_dir, model_cfg.name, ".tflite")

        self._interp = tf.lite.Interpreter(model_path=str(tflite_path))
        self._interp.allocate_tensors()
        self._inp_details, self._out_details = self._interp.get_input_details(), self._interp.get_output_details()

        inp_shape = list(self._inp_details[0]["shape"])
        self._needs_transpose = (len(inp_shape) == 4 and inp_shape[-1] == 1 and inp_shape[1] > 1)
        self._mel = MelPreprocessor(model_cfg, device, paths_cfg)

        _add_efficientat_to_path(paths_cfg.efficientat_repo)
        from helpers.utils import labels as AUDIOSET_LABELS
        self._labels = AUDIOSET_LABELS

    def infer_waveform(self, waveform_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        waveform = torch.from_numpy(waveform_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            spec = self.mel(waveform).unsqueeze(1)
            logits, features = self.model(spec) # Берем как есть
        return torch.sigmoid(logits).squeeze(0).cpu().numpy(), features.squeeze(0).cpu().numpy()

    def get_audioset_labels(self): return self._labels
    @property
    def n_params_m(self) -> float: return 0.0

def load_model(model_cfg: ModelConfig, paths_cfg: PathsConfig, device: torch.device):
    backend = model_cfg.backend
    if backend == "pt": return AudioModel.load(model_cfg, paths_cfg, device)
    elif backend == "onnx": return ONNXAudioModel(model_cfg, paths_cfg, device)
    elif backend == "tflite": return TFLiteAudioModel(model_cfg, paths_cfg, device)
    else: raise ValueError(f"Неизвестный бэкенд: '{backend}'.")

def load_audio(filepath, sr: int = 32000, clip_samples: int = None) -> np.ndarray:
    waveform, _ = librosa.load(str(filepath), sr=sr, mono=True)
    if clip_samples is not None:
        if len(waveform) < clip_samples: waveform = np.pad(waveform, (0, clip_samples - len(waveform)))
        else: waveform = waveform[:clip_samples]
    return waveform.astype(np.float32)