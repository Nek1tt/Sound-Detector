"""
Центральная конфигурация проекта.
Все параметры — в одном месте.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    """Параметры модели EfficientAT."""
    # Варианты: 'mn04_as', 'mn05_as', 'mn10_as', 'mn20_as', 'dymn10_as'
    # Для Raspberry Pi 4 → mn04_as или mn05_as
    name: str = "mn04_as"

    # Аудио параметры (должны совпадать с тем, на чём обучалась модель)
    sample_rate: int = 32000
    win_size: int = 800
    hop_size: int = 320
    n_fft: int = 1024
    n_mels: int = 128

    # 5 секунд × 32000 Hz = длина клипа в ESC-50
    clip_samples: int = 32000 * 2


@dataclass
class PathsConfig:
    """Пути к данным и артефактам."""
    # Папка с ESC-50 (содержит meta/ и audio/)
    esc50_dir: Path = Path("data/ESC-50-master")

    # Куда сохранять results.json
    outputs_dir: Path = Path("outputs")

    # Клонированный EfficientAT — добавляется в sys.path
    efficientat_repo: Path = Path("third_party/EfficientAT")


@dataclass
class InferenceConfig:
    """Параметры инференса."""
    # fold для тестирования (1–5). None = все folds
    test_fold: Optional[int] = None

    # Максимум потоков CPU (для Pi 4 оптимально 4)
    num_threads: int = 4

    # 'auto' | 'cpu' | 'cuda'
    device: str = "auto"


@dataclass
class DaemonConfig:
    """Параметры фонового режима (daemon / микрофон)."""
    # Длина окна анализа в секундах
    window_seconds: float = 1.0

    # Шаг скользящего окна в секундах
    hop_seconds: float = 0.2

    # Порог уверенности для вывода алерта
    confidence_threshold: float = 0.3

    # Частота дискретизации микрофона (до ресемплинга до 32 kHz)
    mic_sample_rate: int = 48000 #44100

    # Размер чанка PyAudio/sounddevice в сэмплах
    chunk_size: int = 1024


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
