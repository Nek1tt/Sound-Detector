"""
Загрузка датасета ESC-50 и маппинг категорий на классы AudioSet.
"""
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import PathsConfig


# ── Маппинг: категория ESC-50 → индекс(ы) класса AudioSet (из 527) ───────
ESC50_TO_AUDIOSET: Dict[str, List[int]] = {
    # Animals
    "dog":               [74],
    "rooster":           [101],
    "pig":               [93],
    "cow":               [90],
    "frog":              [132],
    "cat":               [81],
    "hen":               [99],
    "insects":           [126],
    "sheep":             [97],
    "crow":              [117],
    # Natural soundscapes
    "rain":              [289],
    "sea_waves":         [294],
    "crackling_fire":    [298],
    "crickets":          [127],
    "chirping_birds":    [112],
    "water_drops":       [448],
    "wind":              [283],
    "pouring_water":     [449],
    "toilet_flush":      [374],
    "thunderstorm":      [286],
    # Human non-speech
    "crying_baby":       [23],
    "sneezing":          [49],
    "clapping":          [63],
    "breathing":         [41],
    "coughing":          [47],
    "footsteps":         [53],
    "laughing":          [16],
    "brushing_teeth":    [375],
    "snoring":           [43],
    "drinking_sipping":  [54],
    # Interior/domestic
    "door_wood_knock":   [359],
    "mouse_click":       [491],
    "keyboard_typing":   [384],
    "door_wood_creaks":  [486],
    "can_opening":       [444],
    "washing_machine":   [412],
    "vacuum_cleaner":    [377],
    "clock_alarm":       [395],
    "clock_tick":        [407],
    "glass_breaking":    [443],
    # Exterior/urban
    "helicopter":        [339],
    "chainsaw":          [347],
    "siren":             [396],
    "car_horn":          [308],
    "engine":            [343],
    "train":             [329],
    "church_bells":      [201],
    "airplane":          [340],
    "fireworks":         [432],
    "hand_saw":          [421],
}


class ESC50Dataset:
    """
    Обёртка над метаданными датасета ESC-50.

    Атрибуты:
        meta        — DataFrame: filename, fold, target, category
        audio_dir   — Path к папке с .wav файлами
        categories  — отсортированный список 50 категорий
    """

    def __init__(self, paths_cfg: PathsConfig):
        esc50_dir = Path(paths_cfg.esc50_dir)
        if not esc50_dir.exists():
            raise FileNotFoundError(
                f"ESC-50 не найден: {esc50_dir}\n"
                f"Запустите: python scripts/download_data.py"
            )

        self.meta       = pd.read_csv(esc50_dir / "meta" / "esc50.csv")
        self.audio_dir  = esc50_dir / "audio"
        self.categories = sorted(self.meta["category"].unique().tolist())

        missing = set(self.categories) - set(ESC50_TO_AUDIOSET.keys())
        if missing:
            print(f"[dataset] Нет маппинга для категорий: {missing}")

        print(
            f"[dataset] ESC-50: {len(self.meta)} файлов | "
            f"{len(self.categories)} категорий | "
            f"{self.meta['fold'].nunique()} folds"
        )

    def get_fold(self, fold: Optional[int]) -> pd.DataFrame:
        """fold=None → все записи; fold=1..5 → конкретный fold."""
        if fold is None:
            return self.meta
        return self.meta[self.meta["fold"] == fold]

    def audio_path(self, filename: str) -> Path:
        return self.audio_dir / filename

    def build_score_matrix(
        self,
        all_probs: np.ndarray,
        all_cats: np.ndarray,
        all_labels: np.ndarray,
    ):
        """
        Строит матрицы для zero-shot оценки.

        Возвращает:
            y_true_50  : (N, 50) one-hot ground-truth
            y_score_50 : (N, 50) агрегированные вероятности через AudioSet-маппинг
        """
        N = len(all_labels)
        mapped_indices = [ESC50_TO_AUDIOSET[cat] for cat in self.categories]

        y_true_50  = np.zeros((N, 50), dtype=np.float32)
        y_score_50 = np.zeros((N, 50), dtype=np.float32)

        for i, (label, cat) in enumerate(zip(all_labels, all_cats)):
            esc_idx = self.categories.index(cat)
            y_true_50[i, esc_idx] = 1.0
            for c_idx, as_indices in enumerate(mapped_indices):
                y_score_50[i, c_idx] = all_probs[i, as_indices].max()

        return y_true_50, y_score_50
