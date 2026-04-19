"""
Вычисление метрик оценки: Accuracy, F1, mAP.

Два режима:
  1. Zero-shot  — маппинг ESC-50 категорий → AudioSet индексы
  2. Linear probe — логистическая регрессия поверх эмбеддингов (5-fold CV)
"""
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def compute_zeroshot_metrics(
    y_true_50: np.ndarray,
    y_score_50: np.ndarray,
    categories: List[str],
) -> Dict:
    """
    Zero-shot метрики через маппинг AudioSet → ESC-50.

    Args:
        y_true_50  : (N, 50) one-hot ground-truth
        y_score_50 : (N, 50) агрегированные AudioSet-вероятности

    Returns:
        dict с ключами: mAP, accuracy, f1_macro, f1_micro, f1_weighted,
                        ap_per_class, y_pred, y_true, categories
    """
    ap_per_class = metrics.average_precision_score(
        y_true_50, y_score_50, average=None
    )
    mAP = float(ap_per_class.mean())

    y_pred    = y_score_50.argmax(axis=1)
    y_true_cls = y_true_50.argmax(axis=1)

    return {
        "mAP":          mAP,
        "accuracy":     float(metrics.accuracy_score(y_true_cls, y_pred)),
        "f1_macro":     float(metrics.f1_score(y_true_cls, y_pred, average="macro",    zero_division=0)),
        "f1_micro":     float(metrics.f1_score(y_true_cls, y_pred, average="micro",    zero_division=0)),
        "f1_weighted":  float(metrics.f1_score(y_true_cls, y_pred, average="weighted", zero_division=0)),
        "ap_per_class": ap_per_class,   # (50,) — для JSON сериализуется отдельно
        "y_pred":       y_pred,
        "y_true":       y_true_cls,
        "categories":   categories,
    }


def compute_linear_probe_metrics(
    all_features: np.ndarray,
    all_labels: np.ndarray,
    all_folds: np.ndarray,
    n_folds: int = 5,
) -> pd.DataFrame:
    """
    5-fold CV: логистическая регрессия поверх эмбеддингов модели.
    Стандартная оценка качества представлений (transfer learning).

    Returns:
        DataFrame с колонками: fold, acc, f1_macro, f1_weighted, mAP
    """
    fold_results = []
    print("[eval] 5-fold linear probe CV:")

    for fold in range(1, n_folds + 1):
        test_mask  = all_folds == fold
        train_mask = ~test_mask

        X_train = all_features[train_mask]
        X_test  = all_features[test_mask]
        y_train = all_labels[train_mask]
        y_test  = all_labels[test_mask]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        n_cls     = len(np.unique(all_labels))
        y_test_oh = np.eye(n_cls)[y_test]
        ap_fold   = metrics.average_precision_score(y_test_oh, y_prob, average="macro")

        acc  = metrics.accuracy_score(y_test, y_pred)
        f1_m = metrics.f1_score(y_test, y_pred, average="macro",    zero_division=0)
        f1_w = metrics.f1_score(y_test, y_pred, average="weighted", zero_division=0)

        fold_results.append({
            "fold": fold, "acc": acc,
            "f1_macro": f1_m, "f1_weighted": f1_w, "mAP": ap_fold,
        })
        print(f"  Fold {fold}: Acc={acc:.4f}  F1_macro={f1_m:.4f}  mAP={ap_fold:.4f}")

    return pd.DataFrame(fold_results)


def print_summary(
    model_name: str,
    n_files: int,
    device: str,
    n_params_m: float,
    zs: Dict,
    lp: pd.DataFrame,
) -> None:
    """Итоговый отчёт в консоль."""
    sep = "═" * 58
    print(sep)
    print(f"{'ИТОГОВЫЙ ОТЧЁТ':^58}")
    print(sep)
    print(f"  Модель      : {model_name}")
    print(f"  Датасет     : ESC-50 ({n_files} файлов, 50 классов)")
    print(f"  Устройство  : {device}")
    print(f"  Параметров  : {n_params_m:.2f}M")
    print()
    print("  ── Zero-shot (AudioSet → ESC-50 маппинг) ──────────")
    print(f"  Accuracy    : {zs['accuracy']:.4f}  ({zs['accuracy']*100:.1f}%)")
    print(f"  F1 macro    : {zs['f1_macro']:.4f}")
    print(f"  F1 micro    : {zs['f1_micro']:.4f}")
    print(f"  F1 weighted : {zs['f1_weighted']:.4f}")
    print(f"  mAP         : {zs['mAP']:.4f}  ({zs['mAP']*100:.1f}%)")
    print()
    print("  ── Линейный зонд на эмбеддингах (5-fold CV) ───────")
    print(f"  Accuracy    : {lp['acc'].mean():.4f} ± {lp['acc'].std():.4f}")
    print(f"  F1 macro    : {lp['f1_macro'].mean():.4f} ± {lp['f1_macro'].std():.4f}")
    print(f"  F1 weighted : {lp['f1_weighted'].mean():.4f} ± {lp['f1_weighted'].std():.4f}")
    print(f"  mAP         : {lp['mAP'].mean():.4f} ± {lp['mAP'].std():.4f}")
    print(sep)


def per_class_accuracy_report(zs: Dict) -> str:
    """
    Текстовый отчёт по точности каждого из 50 классов.
    Возвращает строку — можно печатать или сохранять в файл.
    """
    categories  = zs["categories"]
    y_true      = zs["y_true"]
    y_pred      = zs["y_pred"]
    ap_per_class = zs["ap_per_class"]

    lines = ["", "Per-class Accuracy & AP (zero-shot):", "-" * 46]
    rows = []
    for i, cat in enumerate(categories):
        mask = y_true == i
        acc_i = float((y_pred[mask] == i).mean()) if mask.sum() > 0 else 0.0
        rows.append((cat, acc_i, float(ap_per_class[i])))

    rows.sort(key=lambda x: x[1])  # по возрастанию accuracy
    for cat, acc_i, ap_i in rows:
        bar = "█" * int(acc_i * 20)
        lines.append(f"  {cat:<22} acc={acc_i:.3f}  ap={ap_i:.3f}  {bar}")

    return "\n".join(lines)
