#!/usr/bin/env python3
"""
EfficientAT Audio Classifier — CLI

Команды:
  evaluate  — оценка модели на ESC-50 (Accuracy, F1, mAP → results.json)
  infer     — инференс одного аудио-файла
  daemon    — фоновый режим: анализ из файла (mock) или с микрофона (mic)

Бэкенды (--backend):
  pt      — PyTorch оригинал (по умолчанию, наиболее точный)
  onnx    — ONNX Runtime (быстрее на CPU, нужен onnxruntime)
  tflite  — TensorFlow Lite (минимальная память, нужен tensorflow)

Примеры:
  python main.py evaluate --model mn10_as
  python main.py evaluate --model mn04_as --fold 1

  python main.py infer sounds/dog.wav --top-k 5
  python main.py infer sounds/siren.mp3 --backend onnx
  python main.py infer sounds/rain.wav --backend tflite --tflite-path exports/mn04_as.tflite

  python main.py daemon --mode mock --source sounds/test.wav --loop
  python main.py daemon --mode mock --source sounds/test.wav --backend onnx --loop
  python main.py daemon --mode mic --threshold 0.4 --backend pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.config import AppConfig
from src.model import load_model, AudioModel, get_device, load_audio


# ── Хелпер: применить --backend / --onnx-path / --tflite-path к конфигу ──

def _apply_backend_args(args, cfg: AppConfig) -> None:
    """Переносит аргументы CLI, связанные с бэкендом, в cfg.model."""
    if getattr(args, "backend", None):
        cfg.model.backend = args.backend
    if getattr(args, "onnx_path", None):
        cfg.model.onnx_path = Path(args.onnx_path)
    if getattr(args, "tflite_path", None):
        cfg.model.tflite_path = Path(args.tflite_path)


# ── evaluate ──────────────────────────────────────────────────────────────

def cmd_evaluate(args, cfg: AppConfig) -> None:
    """Полная оценка на датасете ESC-50. Сохраняет results.json."""
    from src.dataset import ESC50Dataset
    from src.evaluate import (
        compute_zeroshot_metrics,
        compute_linear_probe_metrics,
        print_summary,
        per_class_accuracy_report,
    )

    if args.model:
        cfg.model.name = args.model
    if args.esc50_dir:
        cfg.paths.esc50_dir = Path(args.esc50_dir)
    if args.output_dir:
        cfg.paths.outputs_dir = Path(args.output_dir)
    if args.device:
        cfg.inference.device = args.device
    if args.threads:
        cfg.inference.num_threads = args.threads
    _apply_backend_args(args, cfg)

    torch.set_num_threads(cfg.inference.num_threads)

    out_dir = cfg.paths.outputs_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = ESC50Dataset(cfg.paths)
    meta    = dataset.get_fold(args.fold)
    device  = get_device(cfg.inference.device)

    # evaluate поддерживает только PyTorch (нужны эмбеддинги для linear probe)
    if cfg.model.backend != "pt":
        print(
            f"[warn] evaluate поддерживает только backend=pt.\n"
            f"       ONNX/TFLite не экспортируют фичи → linear probe недоступен.\n"
            f"       Принудительно используем backend=pt."
        )
        cfg.model.backend = "pt"

    model = load_model(cfg.model, cfg.paths, device)

    all_probs, all_features, all_labels, all_cats, all_folds = [], [], [], [], []
    errors = []

    print(f"\n[cli] Инференс по {len(meta)} файлам...")
    t_start = time.time()

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="ESC-50"):
        filepath = dataset.audio_path(row["filename"])
        try:
            waveform = load_audio(
                filepath,
                sr=cfg.model.sample_rate,
                clip_samples=cfg.model.clip_samples,
            )
            probs, feats = model.infer_waveform(waveform)
            all_probs.append(probs)
            all_features.append(feats)
            all_labels.append(row["target"])
            all_cats.append(row["category"])
            all_folds.append(row["fold"])
        except Exception as e:
            errors.append((row["filename"], str(e)))

    elapsed = time.time() - t_start
    n = len(all_labels)
    print(
        f"[cli] Готово: {n} файлов за {elapsed:.1f}с "
        f"({elapsed / n * 1000:.0f} мс/файл)"
    )
    if errors:
        print(f"[cli] Ошибок: {len(errors)} — первые 3: {errors[:3]}")

    all_probs    = np.stack(all_probs)
    all_features = np.stack(all_features)
    all_labels   = np.array(all_labels)
    all_folds    = np.array(all_folds)
    all_cats     = np.array(all_cats)

    y_true_50, y_score_50 = dataset.build_score_matrix(all_probs, all_cats, all_labels)
    zs = compute_zeroshot_metrics(y_true_50, y_score_50, dataset.categories)
    lp = compute_linear_probe_metrics(all_features, all_labels, all_folds)

    print_summary(cfg.model.name, n, str(device), model.n_params_m, zs, lp)
    print(per_class_accuracy_report(zs))

    results = {
        "model":   cfg.model.name,
        "backend": cfg.model.backend,
        "device":  str(device),
        "n_files": n,
        "zeroshot": {
            "accuracy":    zs["accuracy"],
            "f1_macro":    zs["f1_macro"],
            "f1_micro":    zs["f1_micro"],
            "f1_weighted": zs["f1_weighted"],
            "mAP":         zs["mAP"],
            "ap_per_class": {
                cat: float(zs["ap_per_class"][i])
                for i, cat in enumerate(dataset.categories)
            },
        },
        "linear_probe": {
            "accuracy_mean":    float(lp["acc"].mean()),
            "accuracy_std":     float(lp["acc"].std()),
            "f1_macro_mean":    float(lp["f1_macro"].mean()),
            "f1_macro_std":     float(lp["f1_macro"].std()),
            "f1_weighted_mean": float(lp["f1_weighted"].mean()),
            "mAP_mean":         float(lp["mAP"].mean()),
            "mAP_std":          float(lp["mAP"].std()),
            "folds":            lp.to_dict(orient="records"),
        },
    }

    results_path = out_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[cli] Результаты сохранены: {results_path}")


# ── infer ─────────────────────────────────────────────────────────────────

def cmd_infer(args, cfg: AppConfig) -> None:
    """Инференс одного аудио-файла."""
    from src.daemon import infer_single_file

    if args.model:
        cfg.model.name = args.model
    if args.device:
        cfg.inference.device = args.device
    _apply_backend_args(args, cfg)

    result = infer_single_file(
        filepath=args.file,
        model_cfg=cfg.model,
        paths_cfg=cfg.paths,
        inf_cfg=cfg.inference,
        top_k=args.top_k,
    )

    backend_label = cfg.model.backend.upper()
    print(f"\n[infer] Файл     : {result['file']}")
    print(f"[infer] Бэкенд   : {backend_label}")
    print(f"[infer] Время    : {result['elapsed_ms']:.0f} мс")
    print(f"\nТоп-{args.top_k} предсказаний:")
    for i, (label, prob) in enumerate(result["top_predictions"], 1):
        bar = "█" * int(prob * 30)
        print(f"  {i:2}. {prob:.3f}  {bar:<30}  {label}")


# ── daemon ────────────────────────────────────────────────────────────────

def cmd_daemon(args, cfg: AppConfig) -> None:
    """Фоновый режим: непрерывный анализ из файла или с микрофона."""
    from src.daemon import AudioDaemon

    if args.model:
        cfg.model.name = args.model
    if args.device:
        cfg.inference.device = args.device
    if args.threshold is not None:
        cfg.daemon.confidence_threshold = args.threshold
    if args.window is not None:
        cfg.daemon.window_seconds = args.window
    if args.hop is not None:
        cfg.daemon.hop_seconds = args.hop
    _apply_backend_args(args, cfg)

    device = get_device(cfg.inference.device)
    model  = load_model(cfg.model, cfg.paths, device)

    print(f"[daemon] Бэкенд: {cfg.model.backend.upper()}")

    def on_result(result):
        ts        = time.strftime("%H:%M:%S", time.localtime(result["timestamp"]))
        lbl, prob = result["top_predictions"][0]
        above     = result["above_threshold"]
        alert     = f"  ← АЛЕРТ ({len(above)} кл.)" if above else ""
        print(f"[{ts}] {result['elapsed_ms']:>5.0f}мс  {prob:.3f}  {lbl[:45]}{alert}")

    daemon = AudioDaemon(model, cfg.daemon, on_result)

    if args.mode == "mock":
        if not args.source:
            print("Ошибка: для --mode mock нужен --source <file.wav>")
            sys.exit(1)
        daemon.start_mock(args.source, loop=args.loop)
    elif args.mode == "mic":
        daemon.start_mic(device_index=args.mic_device)
    else:
        print(f"Неизвестный режим: {args.mode}")
        sys.exit(1)

    print("[daemon] Запущен. Ctrl+C для остановки.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[daemon] Ctrl+C получен")
    finally:
        daemon.stop()


# ── Парсер ────────────────────────────────────────────────────────────────

BACKEND_HELP = "Бэкенд инференса: pt (PyTorch) | onnx (ONNX Runtime) | tflite (TFLite)"
ONNX_PATH_HELP = "Путь к .onnx файлу (авто: exports/<model>.onnx)"
TFLITE_PATH_HELP = "Путь к .tflite файлу (авто: exports/<model>.tflite)"


def _add_backend_args(parser: argparse.ArgumentParser) -> None:
    """Добавляет общие аргументы бэкенда в подпарсер."""
    parser.add_argument("--backend",      default=None, choices=["pt", "onnx", "tflite"],
                        help=BACKEND_HELP)
    parser.add_argument("--onnx-path",    default=None, dest="onnx_path",
                        help=ONNX_PATH_HELP)
    parser.add_argument("--tflite-path",  default=None, dest="tflite_path",
                        help=TFLITE_PATH_HELP)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EfficientAT Audio Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── evaluate ──────────────────────────────────────────────────────
    ev = sub.add_parser("evaluate", help="Оценка модели на ESC-50")
    ev.add_argument("--model",      default=None,
                    help="Имя модели: mn04_as / mn05_as / mn10_as / mn20_as / dymn10_as")
    ev.add_argument("--esc50-dir",  default=None, dest="esc50_dir",
                    help="Путь к ESC-50-master/")
    ev.add_argument("--output-dir", default=None, dest="output_dir",
                    help="Куда сохранять results.json")
    ev.add_argument("--fold",       default=None, type=int,
                    help="Тестировать только один fold 1–5")
    ev.add_argument("--device",     default=None, help="auto / cpu / cuda")
    ev.add_argument("--threads",    default=None, type=int,
                    help="Число потоков CPU")
    _add_backend_args(ev)

    # ── infer ─────────────────────────────────────────────────────────
    inf = sub.add_parser("infer", help="Инференс одного аудио-файла")
    inf.add_argument("file",            help="Путь к .wav / .mp3 / .flac файлу")
    inf.add_argument("--model",  default=None, help="Имя модели")
    inf.add_argument("--top-k",  default=10, type=int, dest="top_k",
                     help="Кол-во топ-предсказаний (по умолч. 10)")
    inf.add_argument("--device", default=None, help="auto / cpu / cuda")
    _add_backend_args(inf)

    # ── daemon ────────────────────────────────────────────────────────
    dm = sub.add_parser("daemon", help="Фоновый анализ: файл или микрофон")
    dm.add_argument("--mode",       default="mock", choices=["mock", "mic"],
                    help="mock — из файла, mic — с микрофона")
    dm.add_argument("--source",     default=None,
                    help="Аудио-файл для mock-режима")
    dm.add_argument("--loop",       action="store_true",
                    help="Зациклить файл в mock-режиме")
    dm.add_argument("--model",      default=None, help="Имя модели")
    dm.add_argument("--device",     default=None, help="auto / cpu / cuda")
    dm.add_argument("--threshold",  default=None, type=float,
                    help="Порог уверенности для АЛЕРТ (0–1)")
    dm.add_argument("--window",     default=None, type=float,
                    help="Длина окна анализа в секундах")
    dm.add_argument("--hop",        default=None, type=float,
                    help="Шаг окна в секундах")
    dm.add_argument("--mic-device", default=None, type=int, dest="mic_device",
                    help="Индекс устройства ввода")
    _add_backend_args(dm)

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()
    cfg    = AppConfig()

    if args.command == "evaluate":
        cmd_evaluate(args, cfg)
    elif args.command == "infer":
        cmd_infer(args, cfg)
    elif args.command == "daemon":
        cmd_daemon(args, cfg)


if __name__ == "__main__":
    main()