#!/usr/bin/env python3
"""
EfficientAT Audio Classifier — CLI

Команды:
  evaluate  — оценка модели на ESC-50 (Accuracy, F1, mAP → results.json)
  infer     — инференс одного аудио-файла
  daemon    — фоновый режим: анализ из файла (mock) или с микрофона (mic)

Примеры:
  python main.py evaluate --model mn10_as
  python main.py evaluate --model mn04_as --fold 1
  python main.py infer sounds/dog.wav --top-k 5
  python main.py daemon --mode mock --source sounds/test.wav --loop
  python main.py daemon --mode mic --threshold 0.4
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
from src.model import AudioModel, get_device, load_audio


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

    # CLI-аргументы перекрывают дефолты конфига
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

    torch.set_num_threads(cfg.inference.num_threads)

    out_dir = cfg.paths.outputs_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Данные и модель
    dataset  = ESC50Dataset(cfg.paths)
    meta     = dataset.get_fold(args.fold)
    device   = get_device(cfg.inference.device)
    model    = AudioModel.load(cfg.model, cfg.paths, device)

    # ── Инференс по всем файлам ───────────────────────────────────────
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

    # ── Метрики ───────────────────────────────────────────────────────
    y_true_50, y_score_50 = dataset.build_score_matrix(
        all_probs, all_cats, all_labels
    )
    zs = compute_zeroshot_metrics(y_true_50, y_score_50, dataset.categories)
    lp = compute_linear_probe_metrics(all_features, all_labels, all_folds)

    print_summary(cfg.model.name, n, str(device), model.n_params_m, zs, lp)

    # Текстовый отчёт по классам
    report = per_class_accuracy_report(zs)
    print(report)

    # ── Сохранение результатов ────────────────────────────────────────
    results = {
        "model":   cfg.model.name,
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

    result = infer_single_file(
        filepath=args.file,
        model_cfg=cfg.model,
        paths_cfg=cfg.paths,
        inf_cfg=cfg.inference,
        top_k=args.top_k,
    )

    print(f"\n[infer] Файл     : {result['file']}")
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

    device = get_device(cfg.inference.device)
    model  = AudioModel.load(cfg.model, cfg.paths, device)

    def on_result(result):
        ts  = time.strftime("%H:%M:%S", time.localtime(result["timestamp"]))
        lbl, prob = result["top_predictions"][0]
        above = result["above_threshold"]
        alert = f"  ← АЛЕРТ ({len(above)} кл.)" if above else ""
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

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EfficientAT Audio Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # evaluate
    ev = sub.add_parser("evaluate", help="Оценка модели на ESC-50")
    ev.add_argument("--model",      default=None,
                    help="Имя модели: mn04_as / mn05_as / mn10_as / mn20_as / dymn10_as")
    ev.add_argument("--esc50-dir",  default=None, dest="esc50_dir",
                    help="Путь к ESC-50-master/  (по умолч. data/ESC-50-master)")
    ev.add_argument("--output-dir", default=None, dest="output_dir",
                    help="Куда сохранять results.json  (по умолч. outputs/)")
    ev.add_argument("--fold",       default=None, type=int,
                    help="Тестировать только один fold 1–5 (по умолч. все)")
    ev.add_argument("--device",     default=None, help="auto / cpu / cuda")
    ev.add_argument("--threads",    default=None, type=int,
                    help="Число потоков CPU (рекомендуется 4 для Pi 4)")

    # infer
    inf = sub.add_parser("infer", help="Инференс одного аудио-файла")
    inf.add_argument("file",            help="Путь к .wav / .mp3 / .flac файлу")
    inf.add_argument("--model",  default=None, help="Имя модели")
    inf.add_argument("--top-k",  default=10, type=int, dest="top_k",
                     help="Кол-во топ-предсказаний  (по умолч. 10)")
    inf.add_argument("--device", default=None, help="auto / cpu / cuda")

    # daemon
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
                    help="Порог уверенности для АЛЕРТ (0–1, по умолч. 0.3)")
    dm.add_argument("--window",     default=None, type=float,
                    help="Длина окна анализа в секундах (по умолч. 5.0)")
    dm.add_argument("--hop",        default=None, type=float,
                    help="Шаг окна в секундах (по умолч. 0.5)")
    dm.add_argument("--mic-device", default=None, type=int, dest="mic_device",
                    help="Индекс устройства ввода (узнать: python -c "
                         "\"import sounddevice; print(sounddevice.query_devices())\")")

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

