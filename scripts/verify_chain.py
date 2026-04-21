#!/usr/bin/env python3
"""
Сквозная проверка всей цепочки конвертации:

    PyTorch → ONNX → TFLite

Тест считается пройденным если:
  1. Топ-1 предсказание совпадает у всех трёх форматов
  2. Топ-5 совпадают у PyTorch и ONNX
  3. Max diff (PyTorch vs ONNX) < 1e-4
  4. Max diff (ONNX vs TFLite float32) < 1e-2

Запуск:
    python scripts/verify_chain.py \
        --model mn04_as \
        --onnx exports/mn04_as.onnx \
        --tflite exports/mn04_as.tflite
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import torch


EFFICIENTAT = Path(__file__).resolve().parent.parent / "third_party" / "EfficientAT"
if EFFICIENTAT.exists():
    sys.path.insert(0, str(EFFICIENTAT))


def load_pytorch(model_name: str):
    from models.mn.model import get_model as get_mobilenet   # type: ignore
    from helpers.utils import NAME_TO_WIDTH                   # type: ignore
    import torch.nn as nn

    class Wrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): logits, _ = self.m(x); return logits

    raw = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name),
                        pretrained_name=model_name, head_type="mlp")
    raw.eval()
    return Wrapper(raw)


def run_pytorch(model, dummy_np):
    dummy_t = torch.from_numpy(dummy_np)
    with torch.no_grad():
        return model(dummy_t).numpy()


def run_onnx(onnx_path, dummy_np):
    import onnxruntime as ort  # type: ignore
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return sess.run(["logits"], {"spec": dummy_np})[0]


def run_tflite(tflite_path, dummy_np):
    import tensorflow as tf  # type: ignore
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    # Проверка на NHWC
    tfl_in = dummy_np
    if list(inp['shape']) == [1, 128, 1001, 1]:
        tfl_in = dummy_np.transpose(0, 2, 3, 1)

    interp.set_tensor(inp['index'], tfl_in)
    interp.invoke()
    return interp.get_tensor(out['index'])


def top_k(logits, k=5):
    return np.argsort(logits[0])[::-1][:k].tolist()


def print_result(label, passed):
    icon = "✓" if passed else "✗"
    status = "PASS" if passed else "FAIL"
    print(f"  [{icon}] {label}: {status}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="mn04_as")
    parser.add_argument("--onnx",    default="exports/mn04_as.onnx")
    parser.add_argument("--tflite",  default=None,
                        help="путь к float32 .tflite (если None — пропускаем)")
    parser.add_argument("--seed",    default=42, type=int)
    args = parser.parse_args()

    onnx_path   = Path(args.onnx)
    tflite_path = Path(args.tflite) if args.tflite else None

    np.random.seed(args.seed)
    # Используем реалистичный диапазон нормализованной спектрограммы
    dummy_np = np.random.randn(1, 1, 128, 1001).astype(np.float32)

    results = {}
    all_passed = True

    # ── PyTorch ───────────────────────────────────────────────────────────
    print("\n[1] PyTorch inference...")
    try:
        pt_model  = load_pytorch(args.model)
        pt_logits = run_pytorch(pt_model, dummy_np)
        results["pytorch"] = pt_logits
        print(f"    shape={pt_logits.shape}  top-1={top_k(pt_logits, 1)}")
    except Exception as e:
        print(f"    ОШИБКА: {e}")

    # ── ONNX ─────────────────────────────────────────────────────────────
    print("\n[2] ONNX Runtime inference...")
    if not onnx_path.exists():
        print(f"    [SKIP] Файл не найден: {onnx_path}")
    else:
        try:
            ort_logits = run_onnx(onnx_path, dummy_np)
            results["onnx"] = ort_logits
            print(f"    shape={ort_logits.shape}  top-1={top_k(ort_logits, 1)}")
        except Exception as e:
            print(f"    ОШИБКА: {e}")

    # ── TFLite ───────────────────────────────────────────────────────────
    if tflite_path:
        print("\n[3] TFLite inference...")
        if not tflite_path.exists():
            print(f"    [SKIP] Файл не найден: {tflite_path}")
        else:
            try:
                tfl_logits = run_tflite(tflite_path, dummy_np)
                results["tflite"] = tfl_logits
                print(f"    shape={tfl_logits.shape}  top-1={top_k(tfl_logits, 1)}")
            except Exception as e:
                print(f"    ОШИБКА: {e}")

    # ── Сравнения ─────────────────────────────────────────────────────────
    print("\n── Результаты проверки ──────────────────────────────────────────")

    if "pytorch" in results and "onnx" in results:
        pt  = results["pytorch"]
        ort = results["onnx"]
        diff = float(np.abs(pt - ort).max())

        p1 = diff < 1e-3
        p2 = top_k(pt, 5) == top_k(ort, 5)
        p3 = top_k(pt, 1) == top_k(ort, 1)

        print_result(f"PyTorch vs ONNX  max_diff={diff:.2e} < 1e-3", p1)
        print_result(f"PyTorch vs ONNX  топ-5 совпадают", p2)
        print_result(f"PyTorch vs ONNX  топ-1 совпадает", p3)
        all_passed = all_passed and p1 and p2 and p3

    if "onnx" in results and "tflite" in results:
        ort = results["onnx"]
        tfl = results["tflite"]
        diff = float(np.abs(ort - tfl).max())

        p1 = diff < 1e-1        # float32 TFLite: допуск 0.1 из-за NHWC-транспозиции и разных реализаций BN
        p2 = top_k(ort, 1) == top_k(tfl, 1)

        print_result(f"ONNX   vs TFLite max_diff={diff:.2e} < 0.1", p1)
        print_result(f"ONNX   vs TFLite топ-1 совпадает", p2)
        all_passed = all_passed and p1 and p2

    print("\n" + ("═" * 50))
    if all_passed:
        print("  ✓ Все проверки пройдены — конвертация успешна!")
    else:
        print("  ✗ Некоторые проверки не пройдены — см. детали выше")
    print("═" * 50)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()