#!/usr/bin/env python3
"""
Конвертация ONNX → TFLite (через TF SavedModel как промежуточный формат)

Цепочка:
    mn04_as.onnx
        │
        ▼  onnx2tf (рекомендуется) ИЛИ onnx-tf
    SavedModel/
        │
        ▼  tf.lite.TFLiteConverter
    mn04_as.tflite          ← float32, полная точность
    mn04_as_fp16.tflite     ← float16, ~2x меньше, совместимо с GPU delegate
    mn04_as_int8.tflite     ← int8 quantization, ~4x меньше, быстро на CPU/EdgeTPU

Требования (выберите один конвертер):
    # Вариант A — onnx2tf (рекомендуется, лучше поддерживает MobileNet):
    pip install onnx2tf tensorflow

    # Вариант B — onnx-tf (классика, но отстаёт от новых opset):
    pip install onnx-tf tensorflow

Использование:
    python scripts/convert_to_tflite.py --onnx exports/mn04_as.onnx
    python scripts/convert_to_tflite.py --onnx exports/mn04_as.onnx --quant fp16
    python scripts/convert_to_tflite.py --onnx exports/mn04_as.onnx --quant int8
"""
import sys
import argparse
import shutil
from pathlib import Path

import numpy as np


# ══════════════════════════════════════════════════════════════════════════
# Шаг A: ONNX → TF SavedModel
# ══════════════════════════════════════════════════════════════════════════

def onnx_to_saved_model_via_onnx2tf(onnx_path: Path, saved_model_dir: Path) -> bool:
    """
    Конвертация через onnx2tf с предварительной оптимизацией onnxsim.
    """
    try:
        import onnx2tf  # type: ignore
        import onnx
        from onnxsim import simplify
    except ImportError as e:
        print(f"[A] Ошибка импорта библиотек: {e}")
        return False

    # 1. Сначала оптимизируем модель и жестко фиксируем размеры через Python API
    print("[A] Оптимизируем граф ONNX (onnxsim)...")
    opt_onnx_path = onnx_path.parent / f"{onnx_path.stem}_opt.onnx"
    try:
        model = onnx.load(str(onnx_path))
        # Заменяем все динамические размеры ('batch', 'time_frames') на статические
        model_simp, check = simplify(
            model,
            test_input_shapes={'spec':[1, 1, 128, 1001]}
        )
        if not check:
            print("[A] ПРЕДУПРЕЖДЕНИЕ: onnxsim отработал, но верификация не пройдена.")
        onnx.save(model_simp, str(opt_onnx_path))
        print(f"[A] Оптимизированная модель сохранена: {opt_onnx_path}")
    except Exception as e:
        print(f"[A] Ошибка onnxsim: {e}")
        return False

    # 2. Теперь передаем чистую, оптимизированную модель в onnx2tf
    print("[A] Конвертируем оптимизированный ONNX → SavedModel через onnx2tf...")
    saved_model_dir.mkdir(parents=True, exist_ok=True)
    try:
        onnx2tf.convert(
            input_onnx_file_path=str(opt_onnx_path),
            output_folder_path=str(saved_model_dir),
            non_verbose=True,  # Можно включить True, чтобы не мусорить в консоли
            output_integer_quantized_tflite=False,
            # Дублируем статические размеры для надежности
            batch_size=1,
            overwrite_input_shape=['spec:1,1,128,1001'],
        )
        print(f"[A] SavedModel сохранён: {saved_model_dir}")
        
        # Удаляем временный оптимизированный ONNX
        if opt_onnx_path.exists():
            opt_onnx_path.unlink()
            
        return True
    except Exception as e:
        print(f"[A] onnx2tf ОШИБКА: {e}")
        return False


def onnx_to_saved_model_via_onnxtf(onnx_path: Path, saved_model_dir: Path) -> bool:
    """
    Конвертация через onnx-tf (fallback).
    Возвращает True при успехе.
    """
    try:
        import onnx                      # type: ignore
        from onnx_tf.backend import prepare  # type: ignore
    except ImportError:
        return False

    print("[A] Конвертируем ONNX → SavedModel через onnx-tf...")
    try:
        model_onnx = onnx.load(str(onnx_path))
        tf_rep = prepare(model_onnx)
        saved_model_dir.mkdir(parents=True, exist_ok=True)
        tf_rep.export_graph(str(saved_model_dir))
        print(f"[A] SavedModel сохранён: {saved_model_dir}")
        return True
    except Exception as e:
        print(f"[A] onnx-tf ОШИБКА: {e}")
        return False


def onnx_to_saved_model(onnx_path: Path, saved_model_dir: Path) -> Path:
    """
    Пробует onnx2tf, затем onnx-tf, затем падает с понятным сообщением.
    """
    if onnx_to_saved_model_via_onnx2tf(onnx_path, saved_model_dir):
        return saved_model_dir
    if onnx_to_saved_model_via_onnxtf(onnx_path, saved_model_dir):
        return saved_model_dir
    sys.exit(
        "\n[ERROR] Не найден ни onnx2tf, ни onnx-tf.\n"
        "Установите один из вариантов:\n"
        "  pip install onnx2tf tensorflow    # рекомендуется\n"
        "  pip install onnx-tf tensorflow    # альтернатива\n"
    )


# ══════════════════════════════════════════════════════════════════════════
# Шаг B: TF SavedModel → TFLite
# ══════════════════════════════════════════════════════════════════════════

def representative_dataset_gen(n_samples: int = 100):
    """
    Генератор репрезентативных данных для int8 quantization.
    ...
    """
    for _ in range(n_samples):
        # Внимание! onnx2tf переставил оси на NHWC [batch, height, width, channels]
        # Значит: batch=1, n_mels=128, T=1001, channel=1
        sample = np.random.uniform(-1.0, 1.0, (1, 128, 1001, 1)).astype(np.float32)
        yield [sample]


def saved_model_to_tflite(
    saved_model_dir: Path,
    tflite_path: Path,
    quantization: str = "none",   # "none" | "fp16" | "int8"
) -> Path:
    """
    Конвертация TF SavedModel → TFLite с опциональной квантизацией.

    quantization:
        "none"  — float32, максимальная точность
        "fp16"  — float16, ~2x меньше, без потери точности на GPU/NNAPI
        "int8"  — int8, ~4x меньше, подходит для EdgeTPU и мобильных CPU
    """
    try:
        import tensorflow as tf  # type: ignore
    except ImportError:
        sys.exit("[ERROR] TensorFlow не установлен. pip install tensorflow")

    print(f"\n[B] Конвертируем SavedModel → TFLite (квантизация: {quantization})...")

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

    if quantization == "none":
        # float32 — без изменений
        pass

    elif quantization == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    elif quantization == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,  # fallback для операций без int8
        ]
        # Вход/выход оставляем float32 для удобства (иначе нужен ручной квант. вход)
        converter.inference_input_type  = tf.float32
        converter.inference_output_type = tf.float32
    else:
        raise ValueError(f"Неизвестный режим квантизации: {quantization}")

    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"[B] Ошибка конвертации: {e}")
        raise

    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)

    size_mb = tflite_path.stat().st_size / 1024 / 1024
    print(f"[B] Сохранено: {tflite_path}  ({size_mb:.1f} MB)")
    return tflite_path


# ══════════════════════════════════════════════════════════════════════════
# Численная верификация TFLite
# ══════════════════════════════════════════════════════════════════════════

def verify_tflite(tflite_path: Path, onnx_path: Path, atol: float = 1e-2):
    """
    Сравнивает выходы TFLite-интерпретатора и ONNX Runtime на одном входе.
    atol=1e-2 — допуск для fp16/int8 (для float32 можно ставить 1e-4).
    """
    print(f"\n── Численная верификация TFLite ({tflite_path.name}) ──────────────")

    try:
        import tensorflow as tf  # type: ignore
    except ImportError:
        print("  [SKIP] tensorflow не установлен")
        return

    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        print("  [SKIP] onnxruntime не установлен, пропускаем сравнение с ONNX")
        ort = None

    np.random.seed(42)
    dummy_np = np.random.randn(1, 1, 128, 1001).astype(np.float32)

    # ── TFLite inference ──────────────────────────────────────────────────
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"  TFLite вход : {input_details[0]['name']}  shape={input_details[0]['shape']}")
    print(f"  TFLite выход: {output_details[0]['name']}  shape={output_details[0]['shape']}")

    # Некоторые конвертеры меняют порядок осей NCHW→NHWC — проверяем
    expected_shape = input_details[0]['shape']
    if list(expected_shape) == [1, 128, 1001, 1]:
        # Конвертер переставил оси в NHWC
        print("  [!] Конвертер переставил оси в NHWC, транспонируем вход...")
        tflite_input = dummy_np.transpose(0, 2, 3, 1)  # NCHW → NHWC
    else:
        tflite_input = dummy_np

    interpreter.set_tensor(input_details[0]['index'], tflite_input)
    interpreter.invoke()
    tflite_logits = interpreter.get_tensor(output_details[0]['index'])

    # ── ONNX Runtime inference ────────────────────────────────────────────
    if ort is not None and onnx_path.exists():
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_logits = sess.run(["logits"], {"spec": dummy_np})[0]

        max_diff  = float(np.abs(tflite_logits - ort_logits).max())
        mean_diff = float(np.abs(tflite_logits - ort_logits).mean())
        print(f"  Max  diff (TFLite vs ONNX): {max_diff:.2e}  {'✓ OK' if max_diff < atol else '⚠ ВЫШЕ ПОРОГА'}")
        print(f"  Mean diff (TFLite vs ONNX): {mean_diff:.2e}")

        top5_ort   = np.argsort(ort_logits[0])[::-1][:5].tolist()
        top5_tfl   = np.argsort(tflite_logits[0])[::-1][:5].tolist()
        print(f"  Топ-5 ONNX  : {top5_ort}")
        print(f"  Топ-5 TFLite: {top5_tfl}")
        match = top5_ort == top5_tfl
        print(f"  Топ-5 совпадают: {'✓ ДА' if match else '⚠ НЕТ (допустимо для int8)'}")
        return max_diff
    else:
        print(f"  TFLite logits shape: {tflite_logits.shape}")
        print(f"  TFLite top-5 indices: {np.argsort(tflite_logits[0])[::-1][:5].tolist()}")


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Convert ONNX model to TFLite")
    parser.add_argument("--onnx", required=True,
                        help="Путь к .onnx файлу (напр. exports/mn04_as.onnx)")
    parser.add_argument("--output-dir", default=None, dest="output_dir",
                        help="Папка для .tflite (по умолч. рядом с .onnx)")
    parser.add_argument("--quant", default="none",
                        choices=["none", "fp16", "int8"],
                        help="Квантизация: none | fp16 | int8")
    parser.add_argument("--all-quant", action="store_true", dest="all_quant",
                        help="Экспортировать все три варианта квантизации")
    parser.add_argument("--verify", action="store_true",
                        help="Запустить численную верификацию")
    parser.add_argument("--keep-saved-model", action="store_true",
                        dest="keep_saved_model",
                        help="Не удалять промежуточный TF SavedModel")
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        sys.exit(f"[ERROR] Файл не найден: {onnx_path}")

    output_dir = Path(args.output_dir) if args.output_dir else onnx_path.parent
    saved_model_dir = output_dir / f"{onnx_path.stem}_saved_model"

    # Конвертируем ONNX → SavedModel один раз
    onnx_to_saved_model(onnx_path, saved_model_dir)

    quant_modes = ["none", "fp16", "int8"] if args.all_quant else [args.quant]

    tflite_paths = []
    for quant in quant_modes:
        suffix = "" if quant == "none" else f"_{quant}"
        tflite_path = output_dir / f"{onnx_path.stem}{suffix}.tflite"
        saved_model_to_tflite(saved_model_dir, tflite_path, quantization=quant)
        tflite_paths.append((tflite_path, quant))

    if args.verify:
        for tflite_path, quant in tflite_paths:
            atol = 1e-4 if quant == "none" else (5e-2 if quant == "fp16" else 0.15)
            verify_tflite(tflite_path, onnx_path, atol=atol)

    if not args.keep_saved_model:
        shutil.rmtree(saved_model_dir, ignore_errors=True)
        print(f"\n[cleanup] Удалён временный SavedModel: {saved_model_dir}")

    print("\n── Итог ──────────────────────────────────────────────────────────")
    for tflite_path, quant in tflite_paths:
        size_mb = tflite_path.stat().st_size / 1024 / 1024
        print(f"  {quant:6s}  {size_mb:.1f} MB  →  {tflite_path}")


if __name__ == "__main__":
    main()