#!/usr/bin/env python3
"""
Конвертация mn04_as (PyTorch .pt) → ONNX

Вход модели: мел-спектрограмма float32 (batch, 1, 128, T)
Выход модели: logits float32 (batch, 527)
"""
import sys
import os
import argparse
import warnings
from pathlib import Path

# Отключаем спам из предупреждений новых версий PyTorch
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
import torch.nn as nn

# ── Добавляем EfficientAT в sys.path ──────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
EFFICIENTAT = REPO_ROOT / "third_party" / "EfficientAT"

if not EFFICIENTAT.exists():
    sys.exit(
        f"[ERROR] EfficientAT не найден: {EFFICIENTAT}\n"
        "Выполните: git clone https://github.com/fschmid56/EfficientAT.git third_party/EfficientAT"
    )

sys.path.insert(0, str(EFFICIENTAT))

ORIGINAL_CWD = Path.cwd()
os.chdir(str(EFFICIENTAT))

from models.mn.model import get_model as get_mobilenet   # type: ignore
from helpers.utils import NAME_TO_WIDTH                   # type: ignore


# ── Обёртка: возвращает только logits ─────────────────────────────────────
class LogitsOnlyWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(spec)
        # ИСПРАВЛЕНИЕ БАГА PyTorch: принудительно восстанавливаем 2D-форму [batch, num_classes].
        # Это не дает ONNX Runtime "схлопнуть" батч, если подается 1 аудиофайл.
        return logits.view(spec.shape[0], -1)


class BothOutputsWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, spec: torch.Tensor):
        logits, features = self.model(spec)
        logits = logits.view(spec.shape[0], -1)
        return logits, features


# ── Основная функция ──────────────────────────────────────────────────────
def export_onnx(
    model_name: str = "mn04_as",
    output_dir: Path = Path("exports"),
    opset: int = 18,
    dynamic_time: bool = True,
    include_features: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"{model_name}.onnx"

    print(f"[1/4] Загружаем {model_name}...")
    raw_model = get_mobilenet(
        width_mult=NAME_TO_WIDTH(model_name),
        pretrained_name=model_name,
        head_type="mlp",
    )
    raw_model.eval()

    if include_features:
        model = BothOutputsWrapper(raw_model)
        output_names = ["logits", "features"]
    else:
        model = LogitsOnlyWrapper(raw_model)
        output_names = ["logits"]

    # Обязательно переводим саму обертку в режим eval
    model.eval()

    # ── Dummy input ───────────────────────────────────────────────────────
    n_mels = 128
    T = 1001
    # БАТЧ = 2 при трассировке! Это дополнительная страховка, чтобы ONNX 
    # точно запомнил, что нулевая размерность — это батч, и её нельзя удалять.
    dummy = torch.zeros(2, 1, n_mels, T, dtype=torch.float32)

    if dynamic_time:
        dynamic_axes = {
            "spec":   {0: "batch", 3: "time_frames"},
            "logits": {0: "batch"},
        }
        if include_features:
            dynamic_axes["features"] = {0: "batch"}
    else:
        dynamic_axes = None

    print(f"[2/4] Экспортируем в ONNX (opset={opset})...")

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            opset_version=opset,
            input_names=["spec"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            export_params=True,
            verbose=False,
        )

    size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"[3/4] Сохранено: {onnx_path}  ({size_mb:.1f} MB)")

    try:
        import onnx
        model_onnx = onnx.load(str(onnx_path))
        onnx.checker.check_model(model_onnx)
        print("[4/4] onnx.checker: OK ✓")
    except Exception as e:
        print(f"[4/4] onnx.checker ОШИБКА: {e}")

    return onnx_path


# ── Численная проверка через onnxruntime ──────────────────────────────────
def verify_onnx(onnx_path: Path, model_name: str = "mn04_as", atol: float = 1e-4):
    print("\n── Численная верификация ONNX ──────────────────────────────────")
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [SKIP] onnxruntime не установлен")
        return

    raw_model = get_mobilenet(
        width_mult=NAME_TO_WIDTH(model_name),
        pretrained_name=model_name,
        head_type="mlp",
    )
    raw_model.eval()
    wrapped = LogitsOnlyWrapper(raw_model)
    wrapped.eval()

    # Фиксированный вход с батчем = 1 для проверки того, что динамический батч работает
    torch.manual_seed(0)
    dummy = torch.randn(1, 1, 128, 1001, dtype=torch.float32)

    with torch.no_grad():
        pt_logits = wrapped(dummy).numpy()

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    
    # Отключаем спам от onnxruntime
    sess_opts.log_severity_level = 3
    
    sess = ort.InferenceSession(str(onnx_path), sess_opts, providers=["CPUExecutionProvider"])
    ort_logits = sess.run(["logits"], {"spec": dummy.numpy()})[0]

    max_diff = float(np.abs(pt_logits - ort_logits).max())
    mean_diff = float(np.abs(pt_logits - ort_logits).mean())
    print(f"  Форма выхода ONNX: {ort_logits.shape} (ожидается (1, 527))")
    print(f"  Max  diff (logits): {max_diff:.2e}  {'✓ OK' if max_diff < atol else '✗ СЛИШКОМ БОЛЬШАЯ'}")

    top5_pt  = np.argsort(pt_logits[0])[::-1][:5].tolist()
    top5_ort = np.argsort(ort_logits[0])[::-1][:5].tolist()
    
    print(f"  Топ-5 PyTorch : {top5_pt}")
    print(f"  Топ-5 ONNX RT : {top5_ort}")
    print(f"  Топ-5 совпадают: {'✓ ДА' if top5_pt == top5_ort else '✗ НЕТ'}")


# ── CLI ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Export EfficientAT model to ONNX")
    parser.add_argument("--model", default="mn04_as", help="Имя модели")
    parser.add_argument("--output-dir", default="exports", dest="output_dir")
    parser.add_argument("--opset", default=18, type=int, help="ONNX opset version")
    parser.add_argument("--no-dynamic-time", action="store_true", dest="no_dynamic_time")
    parser.add_argument("--include-features", action="store_true", dest="include_features")
    parser.add_argument("--verify", action="store_true", help="Запустить верификацию")
    args = parser.parse_args()

    if Path(args.output_dir).is_absolute():
        final_output_dir = Path(args.output_dir)
    else:
        final_output_dir = ORIGINAL_CWD / args.output_dir

    onnx_path = export_onnx(
        model_name=args.model,
        output_dir=final_output_dir,
        opset=args.opset,
        dynamic_time=not args.no_dynamic_time,
        include_features=args.include_features,
    )

    if args.verify:
        verify_onnx(onnx_path, model_name=args.model)

    print(f"\nГотово: {onnx_path}")


if __name__ == "__main__":
    main()