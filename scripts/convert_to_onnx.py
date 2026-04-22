#!/usr/bin/env python3
import sys
import os
import argparse
import warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

REPO_ROOT = Path(__file__).resolve().parent.parent
EFFICIENTAT = REPO_ROOT / "third_party" / "EfficientAT"

if not EFFICIENTAT.exists():
    sys.exit("[ERROR] EfficientAT не найден.")

sys.path.insert(0, str(EFFICIENTAT))
ORIGINAL_CWD = Path.cwd()
os.chdir(str(EFFICIENTAT))

from models.mn.model import get_model as get_mobilenet
from helpers.utils import NAME_TO_WIDTH


class BothOutputsWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, spec: torch.Tensor):
        # Берем родные выходы EfficientAT
        logits, features = self.model(spec)

        # ЖЕСТКАЯ ФИКСАЦИЯ ФОРМЫ: запрещаем ONNX схлопывать размерности
        logits = logits.view(spec.shape[0], -1)
        features = features.view(spec.shape[0], -1)

        return logits, features


def export_onnx(model_name: str = "mn04_as", output_dir: Path = Path("exports"), opset: int = 18):
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"{model_name}.onnx"

    raw_model = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name, head_type="mlp")
    raw_model.eval()
    model = BothOutputsWrapper(raw_model)
    model.eval()

    dummy = torch.zeros(2, 1, 128, 1001, dtype=torch.float32)
    dynamic_axes = {"spec": {0: "batch", 3: "time_frames"}, "logits": {0: "batch"}, "features": {0: "batch"}}

    print(f"Экспортируем в ONNX с жесткой фиксацией фичей (384)...")
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, str(onnx_path), opset_version=opset,
            input_names=["spec"], output_names=["logits", "features"],
            dynamic_axes=dynamic_axes, do_constant_folding=True, export_params=True
        )

    print(f"Сохранено: {onnx_path}")
    return onnx_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mn04_as")
    parser.add_argument("--output-dir", default="exports", dest="output_dir")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else ORIGINAL_CWD / args.output_dir
    export_onnx(model_name=args.model, output_dir=out_dir)


if __name__ == "__main__":
    main()