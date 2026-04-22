import numpy as np
from pathlib import Path

from src.config import AppConfig
from src.model import AudioModel, get_device, ONNXAudioModel


def verify_conversion():
    cfg = AppConfig()
    cfg.model.name = "mn04_as"
    cfg.model.onnx_path = Path("scripts/exports/mn04_as.onnx")  # Путь к новому файлу

    device = get_device("cpu")

    print("[1] Загрузка PyTorch...")
    cfg.model.backend = "pt"
    pt_model = AudioModel.load(cfg.model, cfg.paths, device)

    print("[2] Загрузка ONNX...")
    cfg.model.backend = "onnx"
    onnx_model = ONNXAudioModel(cfg.model, cfg.paths, device)

    print("[3] Генерация шума (2 сек)...")
    np.random.seed(42)
    dummy = np.random.randn(64000).astype(np.float32)

    print("[4] Инференс...")
    pt_probs, pt_features = pt_model.infer_waveform(dummy)
    onnx_probs, onnx_features = onnx_model.infer_waveform(dummy)

    diff_probs = np.max(np.abs(pt_probs - onnx_probs))
    diff_feats = np.max(np.abs(pt_features - onnx_features))

    print("\n" + "=" * 40)
    print(f"Форма PyTorch фичей : {pt_features.shape}")
    print(f"Форма ONNX фичей    : {onnx_features.shape}")
    print(f"Макс. разница probs : {diff_probs:.8e}")
    print(f"Макс. разница фичей : {diff_feats:.8e}")
    print("=" * 40)

    if diff_probs < 1e-4 and diff_feats < 1e-4:
        print("✅ УСПЕХ: Векторы на 100% идентичны!")
    else:
        print("❌ ОШИБКА: Расхождение.")


if __name__ == "__main__":
    verify_conversion()