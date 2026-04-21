import numpy as np

from src.config import AppConfig
from src.model import AudioModel, get_device
from src.onnx_model import ONNXAudioModel

def verify_conversion():
    # 1. Настройка
    cfg = AppConfig()
    cfg.model.name = "mn04_as"
    device = get_device("cpu")
    
    print("[1] Загрузка оригинальной PyTorch модели...")
    pt_model = AudioModel.load(cfg.model, cfg.paths, device)
    
    print("[2] Загрузка ONNX модели...")
    onnx_model = ONNXAudioModel("mn04_as.onnx", pt_model)
    
    # 2. Генерируем тестовый звук (2 секунды случайного шума)
    # Частота 32000 Гц * 2 сек = 64000 сэмплов
    print("\n[3] Генерация тестового сигнала (белый шум)...")
    dummy_waveform = np.random.randn(64000).astype(np.float32)
    
    # 3. Инференс PyTorch
    print("[4] Инференс через PyTorch...")
    pt_probs, pt_features = pt_model.infer_waveform(dummy_waveform)
    
    # 4. Инференс ONNX
    print("[5] Инференс через ONNX...")
    onnx_probs, onnx_features = onnx_model.infer_waveform(dummy_waveform)
    
    # 5. Математическое сравнение массивов
    # Считаем максимальную абсолютную разницу (Max Absolute Error)
    diff_probs = np.max(np.abs(pt_probs - onnx_probs))
    diff_feats = np.max(np.abs(pt_features - onnx_features))
    
    print("\n" + "="*40)
    print("        РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
    print("="*40)
    print(f"Макс. разница вероятностей (probs) : {diff_probs:.8f}")
    print(f"Макс. разница фичей (features)     : {diff_feats:.8f}")
    print("-" * 40)
    
    if diff_probs < 1e-4 and diff_feats < 1e-4:
        print("✅ УСПЕХ: Веса и граф вычислений сконвертированы абсолютно корректно!")
        print("   ONNX модель выдает результаты, идентичные PyTorch.")
    else:
        print("❌ ВНИМАНИЕ: Обнаружено большое расхождение в результатах.")
        print("   Возможно, конвертация прошла с ошибкой.")

if __name__ == "__main__":
    verify_conversion()