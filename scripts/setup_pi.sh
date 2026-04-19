#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
# setup_pi.sh
# Установка проекта на Raspberry Pi OS Lite (Bookworm, 64-bit, aarch64)
#
# Запуск:
#   chmod +x scripts/setup_pi.sh
#   ./scripts/setup_pi.sh
# ══════════════════════════════════════════════════════════════════════════
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "════════════════════════════════════════════════"
echo "  EfficientAT — установка на Raspberry Pi OS"
echo "  Bookworm 64-bit (aarch64)"
echo "════════════════════════════════════════════════"

# ── 1. Системные зависимости ──────────────────────────────────────────────
echo ""
echo "[1/5] Системные пакеты..."
sudo apt-get update -q
sudo apt-get install -y --no-install-recommends \
    python3-venv \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    libgomp1 \
    git \
    wget

echo "  ✓ Системные пакеты установлены"

# ── 2. Виртуальное окружение ─────────────────────────────────────────────
echo ""
echo "[2/5] Виртуальное окружение..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  ✓ venv создан"
else
    echo "  ✓ venv уже существует"
fi

# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip --quiet

# ── 3. PyTorch ────────────────────────────────────────────────────────────
echo ""
echo "[3/5] PyTorch (CPU)..."
echo "  Может занять 10–20 минут на Pi 4/5..."

install_torch() {
    pip install torch==2.10.0 torchaudio==2.10.0 torchvision==0.25.0 \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --quiet 2>/dev/null
}

install_torch_fallback() {
    echo "  ⚠ torch 2.10.0 недоступен, берём последний стабильный CPU..."
    pip install torch torchaudio torchvision \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --quiet
}

if ! install_torch; then
    install_torch_fallback
fi

TORCH_VER=$(python -c "import torch; print(torch.__version__)")
echo "  ✓ PyTorch $TORCH_VER"

# ── 4. Остальные зависимости ─────────────────────────────────────────────
echo ""
echo "[4/5] Python-зависимости..."
pip install \
    "librosa==0.11.0" \
    "soundfile==0.13.1" \
    "numpy==2.0.2" \
    "pandas==2.2.2" \
    "scikit-learn==1.6.1" \
    "scipy==1.16.3" \
    "matplotlib==3.10.0" \
    "umap-learn==0.5.12" \
    "pacmap==0.9.1" \
    "faiss-cpu==1.13.2" \
    "tqdm==4.67.3" \
    --quiet
echo "  ✓ Зависимости установлены"

# Sounddevice (опционально — нужен для daemon --mode mic)
echo ""
read -r -p "  Установить sounddevice (нужен для режима --mode mic)? [y/N] " ans
if [[ "$ans" =~ ^[Yy]$ ]]; then
    sudo apt-get install -y --no-install-recommends portaudio19-dev -q
    pip install sounddevice==0.4.7 --quiet
    echo "  ✓ sounddevice установлен"
else
    echo "  ↳ Пропущено. Режим --mode mic будет недоступен."
fi

# ── 5. EfficientAT ───────────────────────────────────────────────────────
echo ""
echo "[5/5] EfficientAT..."
if[ ! -d "third_party/EfficientAT" ]; then
    git clone --depth=1 https://github.com/fschmid56/EfficientAT.git \
        third_party/EfficientAT
    echo "  ✓ EfficientAT клонирован"
else
    echo "  ✓ EfficientAT уже существует"
fi

# ── Итог ─────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo "  ✓ Установка завершена!"
echo ""
echo "  Активировать окружение:"
echo "    source venv/bin/activate"
echo ""
echo "  Скачать датасет ESC-50:"
echo "    python scripts/download_data.py"
echo ""
echo "  Оценка (рекомендованная модель для Pi 4/5):"
echo "    python main.py evaluate --model mn04_as --threads 4"
echo ""
echo "  Инференс одного файла:"
echo "    python main.py infer звук.wav --model mn04_as"
echo ""
echo "  Daemon-режим (анализ в фоне):"
echo "    python main.py daemon --mode mock --source звук.wav --loop"
echo "════════════════════════════════════════════════"