#!/usr/bin/env bash
# Mergen-SNN — Terminalden (Bash) Colab'e Bağlantı Scripti

set -e

echo "============================================================"
echo "Mergen-SNN — Google Colab CLI Bağlantı ve Kurulum Aracı"
echo "============================================================"

echo ""
echo ">>> Colab CLI araçları kontrol ediliyor ve kuruluyor..."

if command -v uv &> /dev/null; then
    echo "uv bulundu, colabapi kuruluyor..."
    uv tool install colabapi --force || uv pip install colabapi google-colab-cli
else
    echo "uv bulunamadı, pip ile kuruluyor..."
    python3 -m pip install --upgrade colabapi google-colab-cli
fi

echo ""
echo "============================================================"
echo "Kurulum Tamamlandı! Terminalden Colab'e Bağlanma Talimatları:"
echo "============================================================"
echo ""
echo "1. İlk defa kullanıyorsan Google hesabınla kimlik doğrula:"
echo "   colabapi login"
echo ""
echo "2. Colab üzerinde T4 GPU ile eğitimi doğrudan terminalden başlatmak için:"
echo "   colabapi run --runtime t4 \"python scripts/colab_runner.py\""
echo ""
echo "3. Colab üzerinde interaktif terminal (SSH) açmak için:"
echo "   colabapi ssh --runtime t4"
echo ""
echo "4. Tarayıcı üzerinden çalıştırmak için:"
echo "   Drive'daki 'notebooks/mergen_colab_training.ipynb' dosyasını açıp"
echo "   Hücre 1'i (colab_runner) çalıştırman yeterlidir."
echo "============================================================"
