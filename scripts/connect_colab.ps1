# Mergen-SNN — Windows Terminalinden (PowerShell) Colab'e Bağlantı Scripti

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Mergen-SNN — Google Colab CLI Bağlantı ve Kurulum Aracı" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. uv veya pip ile colabapi / google-colab-cli kurulumu
Write-Host "`n>>> Colab CLI araçları kontrol ediliyor ve kuruluyor..." -ForegroundColor Yellow

$uvInstalled = Get-Command uv -ErrorAction SilentlyContinue

if ($uvInstalled) {
    Write-Host "uv bulundu, colabapi aracı kuruluyor..." -ForegroundColor Green
    uv tool install colabapi --force 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "uv tool fallback: pip ile colabapi kuruluyor..." -ForegroundColor Yellow
        uv pip install colabapi google-colab-cli
    }
} else {
    Write-Host "uv bulunamadı, Python pip kullanılıyor..." -ForegroundColor Yellow
    python -m pip install --upgrade colabapi google-colab-cli
}

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "Kurulum Tamamlandı! Terminalden Colab'e Bağlanma Talimatları:" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

Write-Host "`n1. İlk defa kullanıyorsan Google hesabınla kimlik doğrula:" -ForegroundColor White
Write-Host "   colabapi login" -ForegroundColor Cyan

Write-Host "`n2. Colab üzerinde T4 GPU ile eğitimi doğrudan terminalden başlatmak için:" -ForegroundColor White
Write-Host "   colabapi run --runtime t4 `"python scripts/colab_runner.py`"" -ForegroundColor Cyan

Write-Host "`n3. Colab üzerinde interaktif terminal (SSH) açmak için:" -ForegroundColor White
Write-Host "   colabapi ssh --runtime t4" -ForegroundColor Cyan

Write-Host "`n4. Tarayıcı (Google Colab Notebook) üzerinden çalıştırmak için:" -ForegroundColor White
Write-Host "   Google Drive'daki 'notebooks/mergen_colab_training.ipynb' dosyasını açıp" -ForegroundColor White
Write-Host "   Runtime -> T4 GPU seçtikten sonra Hücre 1'i (colab_runner) çalıştırman yeterlidir." -ForegroundColor Yellow
Write-Host "`n============================================================`n" -ForegroundColor Cyan
