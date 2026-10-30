# Mergen-SNN - Windows Terminalinden Colab Baglanti Scripti

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Mergen-SNN - Google Colab CLI Baglanti ve Kurulum Araci" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. uv veya pip ile colabapi / google-colab-cli kurulumu
Write-Host "`n>>> Colab CLI araclari kontrol ediliyor ve kuruluyor..." -ForegroundColor Yellow

$uvInstalled = Get-Command uv -ErrorAction SilentlyContinue

if ($uvInstalled) {
    Write-Host "uv bulundu, colabapi araci kuruluyor..." -ForegroundColor Green
    uv tool install colabapi --force 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "uv tool fallback: pip ile colabapi kuruluyor..." -ForegroundColor Yellow
        uv pip install colabapi google-colab-cli
    }
} else {
    Write-Host "uv bulunamadi, Python pip kullaniliyor..." -ForegroundColor Yellow
    python -m pip install --upgrade colabapi google-colab-cli
}

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "Kurulum Tamamlandi! Terminalden Colab Baglanti Talimatlari:" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

Write-Host "`n1. Ilk defa kullaniyorsan Google hesabinla kimlik dogrula:" -ForegroundColor White
Write-Host "   colabapi login" -ForegroundColor Cyan

Write-Host "`n2. Colab uzerinde T4 GPU ile egitimi baslatmak icin:" -ForegroundColor White
Write-Host "   colabapi run --runtime t4 ""python scripts/colab_runner.py""" -ForegroundColor Cyan

Write-Host "`n3. Colab uzerinde interaktif terminal (SSH) acmak icin:" -ForegroundColor White
Write-Host "   colabapi ssh --runtime t4" -ForegroundColor Cyan

Write-Host "`n4. Tarayici (Google Colab Notebook) uzerinden calistirmak icin:" -ForegroundColor White
Write-Host "   Google Drive icindeki 'notebooks/mergen_colab_training.ipynb' dosyasini acip" -ForegroundColor White
Write-Host "   Runtime -> Change runtime type -> T4 GPU sectikten sonra Hucre 1 calistirilabilir." -ForegroundColor Yellow
Write-Host "`n============================================================`n" -ForegroundColor Cyan
