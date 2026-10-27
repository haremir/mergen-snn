import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Arayüzsüz (headless) ortamlar için Agg backend
import matplotlib.pyplot as plt
import torch
import yaml

from mergen.data.encoding import poisson_encode
from mergen.data.mnist_loader import get_mnist_loaders
from mergen.utils.logger import get_logger


def load_config() -> dict:
    """default.yaml ve stage1_mnist.yaml dosyalarını yükleyip birleştirir."""
    configs_dir = Path("configs")
    default_cfg_path = configs_dir / "default.yaml"
    stage1_cfg_path = configs_dir / "stage1_mnist.yaml"

    config = {}
    if default_cfg_path.exists():
        with open(default_cfg_path, "r", encoding="utf-8") as f:
            config.update(yaml.safe_load(f) or {})

    if stage1_cfg_path.exists():
        with open(stage1_cfg_path, "r", encoding="utf-8") as f:
            stage1_cfg = yaml.safe_load(f) or {}
            # Alt sözlükleri derinlemesine birleştir
            for k, v in stage1_cfg.items():
                if isinstance(v, dict) and k in config and isinstance(config[k], dict):
                    config[k].update(v)
                else:
                    config[k] = v

    return config


def main():
    logger = get_logger("mergen.demo")
    logger.info("==========================================")
    logger.info("Mergen-SNN — Spike Encoding Demo")
    logger.info("==========================================")

    # 1. Config değerlerini oku
    cfg = load_config()
    data_dir = cfg.get("paths", {}).get("data_raw", "data/raw")
    checkpoints_dir = cfg.get("paths", {}).get("checkpoints", "checkpoints")
    num_steps = cfg.get("model", {}).get("num_steps", 25)
    max_rate = float(cfg.get("encoding", {}).get("max_rate", 1.0))

    logger.info(f"Yapılandırma yüklendi: num_steps={num_steps}, max_rate={max_rate}, data_dir={data_dir}")

    # 2. MNIST DataLoader'dan 8 örnekli bir batch al
    batch_size = 8
    train_loader, _ = get_mnist_loaders(batch_size=batch_size, num_workers=0, data_dir=data_dir)
    images, labels = next(iter(train_loader))

    # 3. Poisson spike kodlama uygula
    spikes = poisson_encode(images, num_steps=num_steps, max_rate=max_rate)

    # 4. Konsola metrikleri yaz
    total_spikes = spikes.sum().item()
    avg_spikes_per_pixel = total_spikes / (batch_size * 784 * num_steps)

    print(f"Batch shape: {images.shape}")
    print(f"Spike train shape: {spikes.shape}")
    print(f"Toplam spike sayısı: {total_spikes}")
    print(f"Ortalama spike/piksel: {avg_spikes_per_pixel:.4f}")

    # 5. İlk görüntüyü görselleştir (images[0], first_image_spikes)
    first_image = images[0, 0]  # (28, 28)
    first_spikes = spikes[:, 0, :]  # (num_steps, 784)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1 (Sol): Orijinal Görüntü
    axes[0].imshow(first_image.cpu().numpy(), cmap="gray")
    axes[0].set_title("Orijinal (piksel değeri = ateşleme olasılığı)", fontsize=11)
    axes[0].axis("off")

    # Panel 2 (Orta): Raster Plot Spike Trenleri
    time_indices, pixel_indices = torch.where(first_spikes == 1.0)
    axes[1].scatter(
        time_indices.cpu().numpy(),
        pixel_indices.cpu().numpy(),
        s=1.5,
        c="black",
        marker=".",
    )
    axes[1].set_xlim(-0.5, num_steps - 0.5)
    axes[1].set_ylim(0, 784)
    axes[1].set_xlabel("Zaman Adımı (Timestep)", fontsize=10)
    axes[1].set_ylabel("Piksel İndeksi (0-783)", fontsize=10)
    axes[1].set_title(f"Poisson Spike Trenleri ({num_steps} timestep)", fontsize=11)

    # Panel 3 (Sağ): Her pikselin toplam spike sayısı
    total_spikes_per_pixel = first_spikes.sum(dim=0).view(28, 28).cpu().numpy()
    im3 = axes[2].imshow(total_spikes_per_pixel, cmap="hot")
    axes[2].set_title("Toplam Spike Sayısı", fontsize=11)
    axes[2].axis("off")
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()

    # 6. Görseli kaydet
    out_dir = Path(checkpoints_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "demo_encoding.png"

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Görsel kaydedildi: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
