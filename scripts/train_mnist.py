import os
from pathlib import Path
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

from mergen.data.mnist_loader import get_mnist_loaders
from mergen.models.snn_mnist import SNNMNIST
from mergen.training.trainer import eval_epoch, train_epoch
from mergen.utils.device import device_name, get_device
from mergen.utils.logger import get_logger


def load_config() -> dict:
    """default.yaml ve stage1_mnist.yaml yapılandırmalarını birleştirir."""
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
            for k, v in stage1_cfg.items():
                if isinstance(v, dict) and k in config and isinstance(config[k], dict):
                    config[k].update(v)
                else:
                    config[k] = v

    return config


def main():
    logger = get_logger("mergen.train_mnist")
    logger.info("==========================================")
    logger.info("Mergen-SNN — Aşama 1: MNIST SNN Eğitimi")
    logger.info("==========================================")

    # 1. Yapılandırma yükleme
    cfg = load_config()

    prefer_device = cfg.get("device", {}).get("prefer", "auto")
    device = get_device(prefer=prefer_device)
    dev_name = device_name(device)

    data_dir = cfg.get("paths", {}).get("data_raw", "data/raw")
    checkpoints_dir = Path(cfg.get("paths", {}).get("checkpoints", "checkpoints"))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(cfg.get("data", {}).get("batch_size", 128))
    num_workers = int(cfg.get("data", {}).get("num_workers", 0))

    input_size = int(cfg.get("model", {}).get("input_size", 784))
    hidden_size = int(cfg.get("model", {}).get("hidden_size", 100))
    output_size = int(cfg.get("model", {}).get("output_size", 10))
    beta = float(cfg.get("model", {}).get("beta", 0.9))
    num_steps = int(cfg.get("model", {}).get("num_steps", 25))

    epochs = int(cfg.get("training", {}).get("epochs", 1))
    lr = float(cfg.get("training", {}).get("learning_rate", 0.001))

    # 2. Model ve optimizatör oluşturma
    model = SNNMNIST(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        beta=beta,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Cihaz (Device): {dev_name}")
    logger.info(f"Model Parametre Sayısı: {num_params:,}")
    logger.info(f"Eğitim Parametreleri: batch_size={batch_size}, num_steps={num_steps}, epochs={epochs}, lr={lr}")

    # 3. Veri yükleyicileri
    train_loader, test_loader = get_mnist_loaders(
        batch_size=batch_size,
        num_workers=num_workers,
        data_dir=data_dir,
    )

    # 4. Optimizatör
    optimizer_name = cfg.get("training", {}).get("optimizer", "adam").lower()
    foreach_opt = cfg.get("training", {}).get("foreach", False)
    momentum_opt = float(cfg.get("training", {}).get("momentum", 0.9))

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, foreach=foreach_opt)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr * 10, momentum=momentum_opt, foreach=foreach_opt)
    else:
        raise ValueError(f"Bilinmeyen optimizer: {optimizer_name}")

    logger.info(f"Optimizer: {optimizer_name} (foreach={foreach_opt})")

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    epoch_durations = []

    # 5. Epoch döngüsü
    final_test_acc = 0.0
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            num_steps=num_steps,
        )

        test_loss, test_acc = eval_epoch(
            model=model,
            loader=test_loader,
            device=device,
            num_steps=num_steps,
        )

        elapsed = time.time() - start_time
        epoch_durations.append(elapsed)
        final_test_acc = test_acc

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        logger.info(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.2%} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.2%} | süre={elapsed:.1f}s"
        )

    # 6. Model kaydetme
    ckpt_path = checkpoints_dir / "mnist_snn.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg,
            "test_acc": final_test_acc,
        },
        ckpt_path,
    )
    logger.info(f"Model ağırlıkları kaydedildi: {ckpt_path.as_posix()}")

    # 7. Eğitim eğrisi grafiği kaydetme
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epoch_axis = range(1, epochs + 1)

    ax1.plot(epoch_axis, train_losses, label="Train Loss", color="royalblue", marker="o")
    ax1.plot(epoch_axis, test_losses, label="Test Loss", color="crimson", marker="s")
    ax1.set_title("Kayıp (Rate Loss)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(epoch_axis, [a * 100 for a in train_accs], label="Train Acc", color="royalblue", marker="o")
    ax2.plot(epoch_axis, [a * 100 for a in test_accs], label="Test Acc", color="crimson", marker="s")
    ax2.set_title("Doğruluk (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    curve_path = checkpoints_dir / "mnist_training_curve.png"
    plt.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Eğitim eğrisi kaydedildi: {curve_path.as_posix()}")

    avg_epoch_time = sum(epoch_durations) / len(epoch_durations)
    logger.info(f"Tamamlandı! Ortalama epoch süresi: {avg_epoch_time:.2f}s | Son Test Doğruluğu: {final_test_acc:.2%}")


if __name__ == "__main__":
    main()
