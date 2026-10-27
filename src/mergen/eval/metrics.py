from typing import Tuple
import torch


def rate_accuracy(spk_out: torch.Tensor, targets: torch.Tensor) -> Tuple[int, int]:
    """Spike frekansına dayalı tahmin doğruluğunu hesaplar.

    Hesaplama ve metrik toplama CPU üzerinde gerçekleştirilir.

    Args:
        spk_out: Çıkış spike tensörü, şekli (num_steps, batch_size, num_classes).
        targets: Hedef sınıf etiketleri tensörü, şekli (batch_size,).

    Returns:
        (doğru tahmin sayısı, toplam örnek sayısı) tuple'ı (int).
    """
    # Tensörleri CPU'ya taşı ve grafikten ayır
    spk_sum = spk_out.detach().cpu().sum(dim=0)  # (batch_size, num_classes)
    targets_cpu = targets.detach().cpu()

    # En çok ateşleyen sınıfı tahmin olarak seç
    pred = spk_sum.argmax(dim=1)
    correct = (pred == targets_cpu).sum().item()
    total = targets_cpu.size(0)

    return int(correct), int(total)
