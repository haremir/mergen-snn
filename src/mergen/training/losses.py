import snntorch.functional as SF
import torch


def rate_loss(spk_out: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Spike trenleri için cross-entropy rate loss hesaplar.

    Zaman pencereleri boyunca üretilen toplam spike sayılarını sınıf olasılıkları olarak değerlendirir.

    Args:
        spk_out: Çıkış spike tensörü, şekli (num_steps, batch_size, num_classes).
        targets: Hedef sınıf etiketleri, şekli (batch_size,).

    Returns:
        Skaler kayıp (loss) tensörü.
    """
    return SF.ce_rate_loss()(spk_out, targets)
