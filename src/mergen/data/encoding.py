import torch


def poisson_encode(
    images: torch.Tensor,
    num_steps: int = 25,
    max_rate: float = 1.0,
) -> torch.Tensor:
    """Görüntü piksellerini Poisson spike trenlerine dönüştürür.

    Her piksel değeri bir olasılık olarak yorumlanır.
    Yüksek piksel değeri = yüksek ateşleme olasılığı anlamına gelir.
    Bernoulli örneklemesi vektörel olarak döngüsüz (torch.rand) gerçekleştirilir.

    Args:
        images: Girdi tensorü. Şekli (batch_size, 1, 28, 28) veya (batch_size, 784), [0, 1] aralığında.
        num_steps: Simülasyon zaman adımı sayısı (zaman penceresi uzunluğu).
        max_rate: Maksimum ateşleme oranı çarpanı (varsayılan: 1.0).

    Returns:
        Spike treni tensorü. Şekli (num_steps, batch_size, 784), float tipinde (0.0 veya 1.0).
    """
    batch_size = images.shape[0]
    # Görüntüyü 1B vektöre düzleştir: (batch_size, 784)
    flat_images = images.view(batch_size, -1)

    # Her piksel değeri p ise, ateşleme olasılığı: p * max_rate
    probabilities = torch.clamp(flat_images * max_rate, 0.0, 1.0)

    # [0, 1) aralığında rastgele tensor üret: (num_steps, batch_size, 784)
    rand_samples = torch.rand((num_steps, batch_size, flat_images.shape[1]), device=images.device)

    # Olasılıktan küçük olan örneklerde spike (1.0), aksi halde (0.0) üretilir
    # Surrogate gradient ve PyTorch otodif mekanizması için float tipinde döndürülür
    spike_train = (rand_samples < probabilities.unsqueeze(0)).float()

    return spike_train


def rate_encode(
    images: torch.Tensor,
    num_steps: int = 25,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Görüntü piksellerini eşik değerine göre deterministik spike trenine dönüştürür.

    Piksel değeri belirlenen eşikten büyükse tüm zaman adımlarında ateşler,
    küçük veya eşitse hiçbir zaman adımında ateşlemez.

    Args:
        images: Girdi tensorü. Şekli (batch_size, 1, 28, 28) veya (batch_size, 784).
        num_steps: Simülasyon zaman adımı sayısı.
        threshold: Ateşleme için eşik piksel değeri (varsayılan: 0.5).

    Returns:
        Spike treni tensorü. Şekli (num_steps, batch_size, 784), float tipinde (0.0 veya 1.0).
    """
    batch_size = images.shape[0]
    flat_images = images.view(batch_size, -1)

    # Eşik değerini aşan pikseller 1.0, aşmayanlar 0.0
    spikes_per_step = (flat_images > threshold).float()  # (batch_size, 784)

    # Zaman boyutu boyunca tekrarla: (num_steps, batch_size, 784)
    spike_train = spikes_per_step.unsqueeze(0).repeat(num_steps, 1, 1)

    return spike_train
