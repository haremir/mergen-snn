from pathlib import Path
from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(
    batch_size: int = 128,
    num_workers: int = 0,
    data_dir: str = "data/raw",
) -> Tuple[DataLoader, DataLoader]:
    """MNIST veri setini indirir ve DataLoader nesneleri döndürür.

    Piksel değerleri normalize edilmez, [0, 1] aralığında bırakılır.
    Bu aralık, Poisson veya rate tabanlı spike kodlama olasılıkları için doğrudan uygundur.

    Args:
        batch_size: Batch boyutu (varsayılan: 128).
        num_workers: Veri yükleme iş parçacığı sayısı (Windows uyumu için varsayılan: 0).
        data_dir: Verinin indirileceği/okunacağı dizin yolu.

    Returns:
        (train_loader, test_loader) tuple'ı.
    """
    raw_path = Path(data_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    # Piksel değerlerini [0.0, 1.0] aralığında tutan ToTensor dönüşümü
    transform = transforms.ToTensor()

    # MNIST eğitim ve test veri setleri indirilir
    train_dataset = datasets.MNIST(
        root=str(raw_path),
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=str(raw_path),
        train=False,
        download=True,
        transform=transform,
    )

    # DataLoader sarmalar ve mini-batch'ler halinde sunar
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, test_loader
