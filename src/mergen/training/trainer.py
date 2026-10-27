from typing import Tuple
import torch
from torch.utils.data import DataLoader

from mergen.eval.metrics import rate_accuracy
from mergen.training.losses import rate_loss


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_steps: int,
) -> Tuple[float, float]:
    """Bir eğitim epoch'u çalıştırır.

    Args:
        model: SNN modeli.
        loader: Eğitim DataLoader'ı.
        optimizer: Optimizatör nesnesi.
        device: Hesaplama yapılacak cihaz (CPU veya DirectML).
        num_steps: Zaman adımı sayısı.

    Returns:
        (ortalama kayıp, ortalama doğruluk) tuple'ı.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, targets in loader:
        data = data.to(device)
        targets = targets.to(device)

        spk_out, _ = model(data, num_steps)
        loss = rate_loss(spk_out, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        c, t = rate_accuracy(spk_out.detach(), targets)
        correct += c
        total += t

    return total_loss / total, correct / total


def eval_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_steps: int,
) -> Tuple[float, float]:
    """Bir değerlendirme (test) epoch'u çalıştırır.

    Args:
        model: SNN modeli.
        loader: Test DataLoader'ı.
        device: Hesaplama yapılacak cihaz.
        num_steps: Zaman adımı sayısı.

    Returns:
        (ortalama kayıp, ortalama doğruluk) tuple'ı.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)

            spk_out, _ = model(data, num_steps)
            loss = rate_loss(spk_out, targets)

            total_loss += loss.item() * data.size(0)
            c, t = rate_accuracy(spk_out, targets)
            correct += c
            total += t

    return total_loss / total, correct / total
