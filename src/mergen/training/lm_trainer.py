"""Mergen-SNN — Stage 2 SNN Dil Modeli Eğitim ve Değerlendirme Fonksiyonları."""

import math
import time
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mergen.utils.logger import get_logger

logger = get_logger("mergen.lm_trainer")


def train_epoch_lm(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_steps: int = 32,
    max_steps: Optional[int] = None,
    log_every: int = 500,
    patience_steps: int = 5000,
) -> Tuple[float, float, float, bool]:
    """SNN Dil Modeli için 1 epoch eğitim adımı.

    Dönen: (avg_loss, elapsed_sec, avg_step_sec, early_stopped)
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0
    step_count = 0
    t_start = time.time()
    t_last_log = t_start

    total_steps = max_steps or len(loader)
    best_loss_window = float("inf")
    steps_without_improvement = 0
    early_stopped = False

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)  # x: (batch, num_steps), y: (batch, num_steps)

        logits = model(x, num_steps)  # (batch, num_steps, vocab_size)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tokens_in_batch = x.numel()
        batch_loss = loss.item()
        total_loss += batch_loss * tokens_in_batch
        total_tokens += tokens_in_batch
        step_count += 1

        if log_every and (step_count % log_every == 0 or step_count == total_steps):
            now = time.time()
            elapsed_interval = now - t_last_log
            steps_in_interval = log_every if step_count % log_every == 0 else (step_count % log_every)
            step_duration = elapsed_interval / max(steps_in_interval, 1)
            steps_per_sec = 1.0 / max(step_duration, 1e-4)

            curr_loss = total_loss / total_tokens
            curr_ppl = math.exp(min(curr_loss, 20.0))

            logger.info(
                f"Step {step_count}/{total_steps} | "
                f"loss={curr_loss:.4f} | ppl={curr_ppl:.2f} | "
                f"{steps_per_sec:.0f}step/s"
            )
            t_last_log = now

            # Plato kontrolü (5000 step boyunca iyileşme yoksa erken durdur)
            if curr_loss < best_loss_window - 1e-4:
                best_loss_window = curr_loss
                steps_without_improvement = 0
            else:
                steps_without_improvement += steps_in_interval
                if steps_without_improvement >= patience_steps:
                    logger.warning(
                        f"UYARI: {steps_without_improvement} adımdır loss düşmüyor (plato tespit edildi: {curr_loss:.4f}). "
                        f"Erken durdurma tetikleniyor."
                    )
                    early_stopped = True
                    break

        if max_steps and step_count >= max_steps:
            break

    elapsed = time.time() - t_start
    avg_step_sec = elapsed / max(step_count, 1)
    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, elapsed, avg_step_sec, early_stopped


def eval_epoch_lm(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_steps: int = 32,
    max_steps: Optional[int] = None,
) -> Tuple[float, float]:
    """SNN Dil Modeli için 1 epoch değerlendirme adımı.

    Dönen: (avg_loss, elapsed_sec)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    step_count = 0
    t_start = time.time()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x, num_steps)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )

            tokens_in_batch = x.numel()
            total_loss += loss.item() * tokens_in_batch
            total_tokens += tokens_in_batch
            step_count += 1

            if max_steps and step_count >= max_steps:
                break

    elapsed = time.time() - t_start
    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, elapsed
