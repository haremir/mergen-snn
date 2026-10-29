"""Mergen-SNN — SNN Dil Modeli Eğitim, Değerlendirme ve Planlayıcı Fonksiyonları (Stage 2 & 3)."""

import math
from pathlib import Path
import time
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mergen.utils.logger import get_logger

logger = get_logger("mergen.lm_trainer")


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup ve ardından Cosine Decay uygulayan klasik LLM öğrenme oranı çizelgeleyicisi."""

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay * (1.0 - min_lr_ratio) + min_lr_ratio)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch_lm(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_steps: int = 32,
    max_steps: Optional[int] = None,
    log_every: int = 500,
    patience_steps: int = 5000,
    clip_grad_norm: Optional[float] = 1.0,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    chars_per_token: float = 3.29,
    checkpoint_every: Optional[int] = None,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_prefix: str = "bpe_lm",
    epoch_idx: int = 0,
    global_step_offset: int = 0,
    best_val_loss: float = float("inf"),
    start_step_in_epoch: int = 0,
    is_bpe: bool = True,
) -> Tuple[float, float, float, float, bool, int]:
    """SNN Dil Modeli için 1 epoch eğitim adımı.

    Dönen: (avg_loss, avg_bpb, elapsed_sec, avg_step_sec, early_stopped, steps_completed)
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

    ln2 = math.log(2)
    last_spk2_ratio = None

    for i, (x, y) in enumerate(loader):
        if i < start_step_in_epoch:
            continue

        x = x.to(device)
        y = y.to(device)  # x: (batch, num_steps), y: (batch, num_steps)

        if hasattr(model, "forward") and "return_spikes" in model.forward.__code__.co_varnames:
            logits, _, spk2 = model(x, num_steps, return_spikes=True)
            last_spk2_ratio = (spk2 > 0).float().mean().item()
        else:
            logits = model(x, num_steps)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping (surrogate gradient patlamalarını önler)
        if clip_grad_norm is not None and clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        tokens_in_batch = x.numel()
        batch_loss = loss.item()
        total_loss += batch_loss * tokens_in_batch
        total_tokens += tokens_in_batch
        step_count += 1
        current_step_in_epoch = start_step_in_epoch + step_count
        global_step = global_step_offset + step_count

        # Step-level Checkpointing (Adım 2: Her checkpoint_every adımda kaydet, son 3'ü tut)
        if checkpoint_every and checkpoint_dir and (global_step % checkpoint_every == 0):
            ckpt_path = Path(checkpoint_dir) / f"{checkpoint_prefix}_step{global_step}.pt"
            torch.save(
                {
                    "step": global_step,
                    "epoch": epoch_idx + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "best_val_loss": best_val_loss,
                    "rng_state": torch.get_rng_state(),
                    "train_loss": total_loss / max(total_tokens, 1),
                    "vocab_size": getattr(model, "vocab_size", None),
                    "embed_dim": getattr(model, "embed_dim", None),
                    "hidden_size": getattr(model, "hidden_size", None),
                    "tie_weights": getattr(model, "tie_weights", False),
                    "beta": getattr(model, "beta", 0.9),
                    "is_bpe": is_bpe,
                },
                ckpt_path,
            )
            # Sadece son 3 step checkpoint'ini tut, eskileri sil
            step_files = sorted(
                Path(checkpoint_dir).glob(f"{checkpoint_prefix}_step*.pt"),
                key=lambda f: int(f.stem.replace(f"{checkpoint_prefix}_step", ""))
                if f.stem.replace(f"{checkpoint_prefix}_step", "").isdigit()
                else 0,
            )
            if len(step_files) > 3:
                for old_f in step_files[:-3]:
                    try:
                        old_f.unlink()
                    except OSError:
                        pass

        if log_every and (current_step_in_epoch % log_every == 0 or current_step_in_epoch == total_steps):
            now = time.time()
            elapsed_interval = now - t_last_log
            steps_in_interval = log_every if current_step_in_epoch % log_every == 0 else (current_step_in_epoch % log_every)
            step_duration = elapsed_interval / max(steps_in_interval, 1)
            steps_per_sec = 1.0 / max(step_duration, 1e-4)

            curr_loss = total_loss / max(total_tokens, 1)
            curr_ppl = math.exp(min(curr_loss, 20.0))
            curr_bpb = curr_loss / (ln2 * chars_per_token)
            curr_lr = optimizer.param_groups[0]["lr"]

            spk_str = f" | spk2={last_spk2_ratio:.3f}" if last_spk2_ratio is not None else ""
            logger.info(
                f"Step {current_step_in_epoch}/{total_steps} (Total {global_step}) | "
                f"loss={curr_loss:.4f} | ppl={curr_ppl:.2f} | bpb={curr_bpb:.3f} | lr={curr_lr:.6f}{spk_str} | "
                f"{steps_per_sec:.1f}step/s"
            )
            t_last_log = now

            # Plato kontrolü (patience_steps boyunca iyileşme yoksa erken durdur)
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

        if max_steps and current_step_in_epoch >= max_steps:
            break

    elapsed = time.time() - t_start
    avg_step_sec = elapsed / max(step_count, 1)
    avg_loss = total_loss / max(total_tokens, 1)
    avg_bpb = avg_loss / (ln2 * chars_per_token)
    return avg_loss, avg_bpb, elapsed, avg_step_sec, early_stopped, step_count


def eval_epoch_lm(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_steps: int = 32,
    max_steps: Optional[int] = None,
    chars_per_token: float = 3.29,
) -> Tuple[float, float, float]:
    """SNN Dil Modeli için 1 epoch değerlendirme adımı.

    Dönen: (avg_loss, avg_bpb, elapsed_sec)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    step_count = 0
    t_start = time.time()
    ln2 = math.log(2)

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
    avg_bpb = avg_loss / (ln2 * chars_per_token)
    return avg_loss, avg_bpb, elapsed
