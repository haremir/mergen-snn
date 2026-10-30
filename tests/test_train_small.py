"""Mergen-SNN — Stage 4 train_small Altyapı ve Resume Testleri."""

from pathlib import Path
import tempfile
import torch

from scripts.train_small import atomic_save, find_latest_checkpoint, prune_old_step_checkpoints


def test_atomic_save_and_find_checkpoint():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Checkpoint dosyası oluştur
        ckpt_5 = tmp_path / "small_lm_step5.pt"
        ckpt_10 = tmp_path / "small_lm_step10.pt"

        atomic_save({"step": 5}, ckpt_5)
        atomic_save({"step": 10}, ckpt_10)

        assert ckpt_5.exists()
        assert ckpt_10.exists()

        latest = find_latest_checkpoint(tmp_path, prefix="small_lm")
        assert latest is not None
        assert latest.name == "small_lm_step10.pt"


def test_prune_old_step_checkpoints():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        for s in [500, 1000, 1500, 2000]:
            f = tmp_path / f"small_lm_step{s}.pt"
            atomic_save({"step": s}, f)

        # 4 checkpoint var, son 2'sini tut
        prune_old_step_checkpoints(tmp_path, prefix="small_lm", keep_last=2)

        remaining = sorted([f.name for f in tmp_path.glob("small_lm_step*.pt")])
        assert remaining == ["small_lm_step1500.pt", "small_lm_step2000.pt"]
