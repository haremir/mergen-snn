"""Mergen-SNN — Karakter Seviyesi Dil Modeli Veri Kümesi (CharDataset).

PyTorch Dataset sınıfı ve charset yükleme/kodlama yardımcı fonksiyonları.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    """Karakter seviyesi dil modeli için sabit pencereli PyTorch veri kümesi."""

    def __init__(self, bin_path: Union[str, Path], block_size: int = 32, stride: Optional[int] = None):
        # np.load ile yükle ve int64 (PyTorch LongTensor) tipine dönüştür
        self.data = np.load(bin_path).astype(np.int64)
        self.block_size = block_size
        self.stride = stride if stride is not None else block_size

    def __len__(self) -> int:
        if len(self.data) < self.block_size + 1:
            return 0
        return (len(self.data) - 1) // self.stride

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        x = self.data[start : start + self.block_size]
        y = self.data[start + 1 : start + self.block_size + 1]
        return torch.from_numpy(x).long(), torch.from_numpy(y).long()


def load_charset(path: Union[str, Path]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """charset.json dosyasını yükler, (stoi, itos) döndürür. itos int anahtarlara sahiptir."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    stoi = data["stoi"]
    # JSON anahtarları string olarak saklandığı için itos tamsayı anahtarlı sözlüğe dönüştürülür
    itos = {int(k): v for k, v in data["itos"].items()}
    return stoi, itos


def encode(text: str, stoi: Dict[str, int]) -> List[int]:
    """Metni stoi eşlemesine göre tamsayı indeks listesine çevirir. Bilinmeyen karakterler <unk> (id=0) olur."""
    unk_id = stoi.get("<unk>", 0)
    return [stoi.get(ch, unk_id) for ch in text]


def decode(ids: Union[List[int], torch.Tensor, np.ndarray], itos: Dict[Union[int, str], str]) -> str:
    """Tamsayı indeks listesini veya tensorunu itos ile karaktere çevirir."""
    itos_int = {int(k) if isinstance(k, str) else k: v for k, v in itos.items()}
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    elif isinstance(ids, np.ndarray):
        ids = ids.tolist()
    return "".join(itos_int.get(int(i), "?") for i in ids)
