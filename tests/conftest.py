"""Pytest konfigürasyonu ve DirectML autograd kuyruk başlatıcısı.

PyTorch 2.4 + torch-directml ortamında autograd engine'in privateuseone
cihaz kuyruğunu tanıması için erken başlatma fixture'ı.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def initialize_directml_autograd():
    try:
        import torch
        import torch_directml

        if torch_directml.device_count() > 0:
            dml = torch_directml.device()
            dummy = torch.tensor([0.0], device=dml, requires_grad=True)
            dummy.backward()
    except Exception:
        pass
