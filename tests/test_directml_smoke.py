import pytest


def test_directml_lif():
    dml_mod = pytest.importorskip("torch_directml")
    import torch
    import snntorch as snn

    dml = dml_mod.device()
    lif = snn.Leaky(beta=0.9).to(dml)
    x = torch.rand(4, 100, device=dml).requires_grad_(True)
    mem = lif.init_leaky()
    spk, mem = lif(x, mem)
    loss = spk.sum()
    loss.backward()
    assert x.grad is not None
    assert spk.device.type == "privateuseone"
