import pytest
import torch

from mergen.models.snn_mnist import SNNMNIST
from mergen.training.losses import rate_loss
from mergen.utils.device import get_device


def test_snn_mnist_forward_backward():
    # torch-directml kontrolü
    dml_mod = pytest.importorskip("torch_directml")
    if dml_mod.device_count() == 0:
        pytest.skip("DirectML cihazı bulunamadı.")

    device = get_device("auto")
    model = SNNMNIST(input_size=784, hidden_size=16, output_size=10, beta=0.9).to(device)

    dummy_input = torch.rand(2, 1, 28, 28, device=device)
    targets = torch.tensor([3, 7], device=device)

    spk_out, mem_out = model(dummy_input, num_steps=3)

    # Şekil doğrulaması
    assert spk_out.shape == (3, 2, 10)
    assert mem_out.shape == (3, 2, 10)

    # Kayıp ve geriye yayılım (backward) doğrulaması
    loss = rate_loss(spk_out, targets)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} parametresinin gradyanı None olamaz!"
