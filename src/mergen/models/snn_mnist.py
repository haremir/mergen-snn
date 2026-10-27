import snntorch as snn
import torch
import torch.nn as nn


class SNNMNIST(nn.Module):
    """MNIST sınıflandırması için 2 katmanlı Spiking Neural Network (SNN) modeli.

    Görüntü her timestep aynı veriliyor; FC1 öğrenilmiş bir kodlayıcı gibi davranır.
    Poisson kodlama opsiyonel olarak encoding.py'den beslenebilir, ilk sürümde ham görüntü kullanılıyor.
    """

    def __init__(self, input_size: int = 784, hidden_size: int = 100, output_size: int = 10, beta: float = 0.9):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x: torch.Tensor, num_steps: int = 25):
        # x: (batch, 1, 28, 28) veya (batch, 784)
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk_out = []
        mem_out = []

        for _ in range(num_steps):
            cur1 = self.fc1(x_flat)  # görüntü her timestep aynı verilir
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk_out.append(spk2)
            mem_out.append(mem2)

        return torch.stack(spk_out, dim=0), torch.stack(mem_out, dim=0)
