import torch


def get_device(prefer: str = "auto"):
    """prefer: 'auto' | 'cpu' | 'directml' | 'cuda' | 'tpu'"""
    if prefer == "tpu":
        return torch.device("tpu")
    if prefer == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if prefer == "directml":
        try:
            import torch_directml
        except ImportError:
            raise RuntimeError("torch-directml kurulu değil.")
        if torch_directml.device_count() == 0:
            raise RuntimeError("DirectML cihazı bulunamadı.")
        return torch_directml.device()
    if prefer == "cpu":
        return torch.device("cpu")

    # auto
    try:
        t = torch.ones((1,), device=torch.device("tpu"))
        return torch.device("tpu")
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        if torch_directml.device_count() > 0:
            return torch_directml.device()
    except ImportError:
        pass
    return torch.device("cpu")


def device_name(device) -> str:
    if getattr(device, "type", "") == "cuda":
        try:
            return f"CUDA ({torch.cuda.get_device_name(device)})"
        except Exception:
            return f"CUDA ({device})"
    if getattr(device, "type", "") == "tpu":
        return f"TPU ({device})"
    if getattr(device, "type", "") == "privateuseone":
        return f"DirectML/PrivateUse ({device})"
    return str(device)
