import torch


def get_device(prefer: str = "auto"):
    """prefer: 'auto' | 'cpu' | 'directml'"""
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "directml":
        try:
            import torch_directml
        except ImportError:
            raise RuntimeError("torch-directml kurulu değil.")
        if torch_directml.device_count() == 0:
            raise RuntimeError("DirectML cihazı bulunamadı.")
        return torch_directml.device()
    # auto
    try:
        import torch_directml

        if torch_directml.device_count() > 0:
            return torch_directml.device()
    except ImportError:
        pass
    return torch.device("cpu")


def device_name(device) -> str:
    if getattr(device, "type", "") == "privateuseone":
        return f"DirectML ({device})"
    return str(device)
