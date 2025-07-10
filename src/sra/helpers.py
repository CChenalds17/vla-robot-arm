import torch
from PIL import Image

def convert_angle_to_arduino(vla_angle: int) -> int:
    """Maps angle from [-90, +90] to [0, +180]"""
    return round(max(0, min(vla_angle + 90, 180)))

def convert_angle_to_vla(arduino_angle: int) -> int:
    """Maps angle from [0, +180] to [-90, +90]"""
    return round(max(-90, min(arduino_angle - 90, 90)))

def tensor_to_pil(t: torch.Tensor, mode="RGB") -> Image.Image:
    """Converts tensor to PIL image to display on computer"""
    t = t.cpu().detach().squeeze(0)
    arr = (t * 255).clamp(0,255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr, mode=mode)