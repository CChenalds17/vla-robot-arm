import torch
from PIL import Image

def convert_angle_to_arduino(vla_angle: int) -> int:
    """Maps angle from [-180, +180] to [0, +180]"""
    return int((vla_angle + 180) / 2.0)

def convert_angle_to_vla(arduino_angle: int) -> int:
    """Maps angle from [0, +180] to [-180, +180]"""
    return int(2.0 * arduino_angle - 180)

def tensor_to_pil(t: torch.Tensor, mode="RGB") -> Image.Image:
    """Converts tensor to PIL image to display on computer"""
    t = t.cpu().detach().squeeze(0)
    arr = (t * 255).clamp(0,255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr, mode=mode)