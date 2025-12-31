# detector/ml/seg_infer.py
import os
import cv2 as cv
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from .unet_model import construct_unet  # 你要把 unet_model.py 放在 detector/ml/ 里

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256

DEFAULT_WEIGHT_PATH = os.path.join(os.path.dirname(__file__), "model.pt")

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_SEG_MODEL = None


def load_ckpt_strip_module(model, path, device):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    # weights_only=True 可能直接给 state_dict，也可能给 dict 包着
    if hasattr(ckpt, "state_dict"):
        sd = ckpt.state_dict()
    elif isinstance(ckpt, dict) and ("state_dict" in ckpt or "model_state_dict" in ckpt):
        sd = ckpt.get("state_dict", ckpt.get("model_state_dict"))
    else:
        sd = ckpt

    if len(sd) > 0:
        first_key = next(iter(sd.keys()))
        if first_key.startswith("module."):
            sd = {k[len("module."):]: v for k, v in sd.items()}

    model.load_state_dict(sd, strict=True)
    return model



def load_unet_once(weight_path: str = DEFAULT_WEIGHT_PATH):
    global _SEG_MODEL
    if _SEG_MODEL is not None:
        return _SEG_MODEL

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"UNet weight not found: {weight_path}")

    unet = construct_unet(5)  # 类别数=5，对应 label 0..4
    unet = load_ckpt_strip_module(unet, weight_path, DEVICE)
    unet.to(DEVICE)
    unet.eval()
    _SEG_MODEL = unet
    return _SEG_MODEL


@torch.no_grad()
def predict_label_mask(image_path: str, weight_path: str = DEFAULT_WEIGHT_PATH) -> np.ndarray:
    """
    返回 label mask: uint8, shape (256,256), 值 0..4
    """
    model = load_unet_once(weight_path)

    bgr = cv.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
    rgb_256 = cv.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv.INTER_AREA)

    x = _transform(rgb_256).unsqueeze(0).to(DEVICE)  # 1x3x256x256
    logits = model(x)  # 期望 1x5x256x256

    if logits.ndim != 4 or logits.shape[1] != 5:
        raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)} (expected 1x5x256x256)")

    label = torch.softmax(logits, dim=1).argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    return label


def save_label_mask(label_hw: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(label_hw, mode="L").save(out_path)


def run_seg_save_label(image_path: str, out_label_path: str, weight_path: str = DEFAULT_WEIGHT_PATH) -> str:
    lab = predict_label_mask(image_path, weight_path=weight_path)
    save_label_mask(lab, out_label_path)
    return out_label_path
