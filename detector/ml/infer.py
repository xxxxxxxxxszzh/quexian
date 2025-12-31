# detector/ml/inferyuan.py
import os
from typing import Dict, Optional, Tuple

import cv2 as cv
import numpy as np

import torch
import torch.nn as nn
from torchvision import models, transforms


# ====== 你需要改这里：checkpoint 路径 & 输入尺寸（必须与训练一致）======
DEFAULT_CKPT_PATH = os.path.join(os.path.dirname(__file__), "checkpoint_011.ckp")
IMG_SIZE = 224  # ⚠️ 如果你训练不是224，这里必须改成训练用的尺寸
# ===============================================================

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL = None  # 全局缓存：Django 启动后只加载一次

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def _build_model() -> torch.nn.Module:
    """
    构建与训练一致的模型结构：
    resnet34 + fc 输出2维（crack / inactive）
    """
    # 兼容 torchvision 不同版本
    try:
        model = models.resnet34(weights=None)
    except TypeError:
        model = models.resnet34(pretrained=False)

    # model.fc = nn.Linear(512, 2)
    model.fc = nn.Sequential(nn.Linear(512, 2))
    return model


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def load_resnet_once(ckpt_path: str = DEFAULT_CKPT_PATH) -> torch.nn.Module:
    """
    加载一次模型并缓存到全局变量，避免每次请求都加载权重。
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = _build_model()
    ckpt = torch.load(ckpt_path, map_location=_DEVICE)

    # 兼容两种保存格式：直接 state_dict 或 {'state_dict': ...}
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    state_dict = _strip_module_prefix(state_dict)

    model.load_state_dict(state_dict, strict=True)
    model.to(_DEVICE)
    model.eval()

    _MODEL = model
    return _MODEL


def _ensure_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """
    把各种可能的输入（灰度/rgba/float）统一成 uint8 的 RGB (H,W,3)
    """
    # OpenCV 读进来一般是 BGR；如果你传入的是别的来源，这里也能兜底
    if img is None:
        raise ValueError("Empty image")

    # 灰度 -> 3通道
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    # RGBA -> RGB
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]

    # 如果是 float 或别的类型，转到 uint8
    if img.dtype != np.uint8:
        # 若是 0~1 浮点，拉到 0~255；若是别的范围，做 clip
        img2 = img.astype(np.float32)
        if img2.max() <= 1.0:
            img2 = np.clip(img2, 0.0, 1.0) * 255.0
        else:
            img2 = np.clip(img2, 0.0, 255.0)
        img = img2.astype(np.uint8)

    # 保证是三通道
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    return img


@torch.no_grad()
def infer_one_image_path(
    img_path: str,
    ckpt_path: str = DEFAULT_CKPT_PATH,
    crack_th: float = 0.5,
    inactive_th: float = 0.5,
) -> Dict[str, float]:
    """
    输入：图片路径
    输出：crack_prob, inactive_prob, crack(bool), inactive(bool), is_bad(bool)
    """
    model = load_resnet_once(ckpt_path)

    img_bgr = cv.imread(img_path, cv.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    # OpenCV BGR -> RGB
    if img_bgr.ndim == 3 and img_bgr.shape[-1] == 3:
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    else:
        # 灰度或 rgba 走兜底
        img_rgb = img_bgr

    img_rgb = _ensure_uint8_rgb(img_rgb)

    x = _transform(img_rgb).unsqueeze(0).to(_DEVICE)  # [1,3,H,W]
    logits = model(x)                 # [1,2]
    probs = torch.sigmoid(logits)[0]  # [2]

    crack_prob = float(probs[0].item())
    inactive_prob = float(probs[1].item())

    crack = crack_prob >= crack_th
    inactive = inactive_prob >= inactive_th

    return {
        "crack_prob": crack_prob,
        "inactive_prob": inactive_prob,
        "crack": bool(crack),
        "inactive": bool(inactive),
        "is_bad": bool(crack or inactive),
    }


def run_infer(
    image_path: str,
    ckpt_path: str = DEFAULT_CKPT_PATH,
    crack_th: float = 0.5,
    inactive_th: float = 0.5,
) -> Dict[str, float]:
    """
    给 Django view 调用的统一入口。
    目前只做 ResNet 分类（先跑通闭环）。
    """
    return infer_one_image_path(
        img_path=image_path,
        ckpt_path=ckpt_path,
        crack_th=crack_th,
        inactive_th=inactive_th,
    )
