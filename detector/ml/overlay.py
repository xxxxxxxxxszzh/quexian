# detector/ml/overlay.py
import os
import cv2 as cv
import numpy as np

# 类别颜色（RGB）——对应 label mask: 1..4；0 背景不涂
COLOR_MAP = {
    1: (222, 156, 83),  # busbar
    2: (165, 59, 63),   # crack
    3: (77, 137, 99),   # cross
    4: (68, 114, 148),  # dark
}

def overlay_on_original(img_path, mask_label_path, out_path, alpha=0.4):
    img_bgr = cv.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    H, W = img_bgr.shape[:2]

    lab = cv.imread(mask_label_path, cv.IMREAD_GRAYSCALE)
    if lab is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_label_path}")

    lab0 = cv.resize(lab, (W, H), interpolation=cv.INTER_NEAREST)

    color_bgr = np.zeros_like(img_bgr, dtype=np.uint8)
    for cls_id, rgb in COLOR_MAP.items():
        bgr = (rgb[2], rgb[1], rgb[0])
        color_bgr[lab0 == cls_id] = bgr

    overlay = cv.addWeighted(img_bgr, 1 - alpha, color_bgr, alpha, 0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv.imwrite(out_path, overlay)
