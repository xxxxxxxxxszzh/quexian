import torch
import torch.nn as nn
from torchvision import models, transforms
from skimage.io import imread
import numpy as np

# ====== 1. 设备 ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== 2. 构建模型（必须和训练一致） ======
model = models.resnet34(weights=None)
model.fc = nn.Sequential(nn.Linear(512, 2))

ckpt_path = "checkpoints/checkpoint_011.ckp"
ckpt = torch.load(ckpt_path, map_location=device)

# 有些 checkpoint 会存成 {'state_dict': ...}，你这里就是这种
state_dict = ckpt["state_dict"]

# 如果当时用了 DataParallel，key 会带 "module."，这里顺手兼容一下
if any(k.startswith("module.") for k in state_dict.keys()):
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

model.load_state_dict(state_dict, strict=True)
model.to(device)
model.eval()

# ====== 3. 图像预处理（尽量与训练一致） ======
IMG_SIZE = 224  # ⚠️ 如果你 data.py 里不是 224，这里一定要改成训练用的尺寸

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # 训练用的是 resnet34 预训练，通常配这个
        std=[0.229, 0.224, 0.225]
    )
])

# ====== 4. 推理函数 ======
@torch.no_grad()
def infer_one_image(img_path, crack_th=0.5, inactive_th=0.5):
    img = imread(img_path)

    # (1) RGBA -> RGB
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]

    # (2) 灰度 -> 3通道
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    # (3) 保证 uint8 / 合理范围（skimage 有时给 float）
    if img.dtype != np.uint8:
        # 若是 0~1 浮点，拉到 0~255；若是别的范围，做 clip
        img = np.clip(img, 0, 1) if img.max() <= 1.0 else np.clip(img, 0, 255)
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

    x = transform(img).unsqueeze(0).to(device)

    logits = model(x)                     # shape: [1, 2]
    probs = torch.sigmoid(logits)[0]      # shape: [2]

    crack_prob = probs[0].item()
    inactive_prob = probs[1].item()

    return {
        "crack_prob": crack_prob,
        "inactive_prob": inactive_prob,
        "crack": crack_prob >= crack_th,
        "inactive": inactive_prob >= inactive_th
    }

# ====== 5. 示例 ======
if __name__ == "__main__":
    img_path = "images/cell0003.png"
    result = infer_one_image(img_path)
    print(result)