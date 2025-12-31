from pathlib import Path
from uuid import uuid4

from django.conf import settings
from django.shortcuts import render
from django.db import transaction

from .forms import UploadForm
from .models import DetectJob, DetectItem

# 你前面已有的推理函数（按你项目实际 import 路径改一下）
from detector.ml.infer import run_infer
from detector.ml.seg_infer import run_seg_save_label
from detector.ml.overlay import overlay_on_original


def _get(obj, key, default=None):
    """兼容 dict / object 两种返回格式"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _save_uploaded_file(uploaded_file, subdir="uploads"):
    """
    把上传文件保存到 media/<subdir>/ 下
    返回：(rel_path, abs_path)
    rel_path 形如：uploads/xxx.png（用于存库、拼 URL）
    abs_path 形如：/abs/.../media/uploads/xxx.png（用于推理）
    """
    media_root = Path(settings.MEDIA_ROOT)

    ext = Path(uploaded_file.name).suffix.lower() or ".png"
    stem = Path(uploaded_file.name).stem
    # 防止重名：原名 + 随机后缀
    save_name = f"{stem}_{uuid4().hex[:8]}{ext}"

    rel_path = f"{subdir}/{save_name}"
    abs_path = media_root / rel_path

    abs_path.parent.mkdir(parents=True, exist_ok=True)

    with abs_path.open("wb+") as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)

    return rel_path, str(abs_path)


def index(request):
    results = []

    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            job = DetectJob.objects.create(status="RUNNING")

            try:
                images = form.cleaned_data["images"]  # 你前面 multiple 的实现会返回 list
                for img in images:
                    # 1) 保存上传文件到 media/uploads/
                    upload_rel, upload_abs = _save_uploaded_file(img, subdir="uploads")

                    # 2) 先创建一条 DetectItem（把 upload_path 先落库）
                    item = DetectItem.objects.create(
                        job=job,
                        filename=img.name,
                        upload_path=upload_rel,
                    )

                    # 3) 分类推理（ResNet）
                    pred = run_infer(upload_abs)
                    item.crack_prob = float(_get(pred, "crack_prob", 0.0))
                    item.inactive_prob = float(_get(pred, "inactive_prob", 0.0))
                    item.is_bad = bool(_get(pred, "is_bad", False))

                    # 4) 坏片才做分割 + overlay（UNet）
                    seg_info = None
                    if item.is_bad:
                        base_stem = Path(upload_rel).stem  # 用同一张图的名字做结果文件名

                        # 结果相对路径（存库用）
                        label_rel = f"results/masks_label/{base_stem}.png"
                        overlay_rel = f"results/overlay_original/{base_stem}.png"

                        # 结果绝对路径（运行用）
                        label_abs = Path(settings.MEDIA_ROOT) / label_rel
                        overlay_abs = Path(settings.MEDIA_ROOT) / overlay_rel
                        label_abs.parent.mkdir(parents=True, exist_ok=True)
                        overlay_abs.parent.mkdir(parents=True, exist_ok=True)

                        # 你的封装：输出 256x256 label mask
                        run_seg_save_label(upload_abs, str(label_abs))
                        # 你的封装：把 label mask 贴回原图尺寸 overlay
                        overlay_on_original(upload_abs, str(label_abs), str(overlay_abs), alpha=0.4)

                        # 写入 DB
                        item.mask_label_path = label_rel
                        item.overlay_path = overlay_rel

                        seg_info = {
                            "label_url": settings.MEDIA_URL + label_rel,
                            "overlay_url": settings.MEDIA_URL + overlay_rel,
                        }

                    # 5) 保存本条 item 的推理结果
                    item.save()

                    # 6) 给前端展示用（保持你之前模板结构：name/img_url/pred/seg）
                    results.append({
                        "name": item.filename,
                        "img_url": settings.MEDIA_URL + upload_rel,
                        "pred": {
                            "crack_prob": item.crack_prob,
                            "inactive_prob": item.inactive_prob,
                            "is_bad": item.is_bad,
                        },
                        "seg": seg_info,  # None 或 dict(label_url/overlay_url)
                    })

                job.status = "DONE"
                job.save()

            except Exception as e:
                # 失败也保留 job 记录，方便排查
                job.status = "FAILED"
                job.error_msg = str(e)
                job.save()

                # 你也可以把错误展示在页面上
                return render(request, "index.html", {"form": form, "results": results, "err": str(e)})

            return render(request, "index.html", {"form": form, "results": results})

    else:
        form = UploadForm()

    return render(request, "index.html", {"form": form, "results": results})
