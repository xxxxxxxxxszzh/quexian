from django.db import models

# Create your models here.
from django.db import models

class DetectJob(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, default="DONE")  # 先简单，未来可扩展
    error_msg = models.TextField(null=True, blank=True)

class DetectItem(models.Model):
    job = models.ForeignKey(DetectJob, on_delete=models.CASCADE, related_name="items")
    created_at = models.DateTimeField(auto_now_add=True)

    filename = models.CharField(max_length=255)
    upload_path = models.CharField(max_length=500)

    crack_prob = models.FloatField(null=True, blank=True)
    inactive_prob = models.FloatField(null=True, blank=True)
    is_bad = models.BooleanField(default=False)

    mask_label_path = models.CharField(max_length=500, null=True, blank=True)
    overlay_path = models.CharField(max_length=500, null=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["-created_at"]),
            models.Index(fields=["is_bad"]),
        ]
