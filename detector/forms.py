# detector/forms.py
from django import forms

class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True

class MultipleImageField(forms.ImageField):
    def clean(self, data, initial=None):
        # 让 ImageField 支持多文件
        if isinstance(data, (list, tuple)):
            return [super().clean(d, initial) for d in data]
        return super().clean(data, initial)

class UploadForm(forms.Form):
    images = MultipleImageField(
        widget=MultipleFileInput(attrs={"multiple": True}),
        required=True,
        label="选择图片（可多选）"
    )

