# CAMEL (Context-Aware Multi-cell Emergent Language Model)

الجمل يعيش في الصحراء بكفاءة مذهلة
موارد محدودة → أداء عالٍ

نفس فلسفة المشروع بالضبط
وهو عربي الأصل — مناسب لمشروع من مصر 🇪🇬

## التحديث الرسمي
* **النموذج:** CAMEL (Context-Aware Multi-cell Emergent Language Model)
* **الـ Router:** Biological Router
* **الخلية الأساسية (Base Cell):** `bigscience/bloom-560m` - مسؤولة عن **فهم اللغة فقط** (Language understanding only).
* **الخلايا المتخصصة (Specialist Cells):** مسؤولة عن **المعرفة المتخصصة** (Knowledge) عبر LoRA adapters.

## Training Hardware
تم تصميم وتدريب CAMEL للعمل بكفاءة قصوى على العتاد المحدود. تم تدريب النموذج على:
- **GPU:** NVIDIA Quadro M1200 (4GB VRAM)
- **CPU:** Intel Core i7-7820HQ
لمزيد من التفاصيل حول إعدادات البيئة وتحدي الـ 4GB VRAM، راجع ملف [HARDWARE.md](HARDWARE.md).
