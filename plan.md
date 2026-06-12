# MSLM — Multi Small Language Models
## A Biologically-Inspired Neural Network of Language Models

**Author:** Adham  
**Status:** Research Proposal + Implementation Plan  
**Version:** 0.1.0

---

## Abstract

نقترح بنية جديدة تسمى **MSLM (Multi Small Language Models)**، تعتمد على شبكة من النماذج اللغوية الصغيرة المتخصصة، مترابطة كشبكة عصبية بيولوجية. بدلاً من نموذج واحد كبير يعمل بكامل طاقته لكل استعلام، تستيقظ فقط الخلايا ذات الصلة بالسياق — محاكاةً لآلية **Sparse Activation** في الدماغ البشري.

---

## 1. المشكلة

النماذج اللغوية الكبيرة (LLMs) تعاني من:

- **تكلفة حسابية ضخمة** — كل الـ parameters تعمل لكل استعلام
- **احتياج hardware مرتفع** — يستحيل تشغيلها محلياً
- **تخصص ضعيف** — نموذج واحد يحاول إتقان كل شيء

---

## 2. الفكرة الأساسية

```
بدل:
[LLM ضخم] ← كل شيء يعمل دائماً

نقترح:
[SLM-history]   💤
[SLM-geography] ✓  ← يستيقظ فقط عند الحاجة
[SLM-code]      💤
[SLM-biology]   💤
[SLM-math]      ✓  ← يستيقظ فقط عند الحاجة
```

كل SLM هو "خلية" متخصصة في مجال واحد.  
الشبكة كلها تعمل كدماغ — لا كآلة حاسبة.

---

## 3. الـ Architecture

### 3.1 الخلية (Cell)

```
كل خلية = SLM صغير جداً
- الحجم: 1M - 5M parameter
- التخصص: مجال معرفي واحد
- المخرج: embedding vector (مش tokens مباشرة)
```

الخلايا لا تتحدث بـ tokens — تتحدث بـ vectors.  
هذا يحل مشكلة التواصل بين النماذج المختلفة.

### 3.2 الشبكة (Network Graph)

```
[history] ──0.9──→ [geography]
    │
   0.6
    ↓
[culture] ──0.4──→ [philosophy]
    │
   0.1 (ضعيف جداً)
    ↓
[code]  💤 لا يستيقظ
```

- الأرقام = قوة الرابط (تتعلم تلقائياً)
- روابط قوية = مجالات متقاربة
- روابط ضعيفة = مجالات بعيدة

### 3.3 الـ Biological Router

أهم جزء في النظام — مستوحى من الدماغ البشري:

```
┌─────────────────────────────────────────┐
│         Biological Router               │
│                                         │
│  Level 1: THALAMUS                      │
│  ← استقبال سريع، تصنيف خشن             │
│  ← keywords + patterns                  │
│  ← output: domain_hint                  │
│                                         │
│  Level 2: PREFRONTAL CORTEX             │
│  ← embedding similarity دقيق           │
│  ← يحسب activation score لكل خلية     │
│  ← output: activation_map               │
│                                         │
│  Level 3: HIPPOCAMPUS                   │
│  ← يراجع سياق المحادثة السابقة         │
│  ← يعدّل الـ scores بناءً على التاريخ  │
│  ← output: final_activation             │
│                                         │
│  Spreading Activation                   │
│  ← الخلايا النشطة توقظ جاراتها         │
│  ← الإشارة تضعف مع البعد               │
│  ← threshold يحدد من يستيقظ فعلاً      │
└─────────────────────────────────────────┘
```

### 3.4 Spreading Activation

```
query: "سؤال عن الحرب العالمية"

history  → score: 0.95  ✓ يستيقظ
    ↓ × 0.9
geography → score: 0.85  ✓ يستيقظ
    ↓ × 0.4
economics → score: 0.34  ✓ يستيقظ خفيف
    ↓ × 0.2
biology  → score: 0.07  💤 أقل من threshold
    ↓
code     → score: 0.01  💤 نائم تماماً
```

### 3.5 Hebbian Learning في الروابط

```
"Neurons that fire together, wire together"

بعد كل استعلام ناجح:
إذا استيقظ [history] و[geography] معاً:
    connection[history][geography] += learning_rate

روابط لم تُستخدم:
    connection *= decay_factor
```

الشبكة **تتعلم بنية المعرفة** تلقائياً مع الوقت.

### 3.6 الـ Aggregator

```
active_cells = {history: 0.95, geography: 0.85, economics: 0.34}

final_output = Σ (cell.embedding × activation_score)
             = weighted combination of all active cells
```

---

## 4. مقارنة مع الأنظمة الموجودة

| | LLM كبير | MoE | MSLM |
|---|---|---|---|
| الحجم | مئات المليارات | مليارات | ملايين لكل خلية |
| Hardware | A100s | A100s | GTX + 32GB RAM |
| التخصص | عام | جزئي | كامل لكل خلية |
| Sparse Activation | لا | داخل النموذج | بين النماذج |
| التعلم المستمر | صعب | صعب | طبيعي عبر الروابط |
| قابل للتوسع | محدود | محدود | إضافة خلايا جديدة |

---

## 5. السؤال البحثي

> هل شبكة من النماذج الصغيرة المتخصصة، مع آلية activation بيولوجية،  
> تتفوق على نموذج واحد بنفس الحجم الإجمالي؟

**الفرضية:** نعم، لأن التخصص + الـ sparse activation يعوضان الفرق في الحجم.

---

## 6. خطة التنفيذ

### المرحلة 0 — الأساس (أسبوع 1-2)
```
الهدف: خلية واحدة تعمل

الملفات:
├── cell.py          ← بنية الـ SLM الأساسية
├── tokenizer.py     ← tokenizer مشترك لكل الخلايا
├── train.py         ← training loop بسيط
└── test_cell.py     ← اختبار الخلية

النموذج: Tiny Transformer
- 4 layers
- 128 hidden dim
- 2M parameter
- يتدرب على dataset صغير متخصص

commit: "feat: single cell implementation"
```

### المرحلة 1 — الـ Router (أسبوع 3-4)
```
الهدف: Biological Router يعمل

الملفات:
├── router/
│   ├── thalamus.py      ← Level 1: keyword + pattern
│   ├── prefrontal.py    ← Level 2: embedding similarity
│   ├── hippocampus.py   ← Level 3: context memory
│   └── router.py        ← يجمع الثلاثة
└── test_router.py

commit: "feat: biological router - 3 levels"
```

### المرحلة 2 — الشبكة (أسبوع 5-6)
```
الهدف: خلايا متعددة مترابطة

الملفات:
├── network/
│   ├── graph.py         ← بنية الـ connection graph
│   ├── spreading.py     ← spreading activation
│   ├── hebbian.py       ← تعلم الروابط تلقائياً
│   └── network.py       ← يجمع كل شيء
└── test_network.py

الخلايا الأولى:
- history
- geography  
- code
- math

commit: "feat: network with spreading activation"
```

### المرحلة 3 — الـ Aggregator (أسبوع 7-8)
```
الهدف: دمج مخرجات الخلايا في إجابة واحدة

الملفات:
├── aggregator.py    ← weighted combination
├── output.py        ← تحويل embedding لنص
└── pipeline.py      ← كل النظام معاً

commit: "feat: aggregator and full pipeline"
```

### المرحلة 4 — القياس (أسبوع 9-10)
```
الهدف: إجابة السؤال البحثي

الملفات:
├── benchmark/
│   ├── baseline.py      ← نموذج واحد بنفس الحجم
│   ├── mslm_eval.py     ← تقييم MSLM
│   └── compare.py       ← مقارنة النتائج
└── results/

commit: "eval: benchmark results"
```

### المرحلة 5 — التوثيق (أسبوع 11-12)
```
الهدف: ورقة بحثية + README كامل

الملفات:
├── paper/
│   └── MSLM_paper.md
├── README.md
└── ARCHITECTURE.md

commit: "docs: research paper and full documentation"
```

---

## 7. هيكل الـ Repository

```
mslm/
├── README.md
├── ARCHITECTURE.md
├── requirements.txt
│
├── core/
│   ├── cell.py
│   ├── tokenizer.py
│   └── train.py
│
├── router/
│   ├── thalamus.py
│   ├── prefrontal.py
│   ├── hippocampus.py
│   └── router.py
│
├── network/
│   ├── graph.py
│   ├── spreading.py
│   ├── hebbian.py
│   └── network.py
│
├── aggregator/
│   ├── aggregator.py
│   └── output.py
│
├── benchmark/
│   ├── baseline.py
│   └── evaluate.py
│
├── data/
│   ├── history/
│   ├── geography/
│   ├── code/
│   └── math/
│
└── paper/
    └── MSLM_paper.md
```

---

## 8. الـ AI Agent Prompt

هذا الـ prompt لأي AI agent يساعد في التنفيذ:

```
أنت مساعد في بناء مشروع بحثي اسمه MSLM.

المشروع:
- شبكة من النماذج اللغوية الصغيرة (1-5M parameter لكل خلية)
- كل خلية متخصصة في مجال معرفي واحد
- الخلايا مترابطة كشبكة عصبية بيولوجية
- آلية Sparse Activation مستوحاة من الدماغ البشري
- الـ Router يعمل على 3 مستويات: Thalamus, Prefrontal, Hippocampus

Hardware المتاح:
- Dell Precision 7520
- Intel Core i7-7820HQ
- NVIDIA Quadro M2200 (4GB VRAM, CUDA 5.2)
- 32GB DDR4 RAM

القيود:
- كل خلية يجب أن تتدرب على 4GB VRAM
- استخدم PyTorch
- اكتب كوداً نظيفاً مع comments واضحة
- كل مرحلة تنتهي بـ commit واضح على GitHub

المرحلة الحالية: [اذكر المرحلة]
المطلوب منك: [اذكر المهمة]
```

---

## 9. Commits Timeline

```
v0.0.1  init: project structure and research plan
v0.1.0  feat: single cell (tiny transformer 2M param)
v0.2.0  feat: biological router - thalamus level
v0.3.0  feat: biological router - prefrontal level  
v0.4.0  feat: biological router - hippocampus level
v0.5.0  feat: connection graph and spreading activation
v0.6.0  feat: hebbian learning for connection weights
v0.7.0  feat: aggregator and full pipeline
v0.8.0  eval: benchmark vs baseline
v0.9.0  fix: optimizations based on benchmark
v1.0.0  docs: research paper and full documentation
```

---

## 10. المقاييس التي سنقيس بها النجاح

```
1. الدقة:
   MSLM accuracy vs single model of same total size

2. الكفاءة:
   VRAM usage per query
   inference time per query

3. الذكاء البيولوجي:
   هل الروابط المتعلمة تعكس العلاقات الحقيقية بين المعرفة؟
   مثال: هل [history] و[geography] قووا رابطهم تلقائياً؟

4. قابلية التوسع:
   هل إضافة خلية جديدة تحسن النتائج؟
```

---

*"The goal is not to build a bigger brain — but a smarter one."*