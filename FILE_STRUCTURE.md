# ADCD-Net Inference Scripts - File Structure & Relationships

```
ADCD-Net/
│
├── 📄 Core Inference Scripts
│   ├── inference.py                 ⭐ Main inference pipeline (SingleImageInference class)
│   ├── example_inference.py         🎓 Beginner-friendly example
│   ├── batch_inference.py           📦 Batch processing multiple images
│   ├── test_setup.py                🔧 Verify setup and environment
│   └── inference_utils.py           🛠️  Utility functions (metrics, visualization)
│
├── 📚 Documentation
│   ├── QUICK_REFERENCE.md           ⚡ Quick commands and API reference
│   ├── INFERENCE_GUIDE.md           📖 Comprehensive user guide
│   ├── INFERENCE_SUMMARY.md         📊 Technical details and examples
│   └── SETUP_COMPLETE.md            ✅ This summary document
│
├── 📦 Dependencies
│   └── requirements_inference.txt   📋 Python package requirements
│
├── 🎯 Original Training Code
│   ├── main.py                      🏋️  Training/evaluation script
│   ├── cfg.py                       ⚙️  Configuration
│   ├── ds.py                        💾 Dataset classes
│   ├── model/                       🧠 Model architecture
│   ├── loss/                        📉 Loss functions
│   └── seg_char/                    🔤 OCR character segmentation
│
└── 📖 Project Documentation
    ├── README.md                    📘 Main project README (updated)
    ├── LICENSE                      ⚖️  License
    └── fig/                         🖼️  Figures and images

```

## 🔄 Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE WORKFLOW                        │
└─────────────────────────────────────────────────────────────┘

1️⃣  SETUP & VERIFICATION
    ├── Install: pip install -r requirements_inference.txt
    └── Verify: python test_setup.py --model ... --docres ...

2️⃣  SINGLE IMAGE INFERENCE
    │
    ├── Option A: Command Line
    │   └── python inference.py --image IMG.jpg ...
    │
    ├── Option B: Example Script
    │   └── python example_inference.py (after editing config)
    │
    └── Option C: Python API
        └── from inference import SingleImageInference

3️⃣  BATCH PROCESSING
    └── python batch_inference.py --input-dir DIR/ ...

4️⃣  POST-PROCESSING & ANALYSIS
    └── from inference_utils import calculate_metrics, ...

```

## 🎯 Script Dependencies

```
inference.py (Core)
    ├── imports: model/model.py (ADCDNet)
    ├── imports: cfg.py (config)
    ├── imports: jpeg2dct (DCT extraction)
    ├── optional: seg_char/CRAFT_pytorch (OCR)
    └── required files:
        ├── ADCD-Net.pth (model checkpoint)
        ├── docres.pkl (DocRes checkpoint)
        └── qt_table.pk (quantization tables)

example_inference.py
    └── imports: inference.py (SingleImageInference)

batch_inference.py
    └── imports: inference.py (SingleImageInference)

test_setup.py
    └── imports: inference.py (SingleImageInference)

inference_utils.py
    └── standalone utilities (no internal imports)

```

## 📊 Data Flow

```
┌──────────────┐
│ Input Image  │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ JPEG Compression     │ ← jpeg_quality parameter
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ DCT Extraction       │ ← jpeg2dct library
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ OCR Mask Generation  │ ← CRAFT or precomputed
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Preprocessing        │ ← normalization, etc.
│ (img, dct, qt, ocr)  │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ ADCD-Net Model       │ ← inference.py
│ Forward Pass         │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Prediction Results   │
│ - Binary mask        │
│ - Probability map    │
│ - Alignment score    │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Visualization        │ ← matplotlib
│ (6-panel figure)     │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Output Files         │
│ - PNG visualization  │
│ - NPY binary mask    │
│ - JSON metrics       │
└──────────────────────┘
```

## 🗂️ Output Structure (Batch Processing)

```
output_dir/
├── visualizations/
│   ├── image001_prediction.png    ← 6-panel visualization
│   ├── image002_prediction.png
│   └── ...
│
├── masks/
│   ├── image001_mask.npy          ← Binary numpy arrays
│   ├── image002_mask.npy
│   └── ...
│
└── results_summary.json           ← Metrics for all images
    └── [
          {
            "image": "image001.jpg",
            "forgery_pixels": 12543,
            "forgery_ratio": 15.23,
            "alignment_score": 0.876,
            "status": "success"
          },
          ...
        ]
```

## 💡 Usage Patterns

### Pattern 1: Quick Test (Beginners)
```
1. Edit example_inference.py (set paths)
2. Run: python example_inference.py
3. Check output visualization
```

### Pattern 2: Command Line (Intermediate)
```
1. Prepare model files
2. Run: python inference.py --image test.jpg ...
3. Check output.png
```

### Pattern 3: Python API (Advanced)
```python
from inference import SingleImageInference

inferencer = SingleImageInference(...)
for img in images:
    results = inferencer.predict(img)
    # Custom processing
```

### Pattern 4: Batch Processing (Production)
```
1. Organize images in directory
2. Run: python batch_inference.py --input-dir ...
3. Review results_summary.json
4. Check visualizations/
```

## 📖 Documentation Flow

```
Start Here
    ↓
QUICK_REFERENCE.md     ← Commands, quick API
    ↓
Need more detail?
    ↓
INFERENCE_GUIDE.md     ← Full user guide
    ↓
Want examples?
    ↓
example_inference.py   ← Working code
    ↓
Advanced usage?
    ↓
INFERENCE_SUMMARY.md   ← Technical details
    ↓
Utility functions?
    ↓
inference_utils.py     ← Helper functions
```

## 🔍 Quick Command Reference

| Task | Command |
|------|---------|
| **Test Setup** | `python test_setup.py --model M --docres D --qt-table Q` |
| **Single Image** | `python inference.py --image I --model M --docres D --qt-table Q` |
| **With CRAFT** | `python inference.py --image I ... --craft C` |
| **Batch** | `python batch_inference.py --input-dir IN --output-dir OUT ...` |
| **CPU Mode** | `python inference.py ... --device cpu` |

## 🎨 Visualization Components

The 6-panel output includes:

```
┌─────────────┬─────────────┬─────────────┐
│   Original  │ Compressed  │  OCR Mask   │
│    Image    │    Image    │  (Text)     │
├─────────────┼─────────────┼─────────────┤
│  Predicted  │  Forgery    │  Overlay    │
│   Binary    │ Probability │  (Colored)  │
│    Mask     │   Heatmap   │             │
└─────────────┴─────────────┴─────────────┘
```

## 🚀 Getting Started Checklist

- [ ] Read QUICK_REFERENCE.md
- [ ] Install dependencies: `pip install -r requirements_inference.txt`
- [ ] Download model checkpoints (ADCD-Net.pth, docres.pkl, qt_table.pk)
- [ ] Run test: `python test_setup.py --model ... --docres ... --qt-table ...`
- [ ] Try example: Edit and run `example_inference.py`
- [ ] Test single image: `python inference.py --image ...`
- [ ] Try batch: `python batch_inference.py --input-dir ...`
- [ ] Explore utilities: Check `inference_utils.py`
- [ ] Read full guide: INFERENCE_GUIDE.md (if needed)

## 📦 What You Get

✅ **4 Executable Scripts**
   - inference.py (main)
   - example_inference.py (beginner)
   - batch_inference.py (batch)
   - test_setup.py (verification)

✅ **1 Utility Module**
   - inference_utils.py (helpers)

✅ **4 Documentation Files**
   - QUICK_REFERENCE.md (quick start)
   - INFERENCE_GUIDE.md (comprehensive)
   - INFERENCE_SUMMARY.md (technical)
   - SETUP_COMPLETE.md (this file)

✅ **1 Requirements File**
   - requirements_inference.txt

✅ **Updated Main README**
   - With inference section

---

**Total: 10 new/updated files for complete inference capability! 🎉**

**Next Step**: Start with [QUICK_REFERENCE.md](QUICK_REFERENCE.md)!
