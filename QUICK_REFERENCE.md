# ADCD-Net Inference - Quick Reference Card

## 🎯 One-Line Commands

### Single Image Inference
```bash
python inference.py --image IMAGE.jpg --model MODEL.pth --docres DOCRES.pkl --qt-table QT.pk --output result.png
```

### Batch Processing
```bash
python batch_inference.py --input-dir DIR/ --output-dir OUT/ --model MODEL.pth --docres DOCRES.pkl --qt-table QT.pk
```

### Test Setup
```bash
python test_setup.py --model MODEL.pth --docres DOCRES.pkl --qt-table QT.pk
```

## 📦 Installation

```bash
pip install -r requirements_inference.txt
```

## 🐍 Python API

```python
from inference import SingleImageInference

# Initialize
inferencer = SingleImageInference(
    model_ckpt_path='ADCD-Net.pth',
    docres_ckpt_path='docres.pkl',
    qt_table_path='qt_table.pk',
    device='cuda'
)

# Predict
results = inferencer.predict('image.jpg', jpeg_quality=100)

# Visualize
inferencer.visualize_results(results, save_path='output.png')

# Access results
pred_mask = results['pred_mask']           # Binary mask (H, W)
pred_prob = results['pred_prob']           # Probability (2, H, W)
align_score = results['align_score'][1]    # Alignment score [0-1]
```

## 📊 Results Structure

```python
results = {
    'pred_mask': np.ndarray,      # Binary: 0=authentic, 1=forged
    'pred_prob': np.ndarray,      # Class probabilities [2, H, W]
    'align_score': np.ndarray,    # DCT alignment [non-align, align]
    'original_img': np.ndarray,   # Original RGB image
    'compressed_img': np.ndarray, # After JPEG compression
    'ocr_mask': np.ndarray        # Character mask
}
```

## 🔧 Common Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--image` | Input image path | Required |
| `--model` | ADCD-Net checkpoint | Required |
| `--docres` | DocRes checkpoint | Required |
| `--qt-table` | Quantization table | Required |
| `--craft` | CRAFT OCR model | None |
| `--ocr-bbox` | Precomputed OCR bbox | None |
| `--jpeg-quality` | JPEG quality (1-100) | 100 |
| `--output` | Output path | auto |
| `--device` | cuda/cpu | cuda |

## 💡 Usage Examples

### With CRAFT OCR
```bash
python inference.py --image doc.jpg --model ADCD-Net.pth --docres docres.pkl --qt-table qt_table.pk --craft craft_mlt_25k.pth
```

### With Precomputed OCR
```bash
python inference.py --image doc.jpg --model ADCD-Net.pth --docres docres.pkl --qt-table qt_table.pk --ocr-bbox doc_bbox.pkl
```

### CPU Mode
```bash
python inference.py --image doc.jpg --model ADCD-Net.pth --docres docres.pkl --qt-table qt_table.pk --device cpu
```

### Different JPEG Quality
```bash
python inference.py --image doc.jpg --model ADCD-Net.pth --docres docres.pkl --qt-table qt_table.pk --jpeg-quality 85
```

## 🔍 Utility Functions

```python
from inference_utils import *

# Create overlay
overlay = create_overlay(image, mask, color=(255,0,0), alpha=0.4)

# Calculate metrics (if ground truth available)
metrics = calculate_metrics(pred_mask, gt_mask)
# Returns: accuracy, precision, recall, f1, iou

# Morphological operations
clean_mask = apply_morphology(mask, operation='close', kernel_size=5)

# Forgery statistics
stats = get_forgery_statistics(mask)
# Returns: num_regions, total_pixels, mean_region_size, etc.

# Draw bounding boxes
bbox_img = draw_bboxes_on_image(image, mask, min_area=100)

# Visualize comparison
visualize_comparison(image, pred_mask, gt_mask, save_path='cmp.png')
```

## 📁 File Structure

```
ADCD-Net/
├── inference.py              # Main inference script
├── example_inference.py      # Simple example
├── batch_inference.py        # Batch processing
├── test_setup.py            # Setup verification
├── inference_utils.py       # Utility functions
├── requirements_inference.txt
├── INFERENCE_GUIDE.md       # Full documentation
└── INFERENCE_SUMMARY.md     # Detailed summary
```

## ✅ Quick Checklist

- [ ] Install dependencies: `pip install -r requirements_inference.txt`
- [ ] Download model files (ADCD-Net.pth, docres.pkl, qt_table.pk)
- [ ] Test setup: `python test_setup.py --model ... --docres ... --qt-table ...`
- [ ] Run inference: `python inference.py --image ...`

## 🐛 Common Issues

**CUDA Out of Memory**
→ Use `--device cpu`

**Import Error**
→ `pip install -r requirements_inference.txt`

**DCT Extraction Fails**
→ Install `jpeg2dct`: `pip install jpeg2dct`

**Slow Inference**
→ Use GPU: `--device cuda`

## 📊 Performance

- **GPU**: ~0.1-0.5s per image
- **CPU**: ~1-5s per image
- **Memory**: ~2-4GB GPU RAM

## 🎓 Citation

```bibtex
@inproceedings{wong2025adcd,
  title={ADCD-Net: Robust Document Image Forgery Localization...},
  author={Wong, Kahim and Zhou, Jicheng and Wu, Haiwei and Si, Yain-Whar and Zhou, Jiantao},
  booktitle={ICCV},
  year={2025}
}
```

## 📚 Documentation

- **Quick Start**: This card
- **Full Guide**: [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)
- **Summary**: [INFERENCE_SUMMARY.md](INFERENCE_SUMMARY.md)
- **Training**: [README.md](README.md)

---

**Need Help?** Check [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) for detailed instructions!
