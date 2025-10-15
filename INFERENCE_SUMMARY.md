# ADCD-Net Inference Scripts - Summary

This document summarizes all the inference scripts created for testing ADCD-Net on single images or batches outside the DocTamper dataset.

## 📁 Files Created

### Core Scripts
1. **`inference.py`** - Main inference pipeline with `SingleImageInference` class
2. **`example_inference.py`** - Simple example script for beginners
3. **`batch_inference.py`** - Batch processing for multiple images
4. **`test_setup.py`** - Setup verification and testing script

### Supporting Files
5. **`inference_utils.py`** - Utility functions for visualization and metrics
6. **`requirements_inference.txt`** - Python dependencies for inference
7. **`INFERENCE_GUIDE.md`** - Comprehensive user guide

### Updated Files
8. **`README.md`** - Updated with inference instructions

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_inference.txt
```

### 2. Download Required Files
You need:
- `ADCD-Net.pth` - Model checkpoint
- `docres.pkl` - DocRes checkpoint  
- `qt_table.pk` - Quantization tables
- Optional: `craft_mlt_25k.pth` - CRAFT OCR model

### 3. Test Your Setup
```bash
python test_setup.py \
    --model path/to/ADCD-Net.pth \
    --docres path/to/docres.pkl \
    --qt-table path/to/qt_table.pk
```

### 4. Run Inference

#### Option A: Simple Example (Recommended for First-Time Users)
Edit `example_inference.py` to set paths, then:
```bash
python example_inference.py
```

#### Option B: Command Line
```bash
python inference.py \
    --image your_image.jpg \
    --model path/to/ADCD-Net.pth \
    --docres path/to/docres.pkl \
    --qt-table path/to/qt_table.pk \
    --output result.png
```

#### Option C: Batch Processing
```bash
python batch_inference.py \
    --input-dir images/ \
    --output-dir outputs/ \
    --model path/to/ADCD-Net.pth \
    --docres path/to/docres.pkl \
    --qt-table path/to/qt_table.pk
```

#### Option D: Python API
```python
from inference import SingleImageInference

inferencer = SingleImageInference(
    model_ckpt_path='ADCD-Net.pth',
    docres_ckpt_path='docres.pkl',
    qt_table_path='qt_table.pk',
    device='cuda'
)

results = inferencer.predict('image.jpg')
inferencer.visualize_results(results, save_path='output.png')
```

## 📊 Output Formats

### Visualization
6-panel figure showing:
1. Original image
2. JPEG compressed image
3. OCR mask (character regions)
4. Predicted forgery mask (binary)
5. Forgery probability heatmap
6. Overlay (image + forgery regions in red)

### Results Dictionary
```python
{
    'pred_mask': np.ndarray,      # Binary mask (H, W)
    'pred_prob': np.ndarray,      # Probability (2, H, W)
    'align_score': np.ndarray,    # DCT alignment [non-aligned, aligned]
    'original_img': np.ndarray,   # Original RGB
    'compressed_img': np.ndarray, # After JPEG
    'ocr_mask': np.ndarray        # Character mask
}
```

### Batch Processing Output
```
output_dir/
├── visualizations/          # PNG visualizations
│   ├── image1_prediction.png
│   └── image2_prediction.png
├── masks/                   # Binary masks (.npy)
│   ├── image1_mask.npy
│   └── image2_mask.npy
└── results_summary.json     # Metrics for all images
```

## 🔧 Key Features

### `inference.py` - SingleImageInference Class

**Methods:**
- `__init__()` - Initialize model and load checkpoints
- `predict()` - Run inference on single image
- `visualize_results()` - Create visualization
- `extract_ocr_mask()` - Get OCR mask (CRAFT or precomputed)
- `preprocess_image()` - JPEG compression + DCT extraction

**Features:**
- Automatic JPEG compression with configurable quality
- DCT coefficient extraction
- Optional CRAFT OCR integration
- Support for precomputed OCR masks
- GPU/CPU compatible

### `example_inference.py`

- Beginner-friendly configuration
- Clear step-by-step execution
- Automatic path validation
- Detailed result summary

### `batch_inference.py`

- Process entire directories
- Progress tracking
- JSON summary with metrics
- Error handling for individual failures
- Save both visualizations and binary masks

### `test_setup.py`

**Tests:**
1. ✓ Python package dependencies
2. ✓ CUDA/GPU availability
3. ✓ Required files exist
4. ✓ Model loading
5. ✓ Inference pipeline

### `inference_utils.py`

**Utility Functions:**
- `create_overlay()` - Overlay forgery on image
- `visualize_comparison()` - Compare with ground truth
- `calculate_metrics()` - Compute precision/recall/F1
- `apply_morphology()` - Clean up masks
- `get_forgery_statistics()` - Region analysis
- `draw_bboxes_on_image()` - Draw bounding boxes

## 📝 Common Use Cases

### 1. Test on Single Image
```bash
python inference.py --image test.jpg --model ADCD-Net.pth --docres docres.pkl --qt-table qt_table.pk
```

### 2. Batch Process Directory
```bash
python batch_inference.py --input-dir images/ --output-dir results/ --model ADCD-Net.pth --docres docres.pkl --qt-table qt_table.pk
```

### 3. Use in Your Code
```python
from inference import SingleImageInference

inferencer = SingleImageInference(
    model_ckpt_path='ADCD-Net.pth',
    docres_ckpt_path='docres.pkl',
    qt_table_path='qt_table.pk'
)

# Process multiple images
for img_path in ['img1.jpg', 'img2.jpg']:
    results = inferencer.predict(img_path)
    print(f"Forgery in {img_path}: {results['pred_mask'].sum()} pixels")
```

### 4. Compare with Ground Truth
```python
from inference_utils import visualize_comparison, calculate_metrics
import numpy as np

# Load ground truth
gt_mask = np.load('ground_truth.npy')

# Get prediction
results = inferencer.predict('image.jpg')

# Calculate metrics
metrics = calculate_metrics(results['pred_mask'], gt_mask)
print(f"F1 Score: {metrics['f1']:.3f}")

# Visualize
visualize_comparison(results['original_img'], results['pred_mask'], gt_mask, 'comparison.png')
```

### 5. Post-process Masks
```python
from inference_utils import apply_morphology, get_forgery_statistics

# Clean up mask
clean_mask = apply_morphology(results['pred_mask'], operation='close', kernel_size=5)

# Analyze forgery regions
stats = get_forgery_statistics(clean_mask)
print(f"Number of forgery regions: {stats['num_regions']}")
print(f"Largest region: {stats['largest_region_size']} pixels")
```

## 🎯 Tips for Best Results

1. **Image Quality**: Use high-resolution images (256×256 to 512×512)
2. **JPEG Quality**: Use 95-100 for best detection accuracy
3. **OCR Masks**: Provide CRAFT model or precomputed OCR for better results
4. **Device**: Use GPU (CUDA) for ~10x faster inference
5. **File Format**: JPEG or PNG work best

## ⚙️ Configuration Options

### JPEG Quality (`--jpeg-quality`)
- **100**: No compression (recommended)
- **95**: Slight compression
- **85**: Moderate compression
- **75**: Higher compression (more challenging)

### Device (`--device`)
- **cuda**: Use GPU (fast, requires CUDA)
- **cpu**: Use CPU (slower, no GPU needed)

### OCR Options
- **CRAFT model** (`--craft`): Automatic text detection
- **Precomputed** (`--ocr-bbox`): Use saved bounding boxes
- **None**: Empty mask (reduced accuracy)

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Use CPU instead
python inference.py --image test.jpg --device cpu ...
```

### Missing Dependencies
```bash
pip install -r requirements_inference.txt
```

### DCT Extraction Fails
- Ensure `jpeg2dct` is installed: `pip install jpeg2dct`
- Script will use dummy DCT if extraction fails

### Slow Inference
- Use GPU: `--device cuda`
- Reduce image size before inference
- For batch: Process fewer images at once

## 📦 Dependencies

Core requirements:
- Python 3.8+
- PyTorch 1.11+
- OpenCV
- Pillow
- NumPy
- Matplotlib
- jpeg2dct
- albumentations

See `requirements_inference.txt` for complete list.

## 📖 Documentation

- **Main Guide**: [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)
- **Training**: See main [README.md](README.md)
- **Paper**: [arXiv:2507.16397](https://arxiv.org/abs/2507.16397)

## 🎓 Citation

```bibtex
@inproceedings{wong2025adcd,
  title={ADCD-Net: Robust Document Image Forgery Localization via Adaptive DCT Feature and Hierarchical Content Disentanglement},
  author={Wong, Kahim and Zhou, Jicheng and Wu, Haiwei and Si, Yain-Whar and Zhou, Jiantao},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  year={2025}
}
```

## 💡 Example Workflow

```bash
# 1. Verify setup
python test_setup.py --model ADCD-Net.pth --docres docres.pkl --qt-table qt_table.pk

# 2. Test on single image
python inference.py --image sample.jpg --model ADCD-Net.pth --docres docres.pkl --qt-table qt_table.pk

# 3. Process entire directory
python batch_inference.py --input-dir my_images/ --output-dir results/ --model ADCD-Net.pth --docres docres.pkl --qt-table qt_table.pk

# 4. Review results
# Check results/visualizations/ for images
# Check results/results_summary.json for metrics
```

## ✅ What's Included

✓ Single image inference  
✓ Batch processing  
✓ Automatic OCR with CRAFT  
✓ Precomputed OCR support  
✓ Visualization tools  
✓ Metrics calculation  
✓ Setup verification  
✓ Comprehensive documentation  
✓ Example scripts  
✓ Utility functions  

## 🚦 Status

All inference scripts are ready to use! Simply:
1. Install dependencies
2. Download model checkpoints
3. Run test_setup.py to verify
4. Start inferencing!

---

For questions or issues, please refer to the [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) or open an issue on GitHub.
