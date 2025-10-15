# Single Image Inference Guide

This guide explains how to use ADCD-Net to perform document forgery localization on a single image.

## Files Overview

- **`inference.py`**: Main inference script with the `SingleImageInference` class
- **`example_inference.py`**: Simple example showing how to use the inference pipeline
- **Command-line usage**: Run inference directly from terminal

## Quick Start

### 1. Prepare Required Files

You need the following files (as mentioned in the main README):

- **ADCD-Net checkpoint**: `ADCD-Net.pth`
- **DocRes checkpoint**: `docres.pkl`
- **Quantization table**: `qt_table.pk`
- **CRAFT checkpoint** (optional): for automatic OCR mask generation
- **Test image**: Your document image to analyze

### 2. Option A: Using Example Script (Recommended for Beginners)

Edit `example_inference.py` and update the paths:

```python
CONFIG = {
    'model_ckpt': 'path/to/ADCD-Net.pth',
    'docres_ckpt': 'path/to/docres.pkl',
    'qt_table': 'path/to/qt_table.pk',
    'craft_ckpt': None,  # Optional
    'device': 'cuda',
}

TEST_IMAGE = 'path/to/your/test_image.jpg'
OUTPUT_PATH = 'prediction_result.png'
```

Then run:

```bash
python example_inference.py
```

### 3. Option B: Using Command Line

```bash
python inference.py \
    --image path/to/test_image.jpg \
    --model path/to/ADCD-Net.pth \
    --docres path/to/docres.pkl \
    --qt-table path/to/qt_table.pk \
    --output prediction_result.png \
    --jpeg-quality 100 \
    --device cuda
```

### 4. Option C: Using as a Python Module

```python
from inference import SingleImageInference

# Initialize
inferencer = SingleImageInference(
    model_ckpt_path='path/to/ADCD-Net.pth',
    docres_ckpt_path='path/to/docres.pkl',
    qt_table_path='path/to/qt_table.pk',
    craft_ckpt_path=None,  # Optional
    device='cuda'
)

# Run inference
results = inferencer.predict(
    img_path='path/to/test_image.jpg',
    jpeg_quality=100
)

# Visualize
inferencer.visualize_results(results, save_path='output.png')

# Access results
print(f"Forgery mask shape: {results['pred_mask'].shape}")
print(f"Forgery probability: {results['pred_prob'][1].mean():.4f}")
print(f"Alignment score: {results['align_score'][1]:.4f}")
```

## Command-Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--image` | ✓ | Path to input document image |
| `--model` | ✓ | Path to ADCD-Net checkpoint (.pth file) |
| `--docres` | ✓ | Path to DocRes checkpoint (.pkl file) |
| `--qt-table` | ✓ | Path to JPEG quantization table (.pk file) |
| `--craft` | ✗ | Path to CRAFT OCR model checkpoint |
| `--ocr-bbox` | ✗ | Path to precomputed OCR bbox pickle file |
| `--jpeg-quality` | ✗ | JPEG quality factor (1-100), default: 100 |
| `--output` | ✗ | Path to save visualization, default: `{image_name}_prediction.png` |
| `--device` | ✗ | Device to use ('cuda' or 'cpu'), default: 'cuda' |

## Output Explanation

The inference script generates a visualization with 6 panels:

1. **Original Image**: The input document image
2. **Compressed Image**: Image after JPEG compression
3. **OCR Mask**: Character regions detected by CRAFT (or provided)
4. **Predicted Forgery Mask**: Binary mask (0=authentic, 1=forged)
5. **Forgery Probability**: Heatmap showing forgery likelihood (0-1)
6. **Forgery Overlay**: Original image with forgery regions highlighted in red

### Results Dictionary

The `predict()` method returns a dictionary with:

```python
{
    'pred_mask': np.ndarray,      # Binary mask (H, W): 0=authentic, 1=forged
    'pred_prob': np.ndarray,      # Probability map (2, H, W): class probabilities
    'align_score': np.ndarray,    # DCT alignment score [non-aligned, aligned]
    'original_img': np.ndarray,   # Original RGB image
    'compressed_img': np.ndarray, # JPEG compressed image
    'ocr_mask': np.ndarray        # OCR character mask
}
```

## OCR Mask Options

The inference pipeline needs an OCR mask to identify text regions. There are three options:

### Option 1: Use CRAFT Model (Automatic)
Provide the CRAFT checkpoint path to automatically detect text:
```bash
python inference.py --image test.jpg --craft path/to/craft.pth ...
```

### Option 2: Use Precomputed OCR Bbox
If you already have OCR bboxes saved as a pickle file:
```bash
python inference.py --image test.jpg --ocr-bbox path/to/bbox.pkl ...
```

The pickle file should contain a list of bounding boxes:
```python
[(x1, y1, x2, y2), (x1, y1, x2, y2), ...]
```

### Option 3: No OCR (Empty Mask)
If you don't provide CRAFT or precomputed bboxes, an empty mask will be used:
```bash
python inference.py --image test.jpg ...
```

> **Note**: Using an empty OCR mask may reduce accuracy, as the model is designed to leverage text region information.

## Generating OCR Masks for New Images

To generate OCR masks for images not in DocTamper dataset:

1. Download CRAFT model from [CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch)
2. Use the `seg_char/main.py` script:

```python
from seg_char.main import CharSeger

char_seger = CharSeger(
    ckpt_path='path/to/craft_mlt_25k.pth',
    save_dir='output_dir'
)

char_seger.seg_char_per_img(img_path='path/to/image.jpg')
```

This will save the OCR bboxes as a pickle file in `output_dir/bbox/`.

## JPEG Quality Parameter

The `--jpeg-quality` parameter controls the JPEG compression applied during inference:

- **100**: No compression (recommended for testing)
- **95**: Slight compression
- **85**: Moderate compression
- **75**: Higher compression (more challenging for detection)

Lower quality may affect detection performance. For evaluation, use quality matching your use case.

## Tips for Best Results

1. **Image Size**: The model works best with images around 256×256 to 512×512 pixels
2. **JPEG Quality**: Use quality 95-100 for best results
3. **OCR Mask**: Providing accurate OCR masks improves performance
4. **Device**: Use CUDA/GPU for faster inference (CPU is ~10x slower)
5. **File Format**: Use JPEG or PNG images

## Troubleshooting

### CUDA Out of Memory
If you get CUDA OOM errors:
- Use `--device cpu`
- Or reduce image size before inference

### DCT Extraction Fails
If DCT extraction fails, the script will use a dummy DCT array. This may reduce accuracy.

### No CRAFT Model
If you don't have the CRAFT model, you can:
- Use precomputed OCR bboxes (`--ocr-bbox`)
- Or run without OCR mask (reduced accuracy)

### Import Errors
Make sure all dependencies are installed:
```bash
pip install torch torchvision opencv-python pillow numpy matplotlib jpeg2dct albumentations
```

## Examples

### Basic inference with CRAFT OCR:
```bash
python inference.py \
    --image my_document.jpg \
    --model ADCD-Net.pth \
    --docres docres.pkl \
    --qt-table qt_table.pk \
    --craft craft_mlt_25k.pth \
    --output result.png
```

### Inference with precomputed OCR:
```bash
python inference.py \
    --image my_document.jpg \
    --model ADCD-Net.pth \
    --docres docres.pkl \
    --qt-table qt_table.pk \
    --ocr-bbox my_document_bbox.pkl \
    --output result.png
```

### Batch processing multiple images:
```python
from inference import SingleImageInference
from pathlib import Path

# Initialize once
inferencer = SingleImageInference(
    model_ckpt_path='ADCD-Net.pth',
    docres_ckpt_path='docres.pkl',
    qt_table_path='qt_table.pk',
    device='cuda'
)

# Process multiple images
for img_path in Path('images/').glob('*.jpg'):
    results = inferencer.predict(str(img_path))
    inferencer.visualize_results(
        results, 
        save_path=f'outputs/{img_path.stem}_pred.png'
    )
```

## Performance Notes

- **GPU Inference**: ~0.1-0.5 seconds per image (depending on size)
- **CPU Inference**: ~1-5 seconds per image
- **Memory Usage**: ~2-4GB GPU memory for typical document images

## Citation

If you use this inference code in your research, please cite:

```bibtex
@inproceedings{wong2025adcd,
  title={ADCD-Net: Robust Document Image Forgery Localization via Adaptive DCT Feature and Hierarchical Content Disentanglement},
  author={Wong, Kahim and Zhou, Jicheng and Wu, Haiwei and Si, Yain-Whar and Zhou, Jiantao},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  year={2025}
}
```
