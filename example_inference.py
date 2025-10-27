"""
Simple example script for running inference on a single image.

Usage:
    python example_inference.py

Make sure to update the paths in the CONFIG section below.

Note: This script will automatically use the DCT workaround if jpeg2dct is not available.
"""

from inference import SingleImageInference

# ==================== CONFIG ====================
# TODO: Update these paths to match your setup
CONFIG = {
    'model_ckpt': 'path/to/ADCD-Net.pth',          # ADCD-Net checkpoint
    'qt_table': 'path/to/qt_table.pk',             # Quantization table
    'docres_ckpt': None,                            # Optional: DocRes checkpoint (not needed for trained ADCD-Net)
    'craft_ckpt': None,                             # Optional: CRAFT checkpoint for OCR
    'device': 'cuda',                               # 'cuda' or 'cpu'
    'temp_dir': './temp_inference',                 # Directory for temporary files
}

# Image to test
TEST_IMAGE = 'path/to/test_image.jpg'              # Input image path
OUTPUT_PATH = 'prediction_result.png'              # Output visualization path
JPEG_QUALITY = 100                                  # JPEG quality (75-100 recommended)
OCR_BBOX_PATH = None                                # Optional: precomputed OCR bbox pickle

# ==================== INFERENCE ====================

def run_inference():
    """Run inference on a single image."""
    
    print("="*60)
    print("ADCD-Net Single Image Inference")
    print("="*60)
    
    # Initialize the inference pipeline
    print("\n[1/3] Initializing inference pipeline...")
    inferencer = SingleImageInference(
        model_ckpt_path=CONFIG['model_ckpt'],
        qt_table_path=CONFIG['qt_table'],
        docres_ckpt_path=CONFIG['docres_ckpt'],
        craft_ckpt_path=CONFIG['craft_ckpt'],
        device=CONFIG['device'],
        temp_dir=CONFIG['temp_dir']
    )
    
    # Run prediction
    print("\n[2/3] Running prediction...")
    results = inferencer.predict(
        img_path=TEST_IMAGE,
        jpeg_quality=JPEG_QUALITY,
        ocr_bbox_path=OCR_BBOX_PATH
    )
    
    # Visualize and save results
    print("\n[3/3] Visualizing results...")
    inferencer.visualize_results(results, save_path=OUTPUT_PATH)
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Input image: {TEST_IMAGE}")
    print(f"Output saved to: {OUTPUT_PATH}")
    print(f"Forgery pixels detected: {(results['pred_mask'] == 1).sum():,}")
    print(f"Total pixels: {results['pred_mask'].size:,}")
    print(f"Forgery ratio: {(results['pred_mask'] == 1).sum() / results['pred_mask'].size * 100:.2f}%")
    print(f"DCT alignment score: {results['align_score'][1]:.4f}")
    print(f"  (Higher score indicates better DCT grid alignment)")
    print("="*60)
    
    return results


if __name__ == '__main__':
    import os
    
    # Verify paths exist
    missing_files = []
    if not os.path.exists(CONFIG['model_ckpt']):
        missing_files.append(f"Model checkpoint: {CONFIG['model_ckpt']}")
    if not os.path.exists(CONFIG['docres_ckpt']):
        missing_files.append(f"DocRes checkpoint: {CONFIG['docres_ckpt']}")
    if not os.path.exists(CONFIG['qt_table']):
        missing_files.append(f"Quantization table: {CONFIG['qt_table']}")
    if not os.path.exists(TEST_IMAGE):
        missing_files.append(f"Test image: {TEST_IMAGE}")
    
    if missing_files:
        print("ERROR: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease update the paths in the CONFIG section of this script.")
        exit(1)
    
    # Run inference
    try:
        results = run_inference()
        print("\n✓ Inference completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
