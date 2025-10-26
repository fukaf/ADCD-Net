"""
Batch Inference Script for ADCD-Net

This script processes multiple images in a directory and saves predictions.

Note: This script will automatically use the DCT workaround if jpeg2dct is not available.

Usage:
    python batch_inference.py \
        --input-dir path/to/images/ \
        --output-dir path/to/outputs/ \
        --model path/to/ADCD-Net.pth \
        --docres path/to/docres.pkl \
        --qt-table path/to/qt_table.pk \
        [--craft path/to/craft.pth] \
        [--device cuda]
"""

import argparse
import os
from pathlib import Path
import time
import json

from inference import SingleImageInference


def process_batch(input_dir, output_dir, inferencer, jpeg_quality=100, save_masks=True):
    """
    Process all images in a directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save results
        inferencer: SingleImageInference instance
        jpeg_quality: JPEG quality factor
        save_masks: Whether to save binary masks as numpy files
    
    Returns:
        Dictionary with processing statistics
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    if save_masks:
        mask_dir = os.path.join(output_dir, 'masks')
        os.makedirs(mask_dir, exist_ok=True)
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(Path(input_dir).glob(f'*{ext}'))
        image_paths.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return {}
    
    print(f"Found {len(image_paths)} images to process")
    
    # Process each image
    results_summary = []
    total_time = 0
    
    for idx, img_path in enumerate(image_paths, 1):
        img_name = img_path.stem
        print(f"\n[{idx}/{len(image_paths)}] Processing: {img_path.name}")
        
        try:
            # Run inference
            start_time = time.time()
            results = inferencer.predict(
                img_path=str(img_path),
                jpeg_quality=jpeg_quality
            )
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # Save visualization
            vis_path = os.path.join(vis_dir, f'{img_name}_prediction.png')
            inferencer.visualize_results(results, save_path=vis_path)
            
            # Save binary mask if requested
            if save_masks:
                import numpy as np
                mask_path = os.path.join(mask_dir, f'{img_name}_mask.npy')
                np.save(mask_path, results['pred_mask'])
            
            # Calculate statistics
            total_pixels = results['pred_mask'].size
            forgery_pixels = (results['pred_mask'] == 1).sum()
            forgery_ratio = forgery_pixels / total_pixels * 100
            
            # Store summary
            result_info = {
                'image': img_path.name,
                'total_pixels': int(total_pixels),
                'forgery_pixels': int(forgery_pixels),
                'forgery_ratio': float(forgery_ratio),
                'alignment_score': float(results['align_score'][1]),
                'inference_time': float(inference_time),
                'status': 'success'
            }
            results_summary.append(result_info)
            
            print(f"  ✓ Forgery ratio: {forgery_ratio:.2f}%")
            print(f"  ✓ Alignment score: {results['align_score'][1]:.4f}")
            print(f"  ✓ Time: {inference_time:.3f}s")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results_summary.append({
                'image': img_path.name,
                'status': 'failed',
                'error': str(e)
            })
    
    # Save summary as JSON
    summary_path = os.path.join(output_dir, 'results_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Print overall statistics
    successful = sum(1 for r in results_summary if r['status'] == 'success')
    failed = len(results_summary) - successful
    avg_time = total_time / successful if successful > 0 else 0
    
    stats = {
        'total_images': len(image_paths),
        'successful': successful,
        'failed': failed,
        'total_time': total_time,
        'average_time_per_image': avg_time
    }
    
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"Total images: {stats['total_images']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total time: {stats['total_time']:.2f}s")
    print(f"Average time per image: {stats['average_time_per_image']:.3f}s")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Visualizations: {vis_dir}")
    if save_masks:
        print(f"  - Binary masks: {mask_dir}")
    print(f"  - Summary JSON: {summary_path}")
    print("="*60)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='ADCD-Net Batch Inference')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing input images')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save results')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to ADCD-Net checkpoint')
    parser.add_argument('--qt-table', type=str, required=True,
                       help='Path to quantization table pickle')
    parser.add_argument('--docres', type=str, default=None,
                       help='Path to DocRes checkpoint (optional, not needed for trained ADCD-Net)')
    parser.add_argument('--craft', type=str, default=None,
                       help='Path to CRAFT checkpoint (optional)')
    parser.add_argument('--jpeg-quality', type=int, default=100,
                       help='JPEG quality factor (1-100)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--no-save-masks', action='store_true',
                       help='Do not save binary masks as .npy files')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: Model checkpoint not found: {args.model}")
        return
    
    if not os.path.exists(args.qt_table):
        print(f"Error: Quantization table not found: {args.qt_table}")
        return
    
    if args.docres and not os.path.exists(args.docres):
        print(f"Warning: DocRes checkpoint not found: {args.docres}")
        print("Continuing without docres (OK if using trained ADCD-Net checkpoint)")
        args.docres = None
    
    # Initialize inference pipeline
    print("Initializing ADCD-Net inference pipeline...")
    inferencer = SingleImageInference(
        model_ckpt_path=args.model,
        qt_table_path=args.qt_table,
        docres_ckpt_path=args.docres,
        craft_ckpt_path=args.craft,
        device=args.device
    )
    
    # Process batch
    stats = process_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        inferencer=inferencer,
        jpeg_quality=args.jpeg_quality,
        save_masks=not args.no_save_masks
    )
    
    if stats.get('failed', 0) > 0:
        print("\nSome images failed to process. Check the summary JSON for details.")
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
