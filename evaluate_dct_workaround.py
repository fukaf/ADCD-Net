"""
Evaluation Script: Compare jpeg2dct vs OpenCV DCT Workaround

This script compares the DCT coefficients extracted using:
1. jpeg2dct library (ground truth)
2. OpenCV cv2.dct() workaround

Metrics evaluated:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Correlation coefficient
- Value distribution comparison
- Visual comparison of DCT patterns
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
import os
from scipy.stats import pearsonr

try:
    from jpeg2dct.numpy import load as jpeg2dct_load
    JPEG2DCT_AVAILABLE = True
except ImportError:
    JPEG2DCT_AVAILABLE = False
    print("Warning: jpeg2dct not available, will only test workaround")

from dct_workaround import (
    extract_dct_opencv_blockwise,
    convert_blockwise_to_spatial,
    extract_dct_for_adcdnet
)


def create_test_image(size=(512, 512), pattern='mixed'):
    """
    Create test images with different patterns.
    
    Args:
        size: Image size (H, W)
        pattern: 'random', 'gradient', 'checkerboard', 'text', 'mixed'
    
    Returns:
        RGB image as numpy array
    """
    h, w = size
    
    if pattern == 'random':
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    
    elif pattern == 'gradient':
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            img[i, :, :] = int(255 * i / h)
    
    elif pattern == 'checkerboard':
        img = np.zeros((h, w, 3), dtype=np.uint8)
        block_size = 32
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                if ((i // block_size) + (j // block_size)) % 2 == 0:
                    img[i:i+block_size, j:j+block_size] = 255
    
    elif pattern == 'text':
        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        # Draw some text-like patterns
        for i in range(10):
            y = 50 + i * 40
            cv2.rectangle(img, (50, y), (w-50, y+20), (0, 0, 0), -1)
    
    elif pattern == 'mixed':
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Gradient
        img[:h//2, :w//2] = create_test_image((h//2, w//2), 'gradient')
        # Checkerboard
        img[:h//2, w//2:] = create_test_image((h//2, w//2), 'checkerboard')
        # Random
        img[h//2:, :w//2] = create_test_image((h//2, w//2), 'random')
        # White
        img[h//2:, w//2:] = 255
    
    return img


def compare_dct_methods(image, quality_factor=90, save_jpeg=True):
    """
    Compare DCT extraction between jpeg2dct and OpenCV workaround.
    
    Args:
        image: RGB image as numpy array
        quality_factor: JPEG compression quality
        save_jpeg: Whether to save as JPEG first
    
    Returns:
        Dictionary with comparison results
    """
    results = {
        'quality_factor': quality_factor,
        'image_shape': image.shape,
    }
    
    if save_jpeg:
        # Save as JPEG to simulate real scenario
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img_pil = Image.fromarray(image)
            img_pil.save(tmp.name, 'JPEG', quality=quality_factor)
            tmp_path = tmp.name
    else:
        tmp_path = None
    
    try:
        # Method 1: jpeg2dct (ground truth)
        if JPEG2DCT_AVAILABLE and tmp_path:
            dct_jpeg2dct_blocks, _, _ = jpeg2dct_load(tmp_path, normalized=False)
            dct_jpeg2dct = convert_blockwise_to_spatial(dct_jpeg2dct_blocks)
            results['jpeg2dct_available'] = True
            results['jpeg2dct_shape'] = dct_jpeg2dct.shape
            results['jpeg2dct_range'] = (dct_jpeg2dct.min(), dct_jpeg2dct.max())
            results['jpeg2dct_mean'] = dct_jpeg2dct.mean()
            results['jpeg2dct_std'] = dct_jpeg2dct.std()
        else:
            dct_jpeg2dct = None
            results['jpeg2dct_available'] = False
        
        # Method 2: OpenCV workaround
        input_source = tmp_path if tmp_path else image
        dct_opencv = extract_dct_for_adcdnet(input_source, quality_factor=quality_factor)
        results['opencv_shape'] = dct_opencv.shape
        results['opencv_range'] = (dct_opencv.min(), dct_opencv.max())
        results['opencv_mean'] = dct_opencv.mean()
        results['opencv_std'] = dct_opencv.std()
        
        # Compare if both available
        if dct_jpeg2dct is not None:
            # Ensure same shape
            min_h = min(dct_jpeg2dct.shape[0], dct_opencv.shape[0])
            min_w = min(dct_jpeg2dct.shape[1], dct_opencv.shape[1])
            dct_jpeg2dct = dct_jpeg2dct[:min_h, :min_w]
            dct_opencv = dct_opencv[:min_h, :min_w]
            
            # Calculate metrics
            diff = dct_opencv - dct_jpeg2dct
            results['mae'] = np.abs(diff).mean()
            results['mse'] = (diff ** 2).mean()
            results['rmse'] = np.sqrt(results['mse'])
            results['max_diff'] = np.abs(diff).max()
            
            # Correlation
            flat_jpeg2dct = dct_jpeg2dct.flatten()
            flat_opencv = dct_opencv.flatten()
            results['correlation'], results['correlation_pvalue'] = pearsonr(flat_jpeg2dct, flat_opencv)
            
            # Relative error
            results['relative_mae'] = results['mae'] / (np.abs(dct_jpeg2dct).mean() + 1e-8)
            
            # Store arrays for visualization
            results['dct_jpeg2dct'] = dct_jpeg2dct
            results['dct_opencv'] = dct_opencv
            results['diff'] = diff
        
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    return results


def visualize_comparison(results, save_path=None):
    """
    Visualize the comparison between jpeg2dct and OpenCV DCT.
    
    Args:
        results: Dictionary from compare_dct_methods()
        save_path: Path to save figure (optional)
    """
    if not results['jpeg2dct_available']:
        print("Cannot visualize: jpeg2dct not available")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Get DCT arrays
    dct_jpeg2dct = results['dct_jpeg2dct']
    dct_opencv = results['dct_opencv']
    diff = results['diff']
    
    # Plot 1: jpeg2dct
    im1 = axes[0, 0].imshow(np.abs(dct_jpeg2dct), cmap='hot', vmin=0, vmax=20)
    axes[0, 0].set_title('jpeg2dct (Ground Truth)')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: OpenCV
    im2 = axes[0, 1].imshow(np.abs(dct_opencv), cmap='hot', vmin=0, vmax=20)
    axes[0, 1].set_title('OpenCV Workaround')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot 3: Absolute Difference
    im3 = axes[0, 2].imshow(np.abs(diff), cmap='hot')
    axes[0, 2].set_title(f'Absolute Difference (MAE={results["mae"]:.2f})')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Plot 4: Histogram comparison
    axes[1, 0].hist(dct_jpeg2dct.flatten(), bins=50, alpha=0.5, label='jpeg2dct', density=True)
    axes[1, 0].hist(dct_opencv.flatten(), bins=50, alpha=0.5, label='opencv', density=True)
    axes[1, 0].set_xlabel('DCT Coefficient Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Value Distribution')
    axes[1, 0].legend()
    axes[1, 0].set_xlim(-100, 100)
    
    # Plot 5: Scatter plot
    sample_size = min(10000, dct_jpeg2dct.size)
    indices = np.random.choice(dct_jpeg2dct.size, sample_size, replace=False)
    axes[1, 1].scatter(dct_jpeg2dct.flatten()[indices], 
                       dct_opencv.flatten()[indices], 
                       alpha=0.1, s=1)
    axes[1, 1].plot([-100, 100], [-100, 100], 'r--', label='y=x')
    axes[1, 1].set_xlabel('jpeg2dct')
    axes[1, 1].set_ylabel('OpenCV')
    axes[1, 1].set_title(f'Correlation={results["correlation"]:.4f}')
    axes[1, 1].legend()
    axes[1, 1].set_xlim(-100, 100)
    axes[1, 1].set_ylim(-100, 100)
    
    # Plot 6: Error distribution
    axes[1, 2].hist(diff.flatten(), bins=50, density=True)
    axes[1, 2].set_xlabel('Error (opencv - jpeg2dct)')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].set_title(f'Error Distribution (RMSE={results["rmse"]:.2f})')
    axes[1, 2].axvline(0, color='r', linestyle='--')
    
    plt.suptitle(f'DCT Comparison: QF={results["quality_factor"]}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_comparison_summary(results):
    """Print a summary of the comparison."""
    print("\n" + "="*60)
    print("DCT EXTRACTION COMPARISON SUMMARY")
    print("="*60)
    print(f"Image shape: {results['image_shape']}")
    print(f"JPEG Quality: {results['quality_factor']}")
    print()
    
    if results['jpeg2dct_available']:
        print("jpeg2dct (Ground Truth):")
        print(f"  Shape: {results['jpeg2dct_shape']}")
        print(f"  Range: [{results['jpeg2dct_range'][0]}, {results['jpeg2dct_range'][1]}]")
        print(f"  Mean: {results['jpeg2dct_mean']:.2f}")
        print(f"  Std: {results['jpeg2dct_std']:.2f}")
        print()
    
    print("OpenCV Workaround:")
    print(f"  Shape: {results['opencv_shape']}")
    print(f"  Range: [{results['opencv_range'][0]}, {results['opencv_range'][1]}]")
    print(f"  Mean: {results['opencv_mean']:.2f}")
    print(f"  Std: {results['opencv_std']:.2f}")
    print()
    
    if results['jpeg2dct_available']:
        print("Comparison Metrics:")
        print(f"  MAE (Mean Absolute Error): {results['mae']:.4f}")
        print(f"  RMSE (Root Mean Squared Error): {results['rmse']:.4f}")
        print(f"  Max Absolute Error: {results['max_diff']:.4f}")
        print(f"  Relative MAE: {results['relative_mae']*100:.2f}%")
        print(f"  Correlation: {results['correlation']:.6f}")
        print()
        
        # Interpretation
        print("Interpretation:")
        if results['correlation'] > 0.95:
            print("  ✓ Very high correlation - workaround is excellent")
        elif results['correlation'] > 0.85:
            print("  ✓ High correlation - workaround is good")
        elif results['correlation'] > 0.70:
            print("  ⚠ Moderate correlation - workaround is acceptable")
        else:
            print("  ✗ Low correlation - workaround may not be suitable")
        
        if results['relative_mae'] < 0.1:
            print("  ✓ Low relative error - good approximation")
        elif results['relative_mae'] < 0.3:
            print("  ⚠ Moderate relative error - acceptable")
        else:
            print("  ✗ High relative error - significant difference")
    
    print("="*60)


def run_comprehensive_evaluation():
    """Run comprehensive evaluation across multiple scenarios."""
    print("\n" + "="*60)
    print("COMPREHENSIVE DCT WORKAROUND EVALUATION")
    print("="*60)
    
    if not JPEG2DCT_AVAILABLE:
        print("\n⚠ Warning: jpeg2dct not available")
        print("Installing: pip install jpeg2dct")
        print("Running workaround-only tests...")
    
    # Test scenarios
    test_cases = [
        {'pattern': 'random', 'quality': 100, 'size': (256, 256)},
        {'pattern': 'random', 'quality': 90, 'size': (256, 256)},
        {'pattern': 'random', 'quality': 75, 'size': (256, 256)},
        {'pattern': 'gradient', 'quality': 90, 'size': (256, 256)},
        {'pattern': 'checkerboard', 'quality': 90, 'size': (256, 256)},
        {'pattern': 'text', 'quality': 90, 'size': (256, 256)},
        {'pattern': 'mixed', 'quality': 90, 'size': (512, 512)},
    ]
    
    all_results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n[Test {i+1}/{len(test_cases)}] Pattern: {test_case['pattern']}, QF: {test_case['quality']}")
        
        # Create test image
        img = create_test_image(test_case['size'], test_case['pattern'])
        
        # Compare methods
        results = compare_dct_methods(img, test_case['quality'])
        results['test_case'] = test_case
        all_results.append(results)
        
        # Print summary
        print_comparison_summary(results)
        
        # Save visualization for first test
        if i == 0 and results['jpeg2dct_available']:
            visualize_comparison(results, save_path=f'dct_comparison_{test_case["pattern"]}_qf{test_case["quality"]}.png')
    
    # Overall summary
    if JPEG2DCT_AVAILABLE and len(all_results) > 0:
        print("\n" + "="*60)
        print("OVERALL SUMMARY")
        print("="*60)
        
        correlations = [r['correlation'] for r in all_results if 'correlation' in r]
        maes = [r['mae'] for r in all_results if 'mae' in r]
        rel_maes = [r['relative_mae'] for r in all_results if 'relative_mae' in r]
        
        if correlations:
            print(f"Average Correlation: {np.mean(correlations):.6f} ± {np.std(correlations):.6f}")
            print(f"Average MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f}")
            print(f"Average Relative MAE: {np.mean(rel_maes)*100:.2f}%")
            print()
            print("Recommendation:")
            
            avg_corr = np.mean(correlations)
            avg_rel_mae = np.mean(rel_maes)
            
            if avg_corr > 0.9 and avg_rel_mae < 0.2:
                print("  ✓✓ OpenCV workaround is EXCELLENT - safe to use")
            elif avg_corr > 0.8 and avg_rel_mae < 0.3:
                print("  ✓ OpenCV workaround is GOOD - acceptable for most uses")
            elif avg_corr > 0.7:
                print("  ⚠ OpenCV workaround is ACCEPTABLE - may affect accuracy")
            else:
                print("  ✗ OpenCV workaround shows significant differences")
                print("  Consider using jpeg2dct for critical applications")
        
        print("="*60)


if __name__ == '__main__':
    # Run comprehensive evaluation
    run_comprehensive_evaluation()
    
    print("\n✓ Evaluation complete!")
    print("\nTo use the workaround in your code:")
    print("  from dct_workaround import extract_dct_for_adcdnet")
    print("  dct = extract_dct_for_adcdnet('image.jpg', quality_factor=90)")
