"""
Visual Comparison of jpeg2dct vs OpenCV DCT Extraction

This script processes test images from the input folder and generates
comprehensive visualizations comparing both DCT extraction methods.

Usage:
    1. Place test images in: test_dct_comparison/input/
    2. Run: python visualize_dct_comparison.py
    3. Check results in: test_dct_comparison/output/
"""

import os
import sys
import glob
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tempfile

# Try to import jpeg2dct
try:
    from jpeg2dct.numpy import load as jpeg2dct_load
    JPEG2DCT_AVAILABLE = True
    print("✓ jpeg2dct library available")
except ImportError:
    JPEG2DCT_AVAILABLE = False
    print("⚠ jpeg2dct library not available - will show OpenCV results only")

# Import workaround
from dct_workaround import extract_dct_opencv_blockwise, convert_blockwise_to_spatial


# Configuration
INPUT_DIR = 'test_dct_comparison/input'
OUTPUT_DIR = 'test_dct_comparison/output'
QUALITY_FACTORS = [100, 90, 75]  # Test different JPEG qualities


def ensure_directories():
    """Create input/output directories if they don't exist."""
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n📁 Directories:")
    print(f"  Input:  {os.path.abspath(INPUT_DIR)}")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}")


def extract_dct_both_methods(image_path, quality_factor=90):
    """
    Extract DCT using both methods.
    
    Args:
        image_path: Path to input image
        quality_factor: JPEG quality factor
    
    Returns:
        Dictionary with results from both methods
    """
    results = {
        'image_path': image_path,
        'quality_factor': quality_factor,
    }
    
    # Load original image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    results['original_image'] = img_array
    
    # Save as JPEG with specified quality
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img.save(tmp.name, 'JPEG', quality=quality_factor)
        tmp_path = tmp.name
    
    try:
        # Method 1: jpeg2dct (if available)
        if JPEG2DCT_AVAILABLE:
            try:
                dct_blocks_jpeg2dct, _, _ = jpeg2dct_load(tmp_path, normalized=False)
                dct_jpeg2dct = convert_blockwise_to_spatial(dct_blocks_jpeg2dct)
                results['dct_jpeg2dct'] = dct_jpeg2dct
                results['jpeg2dct_success'] = True
                print(f"    ✓ jpeg2dct: shape={dct_jpeg2dct.shape}, range=[{dct_jpeg2dct.min()}, {dct_jpeg2dct.max()}]")
            except Exception as e:
                print(f"    ✗ jpeg2dct failed: {e}")
                results['jpeg2dct_success'] = False
        else:
            results['jpeg2dct_success'] = False
        
        # Method 2: OpenCV workaround
        try:
            dct_blocks_opencv, _, _ = extract_dct_opencv_blockwise(tmp_path, quality_factor=quality_factor)
            dct_opencv = convert_blockwise_to_spatial(dct_blocks_opencv)
            dct_opencv = np.int32(dct_opencv)  # Convert to int32 like jpeg2dct
            results['dct_opencv'] = dct_opencv
            results['opencv_success'] = True
            print(f"    ✓ OpenCV:   shape={dct_opencv.shape}, range=[{dct_opencv.min()}, {dct_opencv.max()}]")
        except Exception as e:
            print(f"    ✗ OpenCV failed: {e}")
            results['opencv_success'] = False
            
        # Load compressed image
        compressed_img = Image.open(tmp_path).convert('RGB')
        results['compressed_image'] = np.array(compressed_img)
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    # Calculate comparison metrics if both succeeded
    if results.get('jpeg2dct_success') and results.get('opencv_success'):
        dct1 = results['dct_jpeg2dct']
        dct2 = results['dct_opencv']
        
        # Ensure same shape
        min_h = min(dct1.shape[0], dct2.shape[0])
        min_w = min(dct1.shape[1], dct2.shape[1])
        dct1 = dct1[:min_h, :min_w]
        dct2 = dct2[:min_h, :min_w]
        
        diff = dct2 - dct1
        results['diff'] = diff
        results['mae'] = np.abs(diff).mean()
        results['rmse'] = np.sqrt((diff ** 2).mean())
        results['max_diff'] = np.abs(diff).max()
        
        # Correlation
        flat1 = dct1.flatten()
        flat2 = dct2.flatten()
        correlation = np.corrcoef(flat1, flat2)[0, 1]
        results['correlation'] = correlation
        
        print(f"    📊 MAE={results['mae']:.2f}, RMSE={results['rmse']:.2f}, Corr={correlation:.4f}")
    
    return results


def create_comprehensive_visualization(results, save_path):
    """
    Create comprehensive visualization comparing both methods.
    
    Args:
        results: Dictionary from extract_dct_both_methods()
        save_path: Path to save the visualization
    """
    has_both = results.get('jpeg2dct_success') and results.get('opencv_success')
    
    if has_both:
        # Full comparison with 3 rows, 4 columns
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    else:
        # Simplified view with 2 rows, 3 columns
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Row 1: Original and compressed images
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(results['original_image'])
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(results['compressed_image'])
    ax2.set_title(f'JPEG Compressed (QF={results["quality_factor"]})', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    if has_both:
        # jpeg2dct DCT visualization
        ax3 = fig.add_subplot(gs[0, 2])
        dct_jpeg2dct = results['dct_jpeg2dct']
        im3 = ax3.imshow(np.abs(dct_jpeg2dct), cmap='hot', vmin=0, vmax=100)
        ax3.set_title('jpeg2dct - DCT Magnitude', fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        # OpenCV DCT visualization
        ax4 = fig.add_subplot(gs[0, 3])
        dct_opencv = results['dct_opencv']
        im4 = ax4.imshow(np.abs(dct_opencv), cmap='hot', vmin=0, vmax=100)
        ax4.set_title('OpenCV - DCT Magnitude', fontsize=12, fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046)
        
        # Row 2: Detailed DCT comparisons (clipped to [0, 20] like ADCD-Net)
        ax5 = fig.add_subplot(gs[1, 0])
        dct_clipped_jpeg = np.clip(np.abs(dct_jpeg2dct), 0, 20)
        im5 = ax5.imshow(dct_clipped_jpeg, cmap='jet', vmin=0, vmax=20)
        ax5.set_title('jpeg2dct (Clipped 0-20)', fontsize=12, fontweight='bold')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046)
        
        ax6 = fig.add_subplot(gs[1, 1])
        dct_clipped_opencv = np.clip(np.abs(dct_opencv), 0, 20)
        im6 = ax6.imshow(dct_clipped_opencv, cmap='jet', vmin=0, vmax=20)
        ax6.set_title('OpenCV (Clipped 0-20)', fontsize=12, fontweight='bold')
        ax6.axis('off')
        plt.colorbar(im6, ax=ax6, fraction=0.046)
        
        # Difference visualization
        ax7 = fig.add_subplot(gs[1, 2])
        diff = results['diff']
        im7 = ax7.imshow(np.abs(diff), cmap='hot', vmin=0, vmax=50)
        ax7.set_title(f'Absolute Difference\nMAE={results["mae"]:.2f}', fontsize=12, fontweight='bold')
        ax7.axis('off')
        plt.colorbar(im7, ax=ax7, fraction=0.046)
        
        # Signed difference
        ax8 = fig.add_subplot(gs[1, 3])
        im8 = ax8.imshow(diff, cmap='RdBu_r', vmin=-50, vmax=50)
        ax8.set_title(f'Signed Difference\nRMSE={results["rmse"]:.2f}', fontsize=12, fontweight='bold')
        ax8.axis('off')
        plt.colorbar(im8, ax=ax8, fraction=0.046)
        
        # Row 3: Statistical comparisons
        ax9 = fig.add_subplot(gs[2, 0])
        ax9.hist(dct_jpeg2dct.flatten(), bins=100, alpha=0.5, label='jpeg2dct', 
                density=True, color='blue', range=(-100, 100))
        ax9.hist(dct_opencv.flatten(), bins=100, alpha=0.5, label='OpenCV', 
                density=True, color='red', range=(-100, 100))
        ax9.set_xlabel('DCT Coefficient Value', fontsize=10)
        ax9.set_ylabel('Density', fontsize=10)
        ax9.set_title('Value Distribution', fontsize=12, fontweight='bold')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        ax10 = fig.add_subplot(gs[2, 1])
        sample_size = min(5000, dct_jpeg2dct.size)
        indices = np.random.choice(dct_jpeg2dct.size, sample_size, replace=False)
        ax10.scatter(dct_jpeg2dct.flatten()[indices], dct_opencv.flatten()[indices], 
                    alpha=0.3, s=1, c='blue')
        ax10.plot([-200, 200], [-200, 200], 'r--', linewidth=2, label='y=x')
        ax10.set_xlabel('jpeg2dct', fontsize=10)
        ax10.set_ylabel('OpenCV', fontsize=10)
        ax10.set_title(f'Correlation = {results["correlation"]:.4f}', fontsize=12, fontweight='bold')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        ax10.set_xlim(-200, 200)
        ax10.set_ylim(-200, 200)
        
        ax11 = fig.add_subplot(gs[2, 2])
        ax11.hist(diff.flatten(), bins=100, density=True, color='purple', alpha=0.7)
        ax11.axvline(0, color='red', linestyle='--', linewidth=2)
        ax11.set_xlabel('Error (OpenCV - jpeg2dct)', fontsize=10)
        ax11.set_ylabel('Density', fontsize=10)
        ax11.set_title('Error Distribution', fontsize=12, fontweight='bold')
        ax11.grid(True, alpha=0.3)
        
        # Statistics table
        ax12 = fig.add_subplot(gs[2, 3])
        ax12.axis('off')
        stats_text = [
            f"{'Metric':<20} {'Value':>12}",
            "-" * 35,
            f"{'MAE':<20} {results['mae']:>12.4f}",
            f"{'RMSE':<20} {results['rmse']:>12.4f}",
            f"{'Max Abs Error':<20} {results['max_diff']:>12.4f}",
            f"{'Correlation':<20} {results['correlation']:>12.6f}",
            "",
            f"{'jpeg2dct Mean':<20} {dct_jpeg2dct.mean():>12.2f}",
            f"{'OpenCV Mean':<20} {dct_opencv.mean():>12.2f}",
            f"{'jpeg2dct Std':<20} {dct_jpeg2dct.std():>12.2f}",
            f"{'OpenCV Std':<20} {dct_opencv.std():>12.2f}",
            "",
            "Assessment:",
            f"  {'Correlation:':<15} {'Excellent' if results['correlation'] > 0.9 else 'Good' if results['correlation'] > 0.8 else 'Fair'}",
            f"  {'Error Level:':<15} {'Low' if results['mae'] < 10 else 'Medium' if results['mae'] < 20 else 'High'}",
        ]
        ax12.text(0.1, 0.5, '\n'.join(stats_text), fontsize=10, family='monospace',
                 verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
    else:
        # Simplified visualization (only OpenCV or only one method)
        if results.get('opencv_success'):
            dct = results['dct_opencv']
            method_name = 'OpenCV DCT Workaround'
        else:
            print("No DCT extraction succeeded!")
            plt.close(fig)
            return
        
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(np.abs(dct), cmap='hot', vmin=0, vmax=100)
        ax3.set_title(f'{method_name} - Full Range', fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        ax4 = fig.add_subplot(gs[1, 0])
        dct_clipped = np.clip(np.abs(dct), 0, 20)
        im4 = ax4.imshow(dct_clipped, cmap='jet', vmin=0, vmax=20)
        ax4.set_title('DCT (Clipped 0-20 for ADCD-Net)', fontsize=12, fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046)
        
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.hist(dct.flatten(), bins=100, color='blue', alpha=0.7, density=True)
        ax5.set_xlabel('DCT Coefficient Value', fontsize=10)
        ax5.set_ylabel('Density', fontsize=10)
        ax5.set_title('Value Distribution', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        stats_text = [
            f"DCT Statistics:",
            "-" * 30,
            f"Shape:     {dct.shape}",
            f"Mean:      {dct.mean():.2f}",
            f"Std:       {dct.std():.2f}",
            f"Min:       {dct.min():.2f}",
            f"Max:       {dct.max():.2f}",
            f"Range:     [{dct.min():.0f}, {dct.max():.0f}]",
            "",
            f"Quality:   {results['quality_factor']}",
            f"Method:    {method_name}",
        ]
        ax6.text(0.1, 0.5, '\n'.join(stats_text), fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Overall title
    img_name = os.path.basename(results['image_path'])
    title = f'DCT Extraction Comparison: {img_name} (QF={results["quality_factor"]})'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  💾 Saved: {save_path}")
    plt.close(fig)


def process_all_images():
    """Process all images in the input directory."""
    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    
    if not image_paths:
        print(f"\n⚠ No images found in {INPUT_DIR}")
        print(f"  Please add test images to: {os.path.abspath(INPUT_DIR)}")
        print(f"  Supported formats: JPG, PNG, BMP, TIFF")
        return
    
    print(f"\n🖼️  Found {len(image_paths)} image(s) to process")
    
    # Process each image with different quality factors
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        img_basename = os.path.splitext(img_name)[0]
        
        print(f"\n📷 Processing: {img_name}")
        
        for qf in QUALITY_FACTORS:
            print(f"  Quality Factor: {qf}")
            
            try:
                # Extract DCT with both methods
                results = extract_dct_both_methods(img_path, quality_factor=qf)
                
                # Create visualization
                output_filename = f"{img_basename}_qf{qf}_comparison.png"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                create_comprehensive_visualization(results, output_path)
                
            except Exception as e:
                print(f"  ✗ Error processing with QF={qf}: {e}")
                import traceback
                traceback.print_exc()


def create_summary_comparison(image_path):
    """
    Create a summary showing the same image at different quality factors.
    
    Args:
        image_path: Path to input image
    """
    img_name = os.path.basename(image_path)
    img_basename = os.path.splitext(img_name)[0]
    
    print(f"\n📊 Creating quality factor comparison for: {img_name}")
    
    # Extract DCT for all quality factors
    all_results = []
    for qf in QUALITY_FACTORS:
        try:
            results = extract_dct_both_methods(image_path, quality_factor=qf)
            all_results.append(results)
        except Exception as e:
            print(f"  ✗ Failed at QF={qf}: {e}")
    
    if not all_results:
        print("  ✗ No results to visualize")
        return
    
    # Create multi-QF comparison
    n_qf = len(all_results)
    fig, axes = plt.subplots(n_qf, 4, figsize=(20, 5*n_qf))
    if n_qf == 1:
        axes = axes.reshape(1, -1)
    
    for idx, results in enumerate(all_results):
        qf = results['quality_factor']
        has_both = results.get('jpeg2dct_success') and results.get('opencv_success')
        
        # Original image (only show once)
        if idx == 0:
            axes[idx, 0].imshow(results['original_image'])
            axes[idx, 0].set_title('Original Image', fontsize=10, fontweight='bold')
        else:
            axes[idx, 0].axis('off')
        
        # Compressed image
        axes[idx, 1].imshow(results['compressed_image'])
        axes[idx, 1].set_title(f'JPEG QF={qf}', fontsize=10, fontweight='bold')
        axes[idx, 1].axis('off')
        
        if has_both:
            # jpeg2dct
            dct_j = np.clip(np.abs(results['dct_jpeg2dct']), 0, 20)
            im1 = axes[idx, 2].imshow(dct_j, cmap='jet', vmin=0, vmax=20)
            axes[idx, 2].set_title(f'jpeg2dct (QF={qf})', fontsize=10, fontweight='bold')
            axes[idx, 2].axis('off')
            plt.colorbar(im1, ax=axes[idx, 2], fraction=0.046)
            
            # OpenCV
            dct_o = np.clip(np.abs(results['dct_opencv']), 0, 20)
            im2 = axes[idx, 3].imshow(dct_o, cmap='jet', vmin=0, vmax=20)
            axes[idx, 3].set_title(f'OpenCV (QF={qf})\nCorr={results["correlation"]:.4f}', 
                                   fontsize=10, fontweight='bold')
            axes[idx, 3].axis('off')
            plt.colorbar(im2, ax=axes[idx, 3], fraction=0.046)
        else:
            # Only OpenCV
            if results.get('opencv_success'):
                dct_o = np.clip(np.abs(results['dct_opencv']), 0, 20)
                im2 = axes[idx, 2].imshow(dct_o, cmap='jet', vmin=0, vmax=20)
                axes[idx, 2].set_title(f'OpenCV (QF={qf})', fontsize=10, fontweight='bold')
                axes[idx, 2].axis('off')
                plt.colorbar(im2, ax=axes[idx, 2], fraction=0.046)
                axes[idx, 3].axis('off')
    
    plt.suptitle(f'Quality Factor Comparison: {img_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, f"{img_basename}_qf_summary.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  💾 Saved: {output_path}")
    plt.close(fig)


def main():
    """Main execution function."""
    print("="*70)
    print("DCT EXTRACTION METHODS - VISUAL COMPARISON")
    print("="*70)
    
    # Ensure directories exist
    ensure_directories()
    
    # Check for images
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.PNG']:
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    
    if not image_paths:
        print("\n" + "="*70)
        print("⚠ NO IMAGES FOUND")
        print("="*70)
        print(f"\nPlease add test images to:")
        print(f"  {os.path.abspath(INPUT_DIR)}")
        print(f"\nSupported formats: JPG, PNG, BMP, TIFF")
        print(f"\nAfter adding images, run this script again:")
        print(f"  python visualize_dct_comparison.py")
        print("="*70)
        return
    
    # Process all images
    process_all_images()
    
    # Create quality factor summaries
    print("\n" + "-"*70)
    print("Creating Quality Factor Summaries")
    print("-"*70)
    for img_path in image_paths:
        create_summary_comparison(img_path)
    
    # Summary
    print("\n" + "="*70)
    print("✓ PROCESSING COMPLETE")
    print("="*70)
    print(f"\nResults saved to:")
    print(f"  {os.path.abspath(OUTPUT_DIR)}")
    print(f"\nGenerated files:")
    output_files = glob.glob(os.path.join(OUTPUT_DIR, '*.png'))
    for f in sorted(output_files):
        print(f"  - {os.path.basename(f)}")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
