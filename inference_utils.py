"""
Utility functions for inference and visualization.

This module provides helper functions for working with inference results.
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def create_overlay(image, mask, color=(255, 0, 0), alpha=0.4):
    """
    Create an overlay visualization of forgery mask on image.
    
    Args:
        image: RGB image as numpy array (H, W, 3)
        mask: Binary mask as numpy array (H, W), where 1 indicates forgery
        color: RGB color tuple for forgery regions
        alpha: Transparency of overlay (0-1)
    
    Returns:
        Overlay image as numpy array
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    overlay = image.copy()
    mask_colored = np.zeros_like(overlay)
    mask_colored[mask == 1] = color
    
    overlay = cv2.addWeighted(overlay, 1-alpha, mask_colored, alpha, 0)
    return overlay


def visualize_comparison(original_img, pred_mask, gt_mask=None, save_path=None):
    """
    Visualize comparison between prediction and ground truth.
    
    Args:
        original_img: Original RGB image
        pred_mask: Predicted forgery mask
        gt_mask: Ground truth mask (optional)
        save_path: Path to save visualization
    """
    if gt_mask is not None:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(gt_mask, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        # Show differences
        correct = (pred_mask == gt_mask).astype(np.uint8)
        diff_vis = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        diff_vis[correct == 1] = [0, 255, 0]  # Green for correct
        diff_vis[(pred_mask == 1) & (gt_mask == 0)] = [255, 0, 0]  # Red for false positive
        diff_vis[(pred_mask == 0) & (gt_mask == 1)] = [0, 0, 255]  # Blue for false negative
        
        axes[3].imshow(diff_vis)
        axes[3].set_title('Difference (Green=Correct, Red=FP, Blue=FN)')
        axes[3].axis('off')
        
        # Calculate metrics
        tp = ((pred_mask == 1) & (gt_mask == 1)).sum()
        fp = ((pred_mask == 1) & (gt_mask == 0)).sum()
        fn = ((pred_mask == 0) & (gt_mask == 1)).sum()
        tn = ((pred_mask == 0) & (gt_mask == 0)).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        
        title = f'Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}'
        plt.suptitle(title, fontsize=14, fontweight='bold')
        
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(pred_mask, cmap='hot')
        axes[1].set_title('Predicted Forgery Mask')
        axes[1].axis('off')
        
        overlay = create_overlay(original_img, pred_mask)
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_mask_as_image(mask, save_path, colormap='hot'):
    """
    Save binary mask as a colored image.
    
    Args:
        mask: Binary mask (H, W)
        save_path: Path to save image
        colormap: Matplotlib colormap name
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap=colormap)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()


def calculate_metrics(pred_mask, gt_mask):
    """
    Calculate evaluation metrics.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
    
    Returns:
        Dictionary with metrics
    """
    tp = ((pred_mask == 1) & (gt_mask == 1)).sum()
    fp = ((pred_mask == 1) & (gt_mask == 0)).sum()
    fn = ((pred_mask == 0) & (gt_mask == 1)).sum()
    tn = ((pred_mask == 0) & (gt_mask == 0)).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'iou': float(iou),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }


def apply_morphology(mask, operation='close', kernel_size=5):
    """
    Apply morphological operations to clean up mask.
    
    Args:
        mask: Binary mask
        operation: 'close', 'open', 'dilate', 'erode'
        kernel_size: Size of morphological kernel
    
    Returns:
        Processed mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if operation == 'close':
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    elif operation == 'open':
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    elif operation == 'dilate':
        mask = cv2.dilate(mask.astype(np.uint8), kernel)
    elif operation == 'erode':
        mask = cv2.erode(mask.astype(np.uint8), kernel)
    
    return mask


def get_forgery_statistics(mask, image=None):
    """
    Get statistics about forgery regions.
    
    Args:
        mask: Binary forgery mask
        image: Optional RGB image for per-region analysis
    
    Returns:
        Dictionary with statistics
    """
    # Find connected components
    num_regions, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    
    # Remove background (label 0)
    num_regions -= 1
    
    if num_regions == 0:
        return {
            'num_regions': 0,
            'total_pixels': 0,
            'mean_region_size': 0,
            'largest_region_size': 0,
            'smallest_region_size': 0
        }
    
    # Calculate statistics
    region_sizes = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_regions + 1)]
    
    stats_dict = {
        'num_regions': num_regions,
        'total_pixels': int(mask.sum()),
        'mean_region_size': float(np.mean(region_sizes)),
        'largest_region_size': int(np.max(region_sizes)),
        'smallest_region_size': int(np.min(region_sizes)),
        'region_sizes': region_sizes,
        'centroids': centroids[1:].tolist()  # Exclude background
    }
    
    return stats_dict


def draw_bboxes_on_image(image, mask, min_area=100):
    """
    Draw bounding boxes around forgery regions.
    
    Args:
        image: RGB image
        mask: Binary forgery mask
        min_area: Minimum area to draw bbox
    
    Returns:
        Image with bounding boxes drawn
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    output = image.copy()
    
    # Find contours
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Add area label
            cv2.putText(output, f'{area:.0f}px', (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return output


if __name__ == '__main__':
    # Example usage
    print("Utility functions for ADCD-Net inference")
    print("\nAvailable functions:")
    print("  - create_overlay: Create forgery overlay on image")
    print("  - visualize_comparison: Compare prediction with ground truth")
    print("  - save_mask_as_image: Save mask as colored image")
    print("  - calculate_metrics: Calculate evaluation metrics")
    print("  - apply_morphology: Apply morphological operations")
    print("  - get_forgery_statistics: Get statistics about forgery regions")
    print("  - draw_bboxes_on_image: Draw bounding boxes around forgeries")
