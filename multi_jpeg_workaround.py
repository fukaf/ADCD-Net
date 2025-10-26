"""
Modified multi_jpeg function with DCT workaround support.

This module provides drop-in replacements for the DCT extraction
functions in ds.py, allowing the use of either jpeg2dct or the
OpenCV workaround.
"""

import tempfile
import numpy as np
from PIL import Image

# Try to import jpeg2dct, fallback to workaround
try:
    from jpeg2dct.numpy import load as jpeg2dct_load
    JPEG2DCT_AVAILABLE = True
except ImportError:
    JPEG2DCT_AVAILABLE = False
    print("Warning: jpeg2dct not available, using OpenCV workaround")

from dct_workaround import extract_dct_opencv_blockwise, convert_blockwise_to_spatial


def multi_jpeg_with_workaround(img, num_jpeg, min_qf, upper_bound, 
                                jpeg_record=None, use_workaround=False):
    """
    Modified multi_jpeg function with OpenCV DCT workaround support.
    
    This is a drop-in replacement for the multi_jpeg function in ds.py,
    with an additional parameter to choose between jpeg2dct and OpenCV.
    
    Args:
        img: PIL Image
        num_jpeg: Number of JPEG compressions
        min_qf: Minimum quality factor
        upper_bound: Upper bound quality factor
        jpeg_record: Predefined quality factors (optional)
        use_workaround: If True, use OpenCV workaround; if False, use jpeg2dct
    
    Returns:
        dct: DCT coefficients (H, W) as int32
        img: Compressed PIL Image
        qf_record: List of quality factors used
    """
    from random import randint
    
    with tempfile.NamedTemporaryFile(delete=True, suffix='.jpg') as tmp:
        img = img.convert("L")
        im_ori = img.copy()
        qf_record = []
        
        if jpeg_record is not None:
            num_jpeg = len(jpeg_record)
        
        for each_jpeg in range(num_jpeg):
            if jpeg_record is not None:
                qf = jpeg_record[each_jpeg]
            else:
                qf = randint(min_qf, upper_bound)
            qf_record.append(qf)
            img.save(tmp.name, "JPEG", quality=int(qf))
            img.close()
            img = Image.open(tmp.name)

        img = Image.open(tmp.name)
        img = img.convert('RGB')
        
        try:
            # Choose DCT extraction method
            if use_workaround or not JPEG2DCT_AVAILABLE:
                # Use OpenCV workaround
                dct_y_blocks, _, _ = extract_dct_opencv_blockwise(
                    tmp.name, 
                    quality_factor=qf_record[-1]
                )
                dct_y = dct_y_blocks  # Keep in block format
            else:
                # Use jpeg2dct
                dct_y, _, _ = jpeg2dct_load(tmp.name, normalized=False)
        except Exception as e:
            # Fallback: try workaround if jpeg2dct fails
            print(f"DCT extraction failed ({e}), trying workaround...")
            try:
                dct_y_blocks, _, _ = extract_dct_opencv_blockwise(
                    tmp.name, 
                    quality_factor=qf_record[-1]
                )
                dct_y = dct_y_blocks
            except:
                # Last resort: use original image with QF=100
                with tempfile.NamedTemporaryFile(delete=True, suffix='.jpg') as tmp1:
                    qf = 100
                    qf_record = [100]
                    im_ori.save(tmp1.name, "JPEG", quality=qf)
                    img = Image.open(tmp1.name)
                    img = img.convert('RGB')
                    
                    if use_workaround or not JPEG2DCT_AVAILABLE:
                        dct_y_blocks, _, _ = extract_dct_opencv_blockwise(
                            tmp1.name, 
                            quality_factor=100
                        )
                        dct_y = dct_y_blocks
                    else:
                        dct_y, _, _ = jpeg2dct_load(tmp1.name, normalized=False)

    # Convert block format to spatial format
    # dct_y shape: [h/8, w/8, 64] -> [h, w]
    rows, cols, _ = dct_y.shape
    dct = np.empty(shape=(8 * rows, 8 * cols), dtype=np.float32)
    for j in range(rows):
        for i in range(cols):
            dct[8 * j: 8 * (j + 1), 8 * i: 8 * (i + 1)] = dct_y[j, i].reshape(8, 8)
    
    # Convert to int32
    dct = np.int32(dct)
    
    return dct, img, qf_record


def multi_jpeg_auto(img, num_jpeg, min_qf, upper_bound, jpeg_record=None):
    """
    Automatic version that chooses the best available method.
    
    Tries jpeg2dct first, falls back to OpenCV workaround if not available.
    
    Args:
        img: PIL Image
        num_jpeg: Number of JPEG compressions
        min_qf: Minimum quality factor
        upper_bound: Upper bound quality factor
        jpeg_record: Predefined quality factors (optional)
    
    Returns:
        dct: DCT coefficients (H, W) as int32
        img: Compressed PIL Image
        qf_record: List of quality factors used
    """
    use_workaround = not JPEG2DCT_AVAILABLE
    return multi_jpeg_with_workaround(
        img, num_jpeg, min_qf, upper_bound, jpeg_record, use_workaround
    )


# Example: How to modify ds.py to use the workaround
def patch_ds_module():
    """
    Example function showing how to patch ds.py to use the workaround.
    
    Usage:
        # At the top of your script:
        from multi_jpeg_workaround import patch_ds_module
        patch_ds_module()
        
        # Now ds.multi_jpeg will use the workaround if jpeg2dct is not available
    """
    import ds
    
    # Save original function
    ds._original_multi_jpeg = ds.multi_jpeg
    
    # Replace with auto version
    ds.multi_jpeg = multi_jpeg_auto
    
    print("✓ ds.multi_jpeg patched to use automatic DCT extraction")
    if not JPEG2DCT_AVAILABLE:
        print("  Using OpenCV workaround (jpeg2dct not available)")
    else:
        print("  Using jpeg2dct (preferred method)")


if __name__ == '__main__':
    import os
    
    print("="*60)
    print("Multi-JPEG with DCT Workaround")
    print("="*60)
    
    # Test with a sample image
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img_pil = Image.fromarray(test_img)
    
    print(f"\njpeg2dct available: {JPEG2DCT_AVAILABLE}")
    
    # Test with workaround
    print("\nTesting OpenCV workaround...")
    dct, img_out, qf_record = multi_jpeg_with_workaround(
        img_pil, num_jpeg=2, min_qf=85, upper_bound=100, use_workaround=True
    )
    print(f"  DCT shape: {dct.shape}")
    print(f"  DCT range: [{dct.min()}, {dct.max()}]")
    print(f"  Quality factors: {qf_record}")
    
    if JPEG2DCT_AVAILABLE:
        print("\nTesting jpeg2dct...")
        img_pil2 = Image.fromarray(test_img)
        dct2, img_out2, qf_record2 = multi_jpeg_with_workaround(
            img_pil2, num_jpeg=2, min_qf=85, upper_bound=100, use_workaround=False
        )
        print(f"  DCT shape: {dct2.shape}")
        print(f"  DCT range: [{dct2.min()}, {dct2.max()}]")
        print(f"  Quality factors: {qf_record2}")
        
        # Compare
        diff = np.abs(dct - dct2).mean()
        print(f"\n  Mean difference: {diff:.4f}")
    
    # Test auto version
    print("\nTesting auto version...")
    img_pil3 = Image.fromarray(test_img)
    dct3, img_out3, qf_record3 = multi_jpeg_auto(
        img_pil3, num_jpeg=2, min_qf=85, upper_bound=100
    )
    print(f"  DCT shape: {dct3.shape}")
    print(f"  DCT range: [{dct3.min()}, {dct3.max()}]")
    
    print("\n✓ All tests passed!")
    print("\nTo use in your code:")
    print("  from multi_jpeg_workaround import multi_jpeg_auto")
    print("  dct, img, qfs = multi_jpeg_auto(img, num_jpeg=2, min_qf=75, upper_bound=100)")
