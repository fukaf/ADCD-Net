"""
DCT Extraction Workaround using OpenCV

This module provides an alternative to jpeg2dct library for extracting
DCT coefficients from JPEG images using cv2.dct().

The workaround approximates JPEG DCT extraction by:
1. Converting image to YCbCr color space
2. Dividing into 8×8 blocks
3. Applying DCT to each block
4. Simulating quantization

Note: This approximation may differ from true JPEG DCT coefficients
extracted directly from the compressed file.
"""

import numpy as np
import cv2
from PIL import Image


def extract_dct_opencv(image_path_or_array, quality_factor=None):
    """
    Extract DCT coefficients using OpenCV as a workaround for jpeg2dct.
    
    This function approximates JPEG DCT extraction by:
    - Loading the image
    - Converting to YCbCr
    - Applying DCT to 8×8 blocks
    - Optionally applying quantization simulation
    
    Args:
        image_path_or_array: Path to image file or numpy array (RGB)
        quality_factor: JPEG quality factor for quantization simulation (optional)
    
    Returns:
        dct_y: DCT coefficients for Y channel (H, W) matching jpeg2dct format
        dct_cb: DCT coefficients for Cb channel (H/2, W/2) or None
        dct_cr: DCT coefficients for Cr channel (H/2, W/2) or None
    """
    # Load image
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_path_or_array, Image.Image):
        img = np.array(image_path_or_array)
    else:
        img = image_path_or_array
    
    # Convert to YCbCr
    img_ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y_channel = img_ycbcr[:, :, 0].astype(np.float32)
    
    # Get dimensions (must be divisible by 8)
    h, w = y_channel.shape
    h_blocks = h // 8
    w_blocks = w // 8
    
    # Crop to make divisible by 8
    y_channel = y_channel[:h_blocks*8, :w_blocks*8]
    
    # Extract DCT coefficients for each 8×8 block
    dct_y = np.zeros((h_blocks*8, w_blocks*8), dtype=np.float32)
    
    for i in range(h_blocks):
        for j in range(w_blocks):
            # Extract 8×8 block
            block = y_channel[i*8:(i+1)*8, j*8:(j+1)*8]
            
            # Center around 128 (JPEG standard)
            block = block - 128.0
            
            # Apply DCT
            dct_block = cv2.dct(block)
            
            # Optional: Apply quantization if quality factor provided
            if quality_factor is not None:
                qt = get_quantization_table(quality_factor)
                dct_block = np.round(dct_block / qt) * qt
            
            # Store back
            dct_y[i*8:(i+1)*8, j*8:(j+1)*8] = dct_block
    
    # Convert to int32 to match jpeg2dct output format
    dct_y = np.int32(dct_y)
    
    return dct_y, None, None


def extract_dct_opencv_blockwise(image_path_or_array, quality_factor=None):
    """
    Extract DCT coefficients in block format (h/8, w/8, 64) matching jpeg2dct.
    
    This matches the exact output format of jpeg2dct.numpy.load():
    - Returns coefficients in blocks of 64 values (8×8 flattened)
    - Shape: (num_blocks_h, num_blocks_w, 64)
    
    Args:
        image_path_or_array: Path to image file or numpy array (RGB)
        quality_factor: JPEG quality factor for quantization simulation (optional)
    
    Returns:
        dct_y: DCT coefficients (h/8, w/8, 64) matching jpeg2dct format
        dct_cb: None (not implemented)
        dct_cr: None (not implemented)
    """
    # Load image
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_path_or_array, Image.Image):
        img = np.array(image_path_or_array)
    else:
        img = image_path_or_array
    
    # Convert to YCbCr
    img_ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y_channel = img_ycbcr[:, :, 0].astype(np.float32)
    
    # Get dimensions
    h, w = y_channel.shape
    h_blocks = h // 8
    w_blocks = w // 8
    
    # Crop to make divisible by 8
    y_channel = y_channel[:h_blocks*8, :w_blocks*8]
    
    # Initialize output in block format
    dct_blocks = np.zeros((h_blocks, w_blocks, 64), dtype=np.float32)
    
    # Get quantization table if needed
    qt = None
    if quality_factor is not None:
        qt = get_quantization_table(quality_factor)
    
    for i in range(h_blocks):
        for j in range(w_blocks):
            # Extract 8×8 block
            block = y_channel[i*8:(i+1)*8, j*8:(j+1)*8]
            
            # Center around 128
            block = block - 128.0
            
            # Apply DCT
            dct_block = cv2.dct(block)
            
            # Apply quantization if provided
            if qt is not None:
                dct_block = np.round(dct_block / qt)
            
            # Flatten and store
            dct_blocks[i, j, :] = dct_block.flatten()
    
    return dct_blocks, None, None


def get_quantization_table(quality_factor):
    """
    Get standard JPEG quantization table for given quality factor.
    
    Args:
        quality_factor: JPEG quality (1-100)
    
    Returns:
        8×8 quantization table
    """
    # Standard JPEG luminance quantization table (quality 50)
    base_table = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float32)
    
    # Scale based on quality factor
    if quality_factor < 50:
        scale = 5000.0 / quality_factor
    else:
        scale = 200.0 - 2.0 * quality_factor
    
    qt = np.floor((base_table * scale + 50.0) / 100.0)
    qt = np.clip(qt, 1, 255)
    
    return qt


def convert_blockwise_to_spatial(dct_blocks):
    """
    Convert block format (h/8, w/8, 64) to spatial format (h, w).
    
    This matches the conversion done in ds.py multi_jpeg() function.
    
    Args:
        dct_blocks: DCT coefficients in block format (rows, cols, 64)
    
    Returns:
        dct_spatial: DCT coefficients in spatial format (8*rows, 8*cols)
    """
    rows, cols, _ = dct_blocks.shape
    dct_spatial = np.empty((8 * rows, 8 * cols), dtype=dct_blocks.dtype)
    
    for j in range(rows):
        for i in range(cols):
            dct_spatial[8*j:8*(j+1), 8*i:8*(i+1)] = dct_blocks[j, i].reshape(8, 8)
    
    return dct_spatial


def extract_dct_for_adcdnet(image_path_or_array, quality_factor=None):
    """
    Extract DCT coefficients in the format expected by ADCD-Net.
    
    This is a drop-in replacement for jpeg2dct.numpy.load() in the context
    of ADCD-Net's data pipeline.
    
    Args:
        image_path_or_array: Path to image file or numpy array
        quality_factor: JPEG quality factor (optional)
    
    Returns:
        dct_spatial: DCT coefficients in spatial format (H, W) as int32
    """
    # Extract in block format
    dct_blocks, _, _ = extract_dct_opencv_blockwise(image_path_or_array, quality_factor)
    
    # Convert to spatial format
    dct_spatial = convert_blockwise_to_spatial(dct_blocks)
    
    # Convert to int32 to match jpeg2dct output
    dct_spatial = np.int32(dct_spatial)
    
    return dct_spatial


# Convenience function matching jpeg2dct API
def load_opencv(file_path, normalized=False, quality_factor=None):
    """
    Load DCT coefficients using OpenCV (mimics jpeg2dct.numpy.load API).
    
    Args:
        file_path: Path to image file
        normalized: Not used (kept for API compatibility)
        quality_factor: JPEG quality factor for quantization
    
    Returns:
        dct_y: Y channel DCT coefficients in block format (h/8, w/8, 64)
        dct_cb: None
        dct_cr: None
    """
    return extract_dct_opencv_blockwise(file_path, quality_factor)


if __name__ == '__main__':
    # Test the workaround
    import tempfile
    
    print("Testing DCT extraction workaround...")
    
    # Create a test image
    test_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Save as JPEG
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img = Image.fromarray(test_img)
        img.save(tmp.name, 'JPEG', quality=90)
        
        # Extract DCT
        dct_y, _, _ = extract_dct_opencv_blockwise(tmp.name, quality_factor=90)
        print(f"DCT blocks shape: {dct_y.shape}")
        
        # Convert to spatial
        dct_spatial = convert_blockwise_to_spatial(dct_y)
        print(f"DCT spatial shape: {dct_spatial.shape}")
        
        # Test drop-in replacement
        dct_adcdnet = extract_dct_for_adcdnet(tmp.name, quality_factor=90)
        print(f"ADCD-Net format shape: {dct_adcdnet.shape}")
        print(f"DCT value range: [{dct_adcdnet.min()}, {dct_adcdnet.max()}]")
        
        import os
        os.unlink(tmp.name)
    
    print("\n✓ DCT extraction workaround test passed!")
