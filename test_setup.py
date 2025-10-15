"""
Test script to verify ADCD-Net inference setup.

This script checks:
1. Required packages are installed
2. Model checkpoints can be loaded
3. Inference pipeline works correctly
4. GPU/CUDA is available (if requested)

Usage:
    python test_setup.py \
        --model path/to/ADCD-Net.pth \
        --docres path/to/docres.pkl \
        --qt-table path/to/qt_table.pk \
        [--craft path/to/craft.pth] \
        [--test-image path/to/image.jpg]
"""

import argparse
import sys
import os


def check_imports():
    """Check if all required packages are installed."""
    print("\n" + "="*60)
    print("1. Checking Python Package Dependencies")
    print("="*60)
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('matplotlib', 'Matplotlib'),
        ('jpeg2dct', 'JPEG2DCT'),
        ('albumentations', 'Albumentations'),
    ]
    
    missing = []
    
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"  ✓ {package_name} ({module_name})")
        except ImportError:
            print(f"  ✗ {package_name} ({module_name}) - NOT INSTALLED")
            missing.append(package_name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("   Install with: pip install -r requirements_inference.txt")
        return False
    else:
        print("\n✓ All required packages are installed")
        return True


def check_cuda():
    """Check CUDA availability."""
    print("\n" + "="*60)
    print("2. Checking CUDA/GPU Availability")
    print("="*60)
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print("\n✓ GPU is available for inference")
            return True
        else:
            print("\n⚠ No GPU available, will use CPU (slower)")
            return False
    except Exception as e:
        print(f"\n✗ Error checking CUDA: {e}")
        return False


def check_files(model_path, docres_path, qt_table_path, craft_path=None, test_image=None):
    """Check if required files exist."""
    print("\n" + "="*60)
    print("3. Checking Required Files")
    print("="*60)
    
    files_to_check = [
        (model_path, "ADCD-Net checkpoint"),
        (docres_path, "DocRes checkpoint"),
        (qt_table_path, "Quantization table"),
    ]
    
    if craft_path:
        files_to_check.append((craft_path, "CRAFT checkpoint"))
    
    if test_image:
        files_to_check.append((test_image, "Test image"))
    
    all_exist = True
    
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  ✓ {description}: {file_path} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {description}: {file_path} - NOT FOUND")
            all_exist = False
    
    if all_exist:
        print("\n✓ All required files are present")
    else:
        print("\n✗ Some required files are missing")
    
    return all_exist


def test_model_loading(model_path, docres_path, qt_table_path, device='cuda'):
    """Test loading the model."""
    print("\n" + "="*60)
    print("4. Testing Model Loading")
    print("="*60)
    
    try:
        from inference import SingleImageInference
        import torch
        
        # Force CPU if CUDA not available
        if device == 'cuda' and not torch.cuda.is_available():
            print("  ⚠ CUDA requested but not available, using CPU")
            device = 'cpu'
        
        print(f"  Loading model on {device}...")
        inferencer = SingleImageInference(
            model_ckpt_path=model_path,
            docres_ckpt_path=docres_path,
            qt_table_path=qt_table_path,
            craft_ckpt_path=None,
            device=device
        )
        
        print(f"  ✓ Model loaded successfully")
        print(f"  Model device: {next(inferencer.model.parameters()).device}")
        
        # Count parameters
        total_params = sum(p.numel() for p in inferencer.model.parameters())
        trainable_params = sum(p.numel() for p in inferencer.model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        print("\n✓ Model loading test passed")
        return True, inferencer
        
    except Exception as e:
        print(f"\n✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_inference(inferencer, test_image=None, device='cuda'):
    """Test running inference on a dummy or real image."""
    print("\n" + "="*60)
    print("5. Testing Inference Pipeline")
    print("="*60)
    
    try:
        import numpy as np
        from PIL import Image
        import tempfile
        
        # Create or load test image
        if test_image and os.path.exists(test_image):
            print(f"  Using test image: {test_image}")
            img_path = test_image
        else:
            print("  Creating dummy test image...")
            # Create a simple test image
            img = Image.new('RGB', (512, 512), color=(255, 255, 255))
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                img.save(tmp.name)
                img_path = tmp.name
        
        print("  Running inference...")
        results = inferencer.predict(img_path, jpeg_quality=100)
        
        # Validate results
        assert 'pred_mask' in results, "Missing pred_mask in results"
        assert 'pred_prob' in results, "Missing pred_prob in results"
        assert 'align_score' in results, "Missing align_score in results"
        
        print(f"  ✓ Inference completed successfully")
        print(f"  Output shape: {results['pred_mask'].shape}")
        print(f"  Forgery pixels: {(results['pred_mask'] == 1).sum()}")
        print(f"  Alignment score: {results['align_score'][1]:.4f}")
        
        # Clean up dummy image if created
        if not test_image or not os.path.exists(test_image):
            os.unlink(img_path)
        
        print("\n✓ Inference test passed")
        return True
        
    except Exception as e:
        print(f"\n✗ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test ADCD-Net inference setup')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to ADCD-Net checkpoint')
    parser.add_argument('--docres', type=str, required=True,
                       help='Path to DocRes checkpoint')
    parser.add_argument('--qt-table', type=str, required=True,
                       help='Path to quantization table pickle')
    parser.add_argument('--craft', type=str, default=None,
                       help='Path to CRAFT checkpoint (optional)')
    parser.add_argument('--test-image', type=str, default=None,
                       help='Path to test image (optional, will create dummy if not provided)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to test on')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("ADCD-Net Inference Setup Test")
    print("="*60)
    
    # Run all checks
    results = []
    
    # 1. Check imports
    results.append(("Package dependencies", check_imports()))
    
    # 2. Check CUDA
    cuda_available = check_cuda()
    if args.device == 'cuda' and not cuda_available:
        print("\n⚠ Warning: CUDA requested but not available, will test with CPU")
        args.device = 'cpu'
    results.append(("CUDA availability", cuda_available))
    
    # 3. Check files
    results.append(("Required files", check_files(
        args.model, args.docres, args.qt_table, args.craft, args.test_image
    )))
    
    # If any check failed so far, stop
    if not all(r[1] for r in results):
        print("\n" + "="*60)
        print("SETUP TEST FAILED")
        print("="*60)
        print("Please fix the issues above before running inference.")
        return 1
    
    # 4. Test model loading
    load_success, inferencer = test_model_loading(
        args.model, args.docres, args.qt_table, args.device
    )
    results.append(("Model loading", load_success))
    
    if not load_success:
        print("\n" + "="*60)
        print("SETUP TEST FAILED")
        print("="*60)
        return 1
    
    # 5. Test inference
    results.append(("Inference pipeline", test_inference(
        inferencer, args.test_image, args.device
    )))
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 All tests passed! Your setup is ready for inference.")
        print("\nYou can now run:")
        print("  - python inference.py --image <your_image.jpg> ...")
        print("  - python example_inference.py")
        print("  - python batch_inference.py --input-dir <dir> ...")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
