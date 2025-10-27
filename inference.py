"""
Single Image Inference Script for ADCD-Net
This script performs document forgery localization on a single image.
"""

import os
import sys
import argparse
import pickle
import tempfile
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T

# Import model and utilities
from model.model import ADCDNet

# Try to import jpeg2dct, fall back to workaround
try:
    from jpeg2dct.numpy import load as dct_load
    DCT_METHOD = 'jpeg2dct'
    print("Using jpeg2dct library for DCT extraction")
except ImportError:
    print("jpeg2dct not available, using OpenCV workaround")
    from dct_workaround import extract_dct_for_adcdnet
    DCT_METHOD = 'opencv'


class SingleImageInference:
    def __init__(self, 
                 model_ckpt_path,
                 qt_table_path,
                 docres_ckpt_path=None,
                 craft_ckpt_path=None,
                 device='cuda',
                 temp_dir='./temp_inference'):
        """
        Initialize the inference pipeline.
        
        Args:
            model_ckpt_path: Path to ADCD-Net checkpoint
            qt_table_path: Path to JPEG quantization table pickle
            docres_ckpt_path: Path to DocRes checkpoint (optional, not needed if using trained ADCD-Net)
            craft_ckpt_path: Path to CRAFT OCR model (optional, for OCR mask generation)
            device: 'cuda' or 'cpu'
            temp_dir: Directory for temporary files (default: './temp_inference')
        """
        self.device = device
        self.model = None
        self.qt_tables = None
        self.craft_model = None
        
        # Create temp directory
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using temp directory: {self.temp_dir.absolute()}")
        
        # Load quantization tables
        self.load_qt_tables(qt_table_path)
        
        # Load model
        self.load_model(model_ckpt_path, docres_ckpt_path)
        
        # Load CRAFT if path provided
        if craft_ckpt_path and os.path.exists(craft_ckpt_path):
            self.load_craft(craft_ckpt_path)
        
        # Image preprocessing transforms
        self.img_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.455, 0.406),
                       std=(0.229, 0.224, 0.225))
        ])
        
    def load_qt_tables(self, qt_path):
        """Load JPEG quantization tables."""
        print(f"Loading quantization tables from {qt_path}")
        with open(qt_path, 'rb') as f:
            qt_dict = pickle.load(f)
        self.qt_tables = {}
        for k, v in qt_dict.items():
            self.qt_tables[k] = torch.LongTensor(v)
        print(f"Loaded {len(self.qt_tables)} quantization tables")
    
    def load_model(self, model_ckpt_path, docres_ckpt_path):
        """Load ADCD-Net model."""
        print(f"Loading ADCD-Net model from {model_ckpt_path}")
        
        # Temporarily set cfg values needed by model
        import cfg
        original_docres = getattr(cfg, 'docres_ckpt_path', None)
        cfg.docres_ckpt_path = docres_ckpt_path
        
        # Initialize model
        self.model = ADCDNet()
        
        # Load checkpoint
        ckpt = torch.load(model_ckpt_path, map_location='cpu')
        if 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
        
        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        # Load the trained ADCD-Net weights
        missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"Warning: Missing keys in checkpoint: {missing}")
        if unexpected:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Restore cfg
        cfg.docres_ckpt_path = original_docres
        print("Model loaded successfully")
    
    def load_craft(self, craft_ckpt_path):
        """Load CRAFT OCR model for character segmentation."""
        try:
            from seg_char.CRAFT_pytorch.craft import CRAFT
            from seg_char.CRAFT_pytorch.test import copyStateDict
            
            print(f"Loading CRAFT model from {craft_ckpt_path}")
            self.craft_model = CRAFT()
            self.craft_model.load_state_dict(copyStateDict(torch.load(craft_ckpt_path)))
            self.craft_model = self.craft_model.to(self.device)
            self.craft_model.eval()
            print("CRAFT model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load CRAFT model: {e}")
            self.craft_model = None
    
    def extract_ocr_mask(self, img, ocr_bbox_path=None):
        """
        Extract OCR mask from image or load from file.
        
        Args:
            img: PIL Image or numpy array
            ocr_bbox_path: Path to precomputed OCR bbox pickle file
            
        Returns:
            OCR mask as numpy array (H, W)
        """
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img
        
        h, w = img_np.shape[:2]
        
        # If bbox file provided, load it
        if ocr_bbox_path and os.path.exists(ocr_bbox_path):
            print(f"Loading OCR bboxes from {ocr_bbox_path}")
            with open(ocr_bbox_path, 'rb') as f:
                bboxes = pickle.load(f)
            return self.bbox_to_mask(bboxes, h, w)
        
        # Try to use CRAFT model
        if self.craft_model is not None:
            print("Generating OCR mask using CRAFT...")
            return self.generate_ocr_mask_with_craft(img_np)
        
        # Return empty mask if no OCR available
        print("Warning: No OCR mask available, using empty mask")
        return np.zeros((h, w), dtype=np.uint8)
    
    def generate_ocr_mask_with_craft(self, img):
        """Generate OCR mask using CRAFT model."""
        from seg_char.CRAFT_pytorch import imgproc
        from torch.autograd import Variable
        
        # Resize image
        resize_img, tgt_ratio, _ = imgproc.resize_aspect_ratio(
            img=img, square_size=280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5
        )
        ratio = 1 / tgt_ratio * 2
        
        # Preprocess
        x = imgproc.normalizeMeanVariance(resize_img)
        x = torch.from_numpy(x).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0)).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            pred, _ = self.craft_model(x)
        
        score_map = pred[0, :, :, 0].cpu().data.numpy()
        bin_map = cv2.threshold(score_map, 0.6, 1, 0)[1]
        bin_map = (bin_map * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(bin_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract bboxes
        bboxes = []
        for contour in contours:
            contour = contour * ratio
            contour = contour.astype(np.int32)
            x, y, w, h = cv2.boundingRect(contour)
            square_side = max(w, h)
            bbox = (
                int(x - square_side / 3),
                int(y - square_side * (1/8) - square_side / 3),
                int(x + square_side + square_side / 3),
                int(y + square_side * (7/8) + square_side / 3)
            )
            bboxes.append(bbox)
        
        h, w = img.shape[:2]
        return self.bbox_to_mask(bboxes, h, w)
    
    def bbox_to_mask(self, bboxes, h, w, expand_ratio=0.1):
        """Convert bounding boxes to binary mask."""
        mask = np.zeros((h, w), dtype=np.uint8)
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            w_box = x2 - x1
            h_box = y2 - y1
            x1 = int(max(0, x1 - w_box * expand_ratio))
            y1 = int(max(0, y1 - h_box * expand_ratio))
            x2 = int(min(w, x2 + w_box * expand_ratio))
            y2 = int(min(h, y2 + h_box * expand_ratio))
            mask[y1:y2, x1:x2] = 1
        return mask
    
    def apply_jpeg_compression(self, img, quality=100):
        """
        Apply JPEG compression to image and extract DCT coefficients.
        
        Args:
            img: PIL Image
            quality: JPEG quality factor (1-100)
            
        Returns:
            compressed_img: PIL Image after JPEG compression
            dct: DCT coefficients as numpy array (H, W)
            qf: quality factor used
        """
        # Use designated temp directory instead of system temp
        import uuid
        temp_filename = self.temp_dir / f"temp_{uuid.uuid4().hex}.jpg"
        
        try:
            # Convert to grayscale for DCT extraction
            img_gray = img.convert("L")
            
            # Save to our temp directory (with write permissions)
            img_gray.save(str(temp_filename), "JPEG", quality=quality)
            
            # Load compressed image
            compressed_img = Image.open(temp_filename)
            compressed_img = compressed_img.convert('RGB')
            
            # Extract DCT coefficients based on available method
            try:
                if DCT_METHOD == 'jpeg2dct':
                    # Use jpeg2dct library
                    dct_y, _, _ = dct_load(str(temp_filename), normalized=False)
                    
                    # Convert DCT from (h, w, 64) to (8h, 8w)
                    rows, cols, _ = dct_y.shape
                    dct = np.empty((8 * rows, 8 * cols), dtype=np.int32)
                    for j in range(rows):
                        for i in range(cols):
                            dct[8*j:8*(j+1), 8*i:8*(i+1)] = dct_y[j, i].reshape(8, 8)
                else:
                    # Use OpenCV workaround
                    dct = extract_dct_for_adcdnet(str(temp_filename), quality_factor=quality)
                    
            except Exception as e:
                # Fallback if DCT extraction fails
                print(f"Warning: DCT extraction failed ({e}), using dummy DCT")
                h, w = np.array(img).shape[:2]
                dct = np.zeros((h, w), dtype=np.int32)
        
        finally:
            # Clean up temp file
            if temp_filename.exists():
                try:
                    temp_filename.unlink()  # Delete the temp file
                except Exception as e:
                    print(f"Warning: Could not delete temp file {temp_filename}: {e}")
        
        return compressed_img, dct, quality
    
    def preprocess_image(self, img_path, jpeg_quality=100, ocr_bbox_path=None):
        """
        Preprocess image for inference.
        
        Args:
            img_path: Path to input image
            jpeg_quality: JPEG quality for compression (1-100)
            ocr_bbox_path: Path to OCR bbox pickle file (optional)
            
        Returns:
            Dictionary containing preprocessed tensors
        """
        # Load image
        img = Image.open(img_path).convert('RGB')
        original_img = np.array(img)
        
        # Apply JPEG compression and extract DCT
        compressed_img, dct, qf = self.apply_jpeg_compression(img, jpeg_quality)
        
        # Get quantization table
        if qf not in self.qt_tables:
            print(f"Warning: QF {qf} not in quantization tables, using QF=100")
            qf = 100
        qt = self.qt_tables[qf]
        
        # Get OCR mask
        ocr_mask = self.extract_ocr_mask(compressed_img, ocr_bbox_path)
        
        # Ensure dimensions match
        img_np = np.array(compressed_img)
        h, w = img_np.shape[:2]
        if dct.shape[0] != h or dct.shape[1] != w:
            # Resize DCT to match image size
            dct = cv2.resize(dct.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            dct = dct.astype(np.int32)
        
        if ocr_mask.shape[0] != h or ocr_mask.shape[1] != w:
            ocr_mask = cv2.resize(ocr_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Transform to tensors
        img_tensor = self.img_transform(compressed_img)
        dct_tensor = torch.from_numpy(np.clip(np.abs(dct), 0, 20)).float()
        ocr_mask_tensor = torch.from_numpy(ocr_mask).long().unsqueeze(0)
        
        return {
            'img': img_tensor.unsqueeze(0),  # Add batch dimension
            'dct': dct_tensor.unsqueeze(0),
            'qt': qt.unsqueeze(0).unsqueeze(1),
            'ocr_mask': ocr_mask_tensor.unsqueeze(0),
            'original_img': original_img,
            'compressed_img': np.array(compressed_img),
            'ocr_mask_vis': ocr_mask
        }
    
    @torch.no_grad()
    def predict(self, img_path, jpeg_quality=100, ocr_bbox_path=None):
        """
        Run inference on a single image.
        
        Args:
            img_path: Path to input image
            jpeg_quality: JPEG quality for compression (1-100)
            ocr_bbox_path: Path to OCR bbox pickle file (optional)
            
        Returns:
            Dictionary containing prediction results
        """
        # Preprocess
        print(f"Preprocessing image: {img_path}")
        data = self.preprocess_image(img_path, jpeg_quality, ocr_bbox_path)
        
        # Move to device
        img = data['img'].to(self.device)
        dct = data['dct'].to(self.device)
        qt = data['qt'].to(self.device)
        ocr_mask = data['ocr_mask'].to(self.device)
        
        # Dummy mask for inference
        dummy_mask = torch.zeros_like(ocr_mask)
        
        # Forward pass
        print("Running inference...")
        logits, loc_feat, align_logits, _, _ = self.model(
            img, dct, qt, dummy_mask, ocr_mask, is_train=False
        )
        
        # Get prediction
        pred_mask = logits.argmax(1).cpu().numpy()[0]  # (H, W)
        pred_prob = F.softmax(logits, dim=1).cpu().numpy()[0]  # (C, H, W)
        
        # Get alignment score
        align_score = F.softmax(align_logits, dim=1).cpu().numpy()[0]
        
        return {
            'pred_mask': pred_mask,
            'pred_prob': pred_prob,
            'align_score': align_score,
            'original_img': data['original_img'],
            'compressed_img': data['compressed_img'],
            'ocr_mask': data['ocr_mask_vis']
        }
    
    def visualize_results(self, results, save_path=None):
        """
        Visualize prediction results.
        
        Args:
            results: Dictionary from predict()
            save_path: Path to save visualization (optional)
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(results['original_img'])
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Compressed image
        axes[0, 1].imshow(results['compressed_img'])
        axes[0, 1].set_title('Compressed Image')
        axes[0, 1].axis('off')
        
        # OCR mask
        axes[0, 2].imshow(results['ocr_mask'], cmap='gray')
        axes[0, 2].set_title('OCR Mask')
        axes[0, 2].axis('off')
        
        # Predicted forgery mask
        axes[1, 0].imshow(results['pred_mask'], cmap='hot')
        axes[1, 0].set_title('Predicted Forgery Mask')
        axes[1, 0].axis('off')
        
        # Forgery probability heatmap
        axes[1, 1].imshow(results['pred_prob'][1], cmap='jet', vmin=0, vmax=1)
        axes[1, 1].set_title('Forgery Probability')
        axes[1, 1].axis('off')
        
        # Overlay
        overlay = results['compressed_img'].copy()
        mask_colored = np.zeros_like(overlay)
        mask_colored[results['pred_mask'] == 1] = [255, 0, 0]  # Red for forgery
        overlay = cv2.addWeighted(overlay, 0.6, mask_colored, 0.4, 0)
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title('Forgery Overlay')
        axes[1, 2].axis('off')
        
        # Add alignment score as text
        align_text = f"Alignment Score: {results['align_score'][1]:.3f}"
        plt.suptitle(align_text, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def cleanup_temp_dir(self):
        """
        Clean up all temporary files (optional, call manually if needed).
        This will remove the entire temp directory and all files within it.
        """
        if self.temp_dir.exists():
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up temp directory: {self.temp_dir}")
            except Exception as e:
                print(f"Warning: Could not clean temp directory: {e}")


def main():
    parser = argparse.ArgumentParser(description='ADCD-Net Single Image Inference')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to ADCD-Net checkpoint')
    parser.add_argument('--qt-table', type=str, required=True,
                       help='Path to quantization table pickle')
    parser.add_argument('--docres', type=str, default=None,
                       help='Path to DocRes checkpoint (optional, not needed for trained ADCD-Net)')
    parser.add_argument('--craft', type=str, default=None,
                       help='Path to CRAFT checkpoint (optional)')
    parser.add_argument('--ocr-bbox', type=str, default=None,
                       help='Path to precomputed OCR bbox pickle (optional)')
    parser.add_argument('--jpeg-quality', type=int, default=100,
                       help='JPEG quality factor (1-100)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output visualization')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--temp-dir', type=str, default='./temp_inference',
                       help='Directory for temporary files (default: ./temp_inference)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
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
    
    # Initialize inference
    print("Initializing inference pipeline...")
    inferencer = SingleImageInference(
        model_ckpt_path=args.model,
        qt_table_path=args.qt_table,
        docres_ckpt_path=args.docres,
        craft_ckpt_path=args.craft,
        device=args.device,
        temp_dir=args.temp_dir
    )
    
    # Run prediction
    results = inferencer.predict(
        img_path=args.image,
        jpeg_quality=args.jpeg_quality,
        ocr_bbox_path=args.ocr_bbox
    )
    
    # Determine output path
    if args.output is None:
        img_name = Path(args.image).stem
        args.output = f"{img_name}_prediction.png"
    
    # Visualize
    inferencer.visualize_results(results, save_path=args.output)
    
    print("\nInference completed successfully!")
    print(f"Forgery detected in {(results['pred_mask'] == 1).sum()} pixels")
    print(f"DCT alignment score: {results['align_score'][1]:.3f}")


if __name__ == '__main__':
    main()
