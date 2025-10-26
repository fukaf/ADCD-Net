# ADCD-Net: Complete Workflow Documentation

## Table of Contents
1. [Overview](#overview)
2. [Training Pipeline](#training-pipeline)
3. [Validation Pipeline](#validation-pipeline)
4. [Data Loading & Preprocessing](#data-loading--preprocessing)
5. [Model Architecture](#model-architecture)
6. [Loss Functions](#loss-functions)
7. [Function Reference](#function-reference)

---

## Overview

**ADCD-Net** (Adaptive DCT-based Document Forgery Detection Network) is a document tampering detection model that uses:
- **RGB image features** (via Restormer transformer encoder)
- **DCT (Discrete Cosine Transform) features** (via FPH encoder) from JPEG compression artifacts
- **Hierarchical content disentanglement** (separating content from forgery)
- **Multi-scale feature fusion** with DCT alignment scoring

**Paper**: ICCV 2025

---

## Training Pipeline

### Entry Point: `main.py`

#### 1. **Initialization** (`Trainer.__init__`)
**Location**: `main.py`, lines 64-80

```python
class Trainer(nn.Module):
    def __init__(self):
        # Load training and validation data loaders
        self.train_dl, self.val_dls = get_train_dl(), get_val_dl()
        
        # Initialize model
        self.model = ADCDNet()
        
        # Load pretrained weights if available
        if cfg.ckpt is not None:
            self.load_ckpt(cfg.ckpt)
        
        # Setup optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, cfg.epochs, eta_min=cfg.min_lr)
        
        # Initialize loss functions
        self.ce = SoftCrossEntropyLoss(smooth_factor=0.1)
        self.lovasz = LovaszLoss(mode='multiclass', per_image=True)
        self.l1 = nn.L1Loss()
        self.align_ce = nn.CrossEntropyLoss()
```

**Key Components**:
- `get_train_dl()` → Returns training DataLoader
- `get_val_dl()` → Returns dictionary of validation DataLoaders
- `ADCDNet()` → Main model architecture
- Multiple loss functions for different training objectives

---

#### 2. **Training Loop** (`Trainer.train`)
**Location**: `main.py`, lines 82-137

```python
def train(self):
    for epoch in range(1, cfg.epochs + 1):
        # Update curriculum learning parameter
        if epoch != 1:
            self.train_dl.dataset.S += cfg.cnt_per_epoch
        
        for items in self.train_dl:
            # Extract batch data
            img, dct, qt, mask, ocr_mask, is_align = items[...]
            
            # Forward pass
            with autocast():
                logits, loc_feat, align_logits, rec_output, focal_losses = \
                    self.model(img, dct, qt, mask, ocr_mask, is_train=True)
            
            # Compute losses
            total_loss = ce_loss + iou_loss + focal_loss + rec_loss + align_loss + norm_loss
            
            # Backward pass
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        # Validation every N epochs
        if epoch % cfg.val_epoch == 0:
            val_score = self.val()
            self.save_ckpt(epoch, val_score)
        
        self.scheduler.step()
```

**Training Flow**:
1. **Curriculum learning**: Gradually increases JPEG compression difficulty
2. **Data loading**: Gets batch with RGB, DCT, quantization tables, masks
3. **Forward pass**: Model produces segmentation logits and auxiliary outputs
4. **Loss computation**: Combines 6 different losses
5. **Backpropagation**: Updates model weights with mixed precision training
6. **Validation**: Periodic evaluation on validation sets

---

## Validation Pipeline

### Validation Function (`Trainer.val`)
**Location**: `main.py`, lines 139-163

```python
def val(self):
    self.model.eval()
    with torch.no_grad():
        val_f1_list = []
        for val_name, dl in self.val_dls.items():
            for items in dl:
                img, dct, qt, mask, ocr_mask = items[...]
                
                # Forward pass (inference mode)
                with autocast():
                    logits = self.model(img, dct, qt, mask, ocr_mask, is_train=False)[0]
                
                # Compute metrics
                per_f1, per_p, per_r = self.compute_f1(logits, mask)
                p_list.append(per_p)
                r_list.append(per_r)
            
            # Calculate F1 score
            p = np.array(p_list).mean()
            r = np.array(r_list).mean()
            f1 = (2 * p * r / (p + r + 1e-8))
        
        return np.array(val_f1_list).mean()
```

**Differences from Training**:
- `is_train=False` → Only returns segmentation logits
- No reconstruction branch
- No focal loss computation
- Metric: F1 score (Precision & Recall)

---

## Data Loading & Preprocessing

### Dataset Architecture

#### Training Dataset: `TrainDs`
**Location**: `ds.py`, lines 181-256

```python
class TrainDs(Dataset):
    def __init__(self):
        # Load LMDB database
        self.lmdb = lmdb.open(lmdb_path, ...)
        
        # Load quantization tables
        self.qts = load_qt(cfg.qt_path)
        
        # Setup augmentations
        self.align_aug = get_align_aug()      # DCT-aligned crops
        self.non_align_aug = get_non_align_aug()  # Non-aligned crops
```

---

### Data Loading Process (`TrainDs.__getitem__`)

#### Step 1: Load Raw Data
**Function**: `load_data()`  
**Location**: `ds.py`, lines 32-44

```python
def load_data(idx, lmdb):
    # Load image from LMDB
    img_key = 'image-%09d' % idx
    img_buf = lmdb.get(img_key.encode('utf-8'))
    img = Image.open(BytesIO(img_buf))
    
    # Load mask from LMDB
    lbl_key = 'label-%09d' % idx
    lbl_buf = lmdb.get(lbl_key.encode('utf-8'))
    mask = cv2.imdecode(np.frombuffer(lbl_buf, dtype=np.uint8), 0)
    
    return img, mask
```

**Output**: PIL Image, numpy array mask

---

#### Step 2: Load OCR Character Bounding Boxes
**Location**: `ds.py`, lines 214-219

```python
# Load character segmentation (OCR bboxes)
char_seg_path = op.join(self.ocr_dir, img_name + '.pkl')
if op.exists(char_seg_path):
    with open(char_seg_path, 'rb') as f:
        c_bbox = pickle.load(f)  # List of [x1, y1, x2, y2]
else:
    c_bbox = []
```

**Purpose**: Text regions are treated specially to avoid false positives

---

#### Step 3: Data Augmentation
**Functions**: `get_align_aug()`, `get_non_align_aug()`  
**Location**: `ds.py`, lines 146-180

```python
if random() > 0.5:  # 50% probability
    aug_func = self.align_aug      # DCT grid-aligned augmentation
    is_align = True
else:
    aug_func = self.non_align_aug  # Non-aligned augmentation
    is_align = False

aug_out = aug_func(image=img, mask=mask, bboxes=c_bbox)
img, mask, c_bbox = aug_out['image'], aug_out['mask'], aug_out['bboxes']
```

**Align Augmentation** (`AlignCrop`):
- Crops aligned to 8×8 DCT grid boundaries
- Geometric transforms: flip, rotate, transpose
- Color augmentations: noise, brightness, contrast

**Non-Align Augmentation** (`NonAlignCrop`):
- Crops NOT aligned to DCT grid
- Additional: downscale, gaussian blur
- Tests model robustness to DCT misalignment

---

#### Step 4: Create OCR Mask
**Function**: `bbox_2_mask()`  
**Location**: `ds.py`, lines 54-67

```python
def bbox_2_mask(bbox, ori_h, ori_w, expand_ratio=0.1):
    ocr_mask = np.zeros([ori_h, ori_w])
    for char_bbox in bbox:
        x1, y1, x2, y2 = char_bbox
        w = x2 - x1
        h = y2 - y1
        # Expand bbox by 10%
        x1 = int(max(0, x1 - w * expand_ratio))
        y1 = int(max(0, y1 - h * expand_ratio))
        x2 = int(min(ori_w, x2 + w * expand_ratio))
        y2 = int(min(ori_h, y2 + h * expand_ratio))
        ocr_mask[int(y1):int(y2), int(x1):int(x2)] = 1
    return ocr_mask
```

**Output**: Binary mask (H, W) where 1 = text region

---

#### Step 5: Multi-JPEG Compression
**Function**: `multi_jpeg()`  
**Location**: `ds.py`, lines 70-107

```python
def multi_jpeg(img, num_jpeg, min_qf, upper_bound, jpeg_record=None):
    # Apply multiple JPEG compressions
    for each_jpeg in range(num_jpeg):
        qf = randint(min_qf, upper_bound)  # Random quality factor
        img.save(tmp.name, "JPEG", quality=int(qf))
        img = Image.open(tmp.name)
    
    # Extract DCT coefficients using jpeg2dct
    dct_y, _, _ = load(tmp.name, normalized=False)  # Shape: [h, w, 64]
    
    # Reshape from block format to spatial format
    rows, cols, _ = dct_y.shape
    dct = np.empty(shape=(8 * rows, 8 * cols))
    for j in range(rows):
        for i in range(cols):
            # Each block is 8×8 DCT coefficients
            dct[8*j : 8*(j+1), 8*i : 8*(i+1)] = dct_y[j, i].reshape(8, 8)
    
    return np.int32(dct), img, qf_record
```

**Process**:
1. Convert image to grayscale
2. Apply 1-3 random JPEG compressions (curriculum learning)
3. Extract DCT coefficients from JPEG file
4. Convert from block format (h, w, 64) to spatial format (H, W)

**Output**: 
- `dct`: DCT coefficients as int32 array (H, W)
- `img`: RGB image after compression
- `qf_record`: List of quality factors used

---

#### Step 6: Load Quantization Table
**Function**: `load_qt()`  
**Location**: `ds.py`, lines 24-30

```python
def load_qt(qt_path):
    with open(qt_path, 'rb') as fpk:
        pks_ = pickle.load(fpk)  # Dict: {quality_factor: qt_matrix}
    pks = {}
    for k, v in pks_.items():
        pks[k] = torch.LongTensor(v)  # Shape: [64] (flattened 8×8 table)
    return pks
```

**Usage**:
```python
qf = qfs[-1]  # Last quality factor used
qt = self.qts[qf]  # Get corresponding quantization table
```

---

#### Step 7: Normalization & Tensorization
**Location**: `ds.py`, lines 175-178, 243-246

```python
# RGB normalization (ImageNet statistics)
img_totsr = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))
])

# Apply transformations
img = self.img_totsr(img)  # (3, H, W), normalized
mask = self.mask_totsr(image=mask.copy())['image']  # (1, H, W), long tensor
ocr_mask = self.mask_totsr(image=ocr_mask.copy())['image']  # (1, H, W)
```

---

#### Step 8: Final Output
**Location**: `ds.py`, lines 248-256

```python
return {
    'img': img,                           # (3, H, W) float32, normalized
    'dct': np.clip(np.abs(dct), 0, 20),  # (H, W) int32, clipped to [0, 20]
    'qt': qt,                             # (64,) int64, quantization table
    'mask': mask.long(),                  # (1, H, W) int64, ground truth
    'ocr_mask': ocr_mask.long(),          # (1, H, W) int64, text regions
    'img_name': img_name,                 # str
    'min_qf': min_qf,                     # int, minimum quality factor
    'is_align': is_align                  # bool, DCT alignment flag
}
```

---

### Validation Dataset: `DtdValDs`
**Location**: `ds.py`, lines 259-328

**Differences from Training**:
- Fixed JPEG compression (from `jpeg_record`)
- No random augmentation (optional `val_aug`)
- Pixel shift evaluation mode (`shift_1p`)
- Returns original image for visualization

---

## Model Architecture

### Main Model: `ADCDNet`
**Location**: `model/model.py`, lines 14-115

#### Model Components

```python
class ADCDNet(nn.Module):
    def __init__(self):
        # 1. RGB Encoder + Localization Branch
        self.restormer_loc = get_restormer(model_name='full_model', out_channels=96)
        
        # 2. DCT Encoder
        self.dct_encoder = FPH()
        
        # 3. DCT Alignment Predictor
        self.dct_align_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(256, 128),
            nn.Linear(128, 2)  # Binary classification: aligned or not
        )
        
        # 4. Reconstruction Branch (training only)
        self.restormer_rec = get_restormer(model_name='decoder_only', out_channels=4)
        
        # 5. Focal Loss Projection Heads
        self.focal_proj = nn.ModuleList([...])
        
        # 6. Pristine Prototype (PP) Modules
        self.pp_scale_proj = get_mlp(in_channels=4, out_channels=96)
        self.pp_bias_proj = get_mlp(in_channels=4, out_channels=96)
        
        # 7. Output Head
        self.out_head = get_mlp(in_channels=96, out_channels=2)
```

---

### Forward Pass

#### Mode 1: Training (`is_train=True`)
**Location**: `model/model.py`, lines 43-88

```python
def forward(self, img, dct, qt, mask, ocr_mask, is_train=True):
    # === STEP 1: DCT Processing ===
    as_feat, ms_dct_feats = self.dct_encoder(dct, qt)
    # as_feat: Alignment score feature (B, 256, H/8, W/8)
    # ms_dct_feats: Multi-scale DCT features [4 scales]
    
    # === STEP 2: Alignment Score Prediction ===
    align_logits = self.dct_align_head(as_feat)  # (B, 2)
    align_score = F.softmax(align_logits, dim=1)[:, -1]  # (B,) probability of being aligned
    
    # === STEP 3: RGB Feature Extraction ===
    feat, loc_feat, cnt_feats, frg_feats, pp_feats = self.restormer_loc(
        img=torch.cat([self.add_coord(img), torch.zeros_like(ocr_mask)], 1),
        ms_dct_feats=ms_dct_feats,
        dct_align_score=align_score,
        ocr_mask=ocr_mask,
        img_size=img_size
    )
    # feat: Main feature (B, 96, H, W)
    # loc_feat: Localization feature (B, 96, H, W)
    # cnt_feats: Content features [4 scales]
    # frg_feats: Forgery features [4 scales]
    # pp_feats: Pristine prototype features [4 scales]
    
    # === STEP 4: Pristine Prototype Map ===
    pp_maps = self.get_pp_map(pp_feats, ocr_mask, img_size)  # (B, 4, H, W)
    pp_scale = self.pp_scale_proj(pp_maps)  # (B, 96, H, W)
    pp_bias = self.pp_bias_proj(pp_maps)    # (B, 96, H, W)
    
    # === STEP 5: Feature Modulation ===
    feat = feat * pp_scale + pp_bias
    
    # === STEP 6: Segmentation Output ===
    logits = self.out_head(feat)  # (B, 2, H, W)
    
    if is_train:
        # === STEP 7a: Reconstruction Branch (Training Only) ===
        rec_img = self.restormer_rec(cnt_feats, frg_feats, is_shuffle=False)
        shuffle_rec_img = self.restormer_rec(cnt_feats, frg_feats, is_shuffle=True)
        norm_dct = dct.float() / 20.0
        rec_output = (rec_img, shuffle_rec_img, norm_dct)
        
        # === STEP 7b: Focal Loss Computation ===
        focal_losses = tuple([
            supcon_parallel(self.focal_proj[i](pp_feats[i]), mask)
            for i in range(len(pp_feats))
        ])
        
        return logits, loc_feat, align_logits, rec_output, focal_losses
    else:
        return logits, loc_feat, align_logits, None, None
```

---

#### Mode 2: Inference (`is_train=False`)
**Location**: `model/model.py`, lines 89-93

```python
if not is_train:  # Validation/Inference
    rec_output, focal_losses = None, None
    return logits, loc_feat, align_logits, rec_output, focal_losses
```

**Simplified Flow**:
1. DCT encoding
2. Alignment prediction
3. RGB encoding + feature fusion
4. Pristine prototype estimation
5. Segmentation output
6. **Skip reconstruction & focal loss**

---

### Sub-Module 1: FPH (Feature Pyramid with Hierarchy)
**Location**: `model/fph.py`, lines 121-191

```python
class FPH(nn.Module):
    def __init__(self):
        # DCT embedding
        self.dct_proj = nn.Embedding(21, 21)  # Map DCT values [0-20] to embeddings
        
        # Quantization table embedding
        self.qt_proj = nn.Embedding(64, 16)   # Map QT values to 16-dim
        
        # DCT processing convolutions
        self.conv_1 = nn.Conv2d(21, 64, kernel_size=3, dilation=8)
        self.conv_2 = nn.Conv2d(64, 16, kernel_size=1)
        
        # MobileNet-style blocks
        self.mbconv_blocks = nn.Sequential(
            nn.Conv2d(35, 256, kernel_size=8, stride=8),  # 16*2 + 3 coords
            MBConvBlock(...),
            MBConvBlock(...),
            MBConvBlock(...)
        )
        
        # Multi-scale output projections
        self.conv_out = nn.ModuleList([
            nn.Conv2d(256, 48),   # Scale 0
            nn.Conv2d(256, 96),   # Scale 1
            nn.Conv2d(256, 192),  # Scale 2
            nn.Conv2d(256, 384)   # Scale 3
        ])
    
    def forward(self, dct, qt):
        # 1. DCT embedding
        dct_embed = self.dct_proj(dct).permute(0, 3, 1, 2)  # (B, 21, H, W)
        
        # 2. Convolutional processing
        x = self.conv_2(self.conv_1(dct_embed))  # (B, 16, H, W)
        
        # 3. Quantization table modulation
        # Reshape x to 8×8 blocks
        x_ = x.reshape(b, c, h//8, 8, w//8, 8).permute(0, 1, 3, 5, 2, 4)
        
        # Embed QT and multiply
        qt_embed = self.qt_proj(qt.long())  # (B, 1, 1, 8, 8, 16)
        times = x_ * qt_embed  # Element-wise multiplication
        
        # Reshape back
        times = times.permute(0, 1, 4, 2, 5, 3).reshape(b, c, h, w)
        
        # 4. Concatenate and add coordinates
        cat = torch.cat((times, x), dim=1)  # (B, 32, H, W)
        cat = self.addcoord(cat)  # Add x, y coords → (B, 35, H, W)
        
        # 5. MBConv processing
        dct_feat = self.mbconv_blocks(cat)  # (B, 256, H/8, W/8)
        
        # 6. Multi-scale outputs
        ms_dct_feats = [
            F.interpolate(self.conv_out[i](dct_feat), size=(h//(2**i), w//(2**i)))
            for i in range(4)
        ]
        
        return dct_feat, ms_dct_feats
```

**Key Operations**:
- **DCT Embedding**: Maps integer DCT coefficients [0-20] to learnable vectors
- **QT Modulation**: Applies quantization table information at block level
- **Coordinate Encoding**: Adds spatial position information
- **Multi-scale**: Outputs features at 4 resolution levels

---

### Sub-Module 2: Restormer (Localization Branch)
**Location**: `model/restormer.py`, lines 245-353

```python
class Restormer(nn.Module):
    def forward(self, img, ms_dct_feats, dct_align_score, ocr_mask, img_size):
        # === Encoder Path ===
        # Level 1: 1/1 resolution
        inp_enc_level1 = self.patch_embed(img)
        inp_enc_level1 = inp_enc_level1 + dct_align_score * ms_dct_feats[0]
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        cnt_feat_lvl1, frg_feat_lvl1 = self.cdm[0](out_enc_level1).chunk(2, dim=1)
        
        # Level 2: 1/2 resolution
        inp_enc_level2 = self.down1_2(out_enc_level1)
        inp_enc_level2 = inp_enc_level2 + dct_align_score * ms_dct_feats[1]
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        cnt_feat_lvl2, frg_feat_lvl2 = self.cdm[1](out_enc_level2).chunk(2, dim=1)
        
        # Level 3: 1/4 resolution
        inp_enc_level3 = self.down2_3(out_enc_level2)
        inp_enc_level3 = inp_enc_level3 + dct_align_score * ms_dct_feats[2]
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        cnt_feat_lvl3, frg_feat_lvl3 = self.cdm[2](out_enc_level3).chunk(2, dim=1)
        
        # Level 4: 1/8 resolution (latent)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        inp_enc_level4 = inp_enc_level4 + dct_align_score * ms_dct_feats[3]
        latent = self.latent(inp_enc_level4)
        cnt_feat_lat, frg_feat_lat = self.cdm[3](latent).chunk(2, dim=1)
        
        # === Decoder Path (Forgery Branch) ===
        inp_dec_level3 = self.up4_3(frg_feat_lat)
        inp_dec_level3 = torch.cat([inp_dec_level3, frg_feat_lvl3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, frg_feat_lvl2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, frg_feat_lvl1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level0 = self.refinement(out_dec_level1)
        
        # === Outputs ===
        cnt_feats = (cnt_feat_lvl1, cnt_feat_lvl2, cnt_feat_lvl3, cnt_feat_lat)
        frg_feats = (frg_feat_lvl1, frg_feat_lvl2, frg_feat_lvl3, frg_feat_lat)
        pp_feats = (out_dec_level3, out_dec_level2, out_dec_level1, out_dec_level0)
        
        out_feat = self.output(out_dec_level0)
        
        return out_feat, out_dec_level0, cnt_feats, frg_feats, pp_feats
```

**Architecture Details**:
- **Transformer-based**: Uses Multi-DConv Head Transposed Attention (MDTA)
- **U-Net structure**: 4-level encoder-decoder with skip connections
- **DCT Fusion**: At each level, adds DCT features weighted by alignment score
- **Content Disentanglement Module (CDM)**: Splits features into content + forgery

---

### Sub-Module 3: Restormer Decoder (Reconstruction Branch)
**Location**: `model/restormer.py`, lines 356-417

```python
class RestormerDec(nn.Module):
    def forward(self, cnt_feats, frg_feats, is_shuffle=False):
        # Combine content + forgery features
        if is_shuffle:
            # Spatially shuffle forgery features (data augmentation)
            out_enc_level1 = cnt_feats[0] + self.spatial_shuffle(frg_feats[0])
            out_enc_level2 = cnt_feats[1] + self.spatial_shuffle(frg_feats[1])
            out_enc_level3 = cnt_feats[2] + self.spatial_shuffle(frg_feats[2])
            latent = cnt_feats[3] + self.spatial_shuffle(frg_feats[3])
        else:
            out_enc_level1 = cnt_feats[0] + frg_feats[0]
            out_enc_level2 = cnt_feats[1] + frg_feats[1]
            out_enc_level3 = cnt_feats[2] + frg_feats[2]
            latent = cnt_feats[3] + frg_feats[3]
        
        # Decoder path (same as main Restormer)
        ...
        
        return reconstructed_image  # (B, 4, H, W) = RGB + DCT
```

**Purpose**: 
- Reconstructs RGB + DCT from disentangled features
- Ensures content/forgery separation is meaningful
- Spatial shuffle prevents trivial solutions

---

### Pristine Prototype Map (`get_pp_map`)
**Location**: `model/model.py`, lines 90-115

```python
def get_pp_map(self, pp_feats, y, img_size):
    maps_per_level = []
    y = y.float()
    
    for f in pp_feats:  # For each pyramid level
        f = F.normalize(f, p=2, dim=1)  # L2 normalize features
        _, c, h, w = f.shape
        
        # Resize mask to feature resolution
        bg_mask = (F.interpolate(y, size=(h, w), mode="nearest") == 0).float()
        
        # Compute background mean feature (pristine prototype)
        bg_sum = (f * bg_mask).sum(dim=(2, 3))  # (B, C)
        bg_count = bg_mask.sum(dim=(2, 3)).clamp_min(1.0)
        bg_mean = F.normalize(bg_sum / bg_count, p=2, dim=1)  # (B, C)
        
        # Compute cosine similarity map
        sim = (f * bg_mean[:, :, None, None]).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Upscale to original size
        sim = F.interpolate(sim, size=(img_size, img_size), 
                           mode="bilinear", align_corners=False)
        
        maps_per_level.append(sim)
    
    # Stack all levels
    return torch.cat(maps_per_level, dim=1)  # (B, 4, H, W)
```

**Concept**:
- Estimate "pristine" feature from authentic (background) regions
- Compute similarity between each pixel and pristine prototype
- Use similarity as attention map to modulate features
- Multi-scale: 4 different resolution levels

---

## Loss Functions

### Total Loss Composition
**Location**: `main.py`, lines 115-119

```python
total_loss = ce_loss + iou_loss + focal_loss + rec_loss + align_loss + norm_loss
```

---

### 1. Cross-Entropy Loss (Segmentation)
**Function**: `SoftCrossEntropyLoss`  
**Location**: `loss/soft_ce_loss.py`  
**Weight**: `cfg.ce_w = 3`

```python
self.ce = SoftCrossEntropyLoss(smooth_factor=0.1, reduction="mean")
ce_loss = cfg.ce_w * self.ce(logits.float(), mask)
```

**Purpose**: 
- Main segmentation loss
- Label smoothing (0.1) prevents overconfidence
- Input: `logits` (B, 2, H, W), `mask` (B, 1, H, W)

---

### 2. Lovász-Softmax Loss (IoU Optimization)
**Function**: `LovaszLoss`  
**Location**: `loss/lovasz_loss.py`  
**Weight**: `1.0` (implicit)

```python
self.lovasz = LovaszLoss(mode='multiclass', per_image=True)
iou_loss = self.lovasz(logits.float(), mask)
```

**Purpose**:
- Directly optimizes IoU metric
- Differentiable surrogate for Jaccard index
- `per_image=True`: Computes loss independently per sample

---

### 3. Focal Loss (Supervised Contrastive)
**Function**: `supcon_parallel`  
**Location**: `loss/focal_loss.py`  
**Weights**: `cfg.focal_w = [0.001, 0.005, 0.02, 0.1]` (4 scales)

```python
focal_losses = [
    cfg.focal_w[idx] * (loss.sum() / (loss != 0).sum())
    for idx, loss in enumerate(focal_losses)
]
focal_loss = torch.stack(focal_losses).sum()
```

**Purpose**:
- Contrastive learning at feature level
- Pulls together features from same class
- Pushes apart features from different classes
- Applied at 4 different pyramid levels

**Algorithm** (`supcon_parallel`):
1. Sample 256 positive + 256 negative features per image
2. Compute pairwise similarity matrix
3. For each anchor, maximize similarity to same-class samples
4. Minimize similarity to different-class samples

---

### 4. Reconstruction Loss
**Type**: L1 Loss  
**Location**: `main.py`, lines 109-111  
**Weight**: `cfg.rec_w = 1`

```python
img_l1_loss = self.l1(rec_img[:, :3], img) + self.l1(shuffle_rec_img[:, :3], img)
dct_l1_loss = self.l1(rec_img[:, -1], norm_dct) + self.l1(shuffle_rec_img[:, -1], norm_dct)
rec_loss = cfg.rec_w * (img_l1_loss + dct_l1_loss)
```

**Purpose**:
- Ensures content/forgery disentanglement is meaningful
- Reconstruct RGB channels + DCT channel
- Two versions: normal + spatially shuffled forgery features

**Components**:
- `rec_img[:, :3]`: RGB reconstruction (channels 0-2)
- `rec_img[:, -1]`: DCT reconstruction (channel 3)
- `shuffle_rec_img`: Reconstruction with shuffled forgery features

---

### 5. Alignment Loss
**Type**: Cross-Entropy  
**Location**: `main.py`, line 113  
**Weight**: `1.0` (implicit)

```python
self.align_ce = nn.CrossEntropyLoss()
align_loss = self.align_ce(align_logits, is_align.long())
```

**Purpose**:
- Train model to predict if image crop is DCT grid-aligned
- Binary classification: aligned (1) or not (0)
- Used to weight DCT feature importance

---

### 6. Normalization Loss
**Type**: Feature Norm Penalty  
**Location**: `main.py`, line 116  
**Weight**: `cfg.norm_w = 0.1`

```python
norm_loss = cfg.norm_w * loc_feat.norm(dim=1).mean()
```

**Purpose**:
- Regularization: Prevents feature magnitudes from exploding
- Computed on localization features
- Improves training stability

---

## Function Reference

### Configuration (`cfg.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `gpus` | '0,1,2,3' | GPU IDs to use |
| `train_bs` | 16 | Training batch size |
| `val_bs` | 32 | Validation batch size |
| `epochs` | 150 | Total training epochs |
| `lr` | 3e-4 | Initial learning rate |
| `min_lr` | 1e-5 | Minimum learning rate (cosine annealing) |
| `img_size` | 256 | Input image size |
| `min_qf` | 75 | Minimum JPEG quality factor |
| `ce_w` | 3 | Cross-entropy loss weight |
| `rec_w` | 1 | Reconstruction loss weight |
| `focal_w` | [0.001, 0.005, 0.02, 0.1] | Focal loss weights (4 scales) |
| `norm_w` | 0.1 | Normalization loss weight |

---

### Key Functions Summary

#### Data Loading
| Function | Location | Input | Output | Purpose |
|----------|----------|-------|--------|---------|
| `load_data()` | ds.py:32 | idx, lmdb | img, mask | Load image and mask from LMDB |
| `load_qt()` | ds.py:24 | qt_path | dict | Load quantization tables |
| `multi_jpeg()` | ds.py:70 | img, num_jpeg, qf | dct, img, qf_list | Apply JPEG compression and extract DCT |
| `bbox_2_mask()` | ds.py:54 | bbox, h, w | ocr_mask | Convert bounding boxes to mask |
| `get_train_dl()` | ds.py:331 | - | DataLoader | Create training data loader |
| `get_val_dl()` | ds.py:339 | - | dict | Create validation data loaders |

#### Augmentation
| Function | Location | Input | Output | Purpose |
|----------|----------|-------|--------|---------|
| `get_align_aug()` | ds.py:146 | - | Compose | DCT-aligned augmentation pipeline |
| `get_non_align_aug()` | ds.py:160 | - | Compose | Non-aligned augmentation pipeline |
| `AlignCrop.apply()` | ds.py:112 | img, coords | img | Crop aligned to 8×8 grid |
| `NonAlignCrop.apply()` | ds.py:119 | img, coords | img | Crop NOT aligned to grid |

#### Model Architecture
| Function | Location | Input | Output | Purpose |
|----------|----------|-------|--------|---------|
| `ADCDNet.__init__()` | model/model.py:14 | - | - | Initialize model components |
| `ADCDNet.forward()` | model/model.py:43 | img, dct, qt, mask, ocr_mask | logits, ... | Main forward pass |
| `ADCDNet.get_pp_map()` | model/model.py:90 | pp_feats, mask, size | pp_maps | Compute pristine prototype maps |
| `FPH.__init__()` | model/fph.py:121 | - | - | Initialize DCT encoder |
| `FPH.forward()` | model/fph.py:168 | dct, qt | as_feat, ms_dct_feats | Encode DCT features |
| `Restormer.forward()` | model/restormer.py:282 | img, ms_dct_feats, ... | feat, cnt_feats, frg_feats, ... | RGB encoding + U-Net |
| `RestormerDec.forward()` | model/restormer.py:381 | cnt_feats, frg_feats | rec_img | Reconstruction decoder |

#### Training
| Function | Location | Input | Output | Purpose |
|----------|----------|-------|--------|---------|
| `Trainer.__init__()` | main.py:64 | - | - | Initialize trainer |
| `Trainer.train()` | main.py:82 | - | - | Main training loop |
| `Trainer.val()` | main.py:139 | - | f1_score | Validation evaluation |
| `Trainer.compute_f1()` | main.py:189 | logit, mask | f1, p, r | Compute F1 metrics |
| `Trainer.save_ckpt()` | main.py:213 | epoch, score | - | Save checkpoint |
| `Trainer.load_ckpt()` | main.py:202 | ckpt_path | - | Load checkpoint |

#### Loss Functions
| Function | Location | Input | Output | Purpose |
|----------|----------|-------|--------|---------|
| `supcon_parallel()` | loss/focal_loss.py:6 | f, y, t | loss | Supervised contrastive loss |
| `SoftCrossEntropyLoss.forward()` | loss/soft_ce_loss.py:62 | y_pred, y_true | loss | CE with label smoothing |
| `LovaszLoss.forward()` | loss/lovasz_loss.py:238 | y_pred, y_true | loss | Lovász-Softmax IoU loss |

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                        │
└─────────────────────────────────────────────────────────────────┘

1. DATA LOADING (ds.py)
   ┌─────────────┐
   │ LMDB Dataset│
   └──────┬──────┘
          │ load_data()
          ▼
   ┌──────────────────┐     ┌──────────────┐
   │ RGB Image + Mask │────►│ OCR Bboxes   │
   └────────┬─────────┘     └──────┬───────┘
            │                      │
            │ Augmentation         │ bbox_2_mask()
            ▼                      ▼
   ┌──────────────────┐     ┌──────────────┐
   │ Augmented Image  │     │  OCR Mask    │
   └────────┬─────────┘     └──────────────┘
            │
            │ multi_jpeg()
            ▼
   ┌──────────────────────────────────┐
   │ DCT Coefficients + Compressed RGB│
   └────────┬─────────────────────────┘
            │
            │ Normalization
            ▼
   ┌──────────────────────────────────────────┐
   │ Batch: img, dct, qt, mask, ocr_mask      │
   └────────┬─────────────────────────────────┘
            │
            │
            ▼

2. MODEL FORWARD (model/model.py)
   ┌─────────────────────┐
   │   DCT Encoder (FPH) │
   │  ┌───────────────┐  │
   │  │ DCT Embedding │  │
   │  │ QT Modulation │  │
   │  │ MBConv Blocks │  │
   │  └───────┬───────┘  │
   └──────────┼──────────┘
              │
              ├─► align_score (scalar per sample)
              │
              └─► ms_dct_feats (4 scales)
                       │
                       │
   ┌───────────────────┼────────────────────────┐
   │  RGB Encoder      │    (Restormer)         │
   │  ┌────────────────▼────────────┐           │
   │  │ Patch Embed + DCT Fusion    │           │
   │  └────────────┬─────────────────┘          │
   │               │                             │
   │  ┌────────────▼─────────────┐              │
   │  │ Encoder Level 1 → CDM    │ ─► cnt_feat, frg_feat
   │  └────────────┬─────────────┘              │
   │               │ Downsample                  │
   │  ┌────────────▼─────────────┐              │
   │  │ Encoder Level 2 → CDM    │ ─► cnt_feat, frg_feat
   │  └────────────┬─────────────┘              │
   │               │ Downsample                  │
   │  ┌────────────▼─────────────┐              │
   │  │ Encoder Level 3 → CDM    │ ─► cnt_feat, frg_feat
   │  └────────────┬─────────────┘              │
   │               │ Downsample                  │
   │  ┌────────────▼─────────────┐              │
   │  │ Latent Level → CDM       │ ─► cnt_feat, frg_feat
   │  └────────────┬─────────────┘              │
   │               │                             │
   │  ┌────────────▼─────────────┐              │
   │  │ Decoder (Forgery Branch) │              │
   │  │  - Upsample + Skip       │              │
   │  │  - Level 3, 2, 1, 0      │              │
   │  └────────────┬─────────────┘              │
   └───────────────┼────────────────────────────┘
                   │
                   ├─► loc_feat (for norm loss)
                   ├─► pp_feats (for focal loss & PP map)
                   │
   ┌───────────────▼──────────────┐
   │ Pristine Prototype Estimation│
   │  - Background mean feature   │
   │  - Cosine similarity         │
   │  - Multi-scale fusion        │
   └───────────────┬──────────────┘
                   │
                   ├─► pp_scale, pp_bias
                   │
   ┌───────────────▼──────────────┐
   │ Feature Modulation           │
   │   feat = feat * scale + bias │
   └───────────────┬──────────────┘
                   │
   ┌───────────────▼──────────────┐
   │ Output Head (MLP)            │
   │   logits: (B, 2, H, W)       │
   └──────────────────────────────┘
                   │
                   │
                   ▼
   (If training: also compute reconstruction & focal loss)

3. LOSS COMPUTATION (main.py)
   ┌────────────────────────────┐
   │ Segmentation Losses        │
   │  - CE Loss (logits, mask)  │
   │  - Lovász Loss (IoU)       │
   └──────────┬─────────────────┘
              │
   ┌──────────▼─────────────────┐
   │ Reconstruction Loss        │
   │  - RGB L1 Loss             │
   │  - DCT L1 Loss             │
   │  - Shuffled version        │
   └──────────┬─────────────────┘
              │
   ┌──────────▼─────────────────┐
   │ Alignment Loss             │
   │  - CE Loss (align_logits)  │
   └──────────┬─────────────────┘
              │
   ┌──────────▼─────────────────┐
   │ Focal Loss (4 scales)      │
   │  - Supervised Contrastive  │
   └──────────┬─────────────────┘
              │
   ┌──────────▼─────────────────┐
   │ Norm Loss                  │
   │  - Feature magnitude       │
   └──────────┬─────────────────┘
              │
   ┌──────────▼─────────────────┐
   │ Total Loss                 │
   │ = Σ all weighted losses    │
   └──────────┬─────────────────┘
              │
              ▼
   ┌──────────────────────────┐
   │ Backpropagation          │
   │  - Mixed Precision (AMP) │
   │  - AdamW Optimizer       │
   └──────────────────────────┘
```

---

## Key Design Decisions

### 1. **Why Multi-JPEG Compression?**
- Real-world documents undergo multiple compressions
- Curriculum learning: Start with high quality, gradually decrease
- Trains model to be robust to various compression levels

### 2. **Why DCT Grid Alignment Matters?**
- JPEG compression works on 8×8 blocks
- Misaligned crops have different DCT patterns
- Model learns to detect and adapt to alignment

### 3. **Why Content Disentanglement?**
- Separates "what" (content) from "whether forged"
- Prevents model from using content shortcuts
- Enables better generalization

### 4. **Why Pristine Prototype?**
- Estimates authentic image characteristics
- Uses only background (non-tampered) regions
- Provides reference for detecting anomalies

### 5. **Why OCR Mask?**
- Text regions have unique compression artifacts
- Prevents false positives from text
- Model learns to ignore text-specific patterns

---

## Inference Workflow

For single image inference (not in training code), the simplified flow is:

```
Input Image (RGB)
    │
    ├─► JPEG Compression (quality factor)
    │   └─► Compressed RGB + DCT Extraction
    │
    ├─► (Optional) OCR Text Detection
    │   └─► OCR Mask
    │
    ▼
Model Forward (is_train=False)
    │
    ├─► DCT Encoding → align_score + ms_dct_feats
    ├─► RGB Encoding + DCT Fusion
    ├─► Pristine Prototype Estimation
    ├─► Feature Modulation
    └─► Segmentation Logits (B, 2, H, W)
         │
         └─► Argmax → Binary Mask (0=authentic, 1=forged)
```

---

## Performance Metrics

### Training Metrics
- **F1 Score**: Primary metric (harmonic mean of precision & recall)
- **Precision**: Correct tampered predictions / All tampered predictions
- **Recall**: Correct tampered predictions / All actual tampered pixels
- **Alignment Accuracy**: % of correct DCT alignment predictions

### Loss Values (Typical)
- Total Loss: 1-3 (decreases during training)
- CE Loss: 0.5-1.5
- IoU Loss: 0.3-0.8
- Reconstruction Loss: 0.1-0.3
- Focal Loss: 0.05-0.2 (sum of 4 scales)
- Alignment Loss: 0.2-0.6

---

## File Organization

```
ADCD-Net/
├── main.py              # Training/validation entry point
├── cfg.py               # Configuration parameters
├── ds.py                # Dataset classes & data loading
├── model/
│   ├── __init__.py
│   ├── model.py         # ADCDNet main architecture
│   ├── fph.py           # DCT encoder (Feature Pyramid with Hierarchy)
│   └── restormer.py     # RGB encoder (Transformer U-Net)
├── loss/
│   ├── __init__.py
│   ├── focal_loss.py    # Supervised contrastive loss
│   ├── soft_ce_loss.py  # Cross-entropy with label smoothing
│   └── lovasz_loss.py   # Lovász-Softmax IoU loss
└── seg_char/            # Optional: OCR text detection (CRAFT)
    └── CRAFT_pytorch/
```

---

## Dependencies

### Core Libraries
- `torch >= 1.11`: Deep learning framework
- `torchvision`: Image transformations
- `albumentations >= 1.3.0`: Data augmentation
- `jpeg2dct`: DCT coefficient extraction
- `lmdb`: Database for efficient data loading
- `PIL / Pillow`: Image I/O
- `opencv-python (cv2)`: Image processing
- `numpy`: Numerical operations
- `einops`: Tensor reshaping

### Optional
- `CRAFT`: Text detection model (for OCR masks)
- `tensorboard`: Training visualization

---

## Additional Notes

### Checkpoint Format
```python
{
    'model': state_dict,  # Model weights (with 'module.' prefix if DataParallel)
}
```

### DocRes Checkpoint
- Pre-trained Restormer weights
- Loaded via `ADCDNet.load_docres()`
- Removes 'output' layer before loading
- Improves initialization for document tasks

### Multi-GPU Training
- Uses `nn.DataParallel` for data parallelism
- Automatically splits batches across GPUs
- Synchronizes gradients during backpropagation

---

**End of Document**

---

*For inference-specific workflows, see: `INFERENCE_GUIDE.md`, `INFERENCE_SUMMARY.md`*  
*For DCT extraction alternatives, see: `DCT_WORKAROUND_README.md`*
