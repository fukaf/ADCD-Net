
# [ICCV'25] ADCD-Net: Robust Document Image Forgery Localization via Adaptive DCT Feature and Hierarchical Content Disentanglement

[![arXiv](https://img.shields.io/badge/arXiv-2507.16397-b31b1b.svg)](https://arxiv.org/abs/2507.16397)

## Description   

The official source code of the paper "ADCD-Net: Robust Document Image Forgery Localization via Adaptive DCT Feature and Hierarchical Content Disentanglement". 

![Model Overview](./fig/model_overview.png)

We present a robust document forgery localization model that adaptively leverages RGB/DCT forensic traces and incorporates key document image traits. To counter DCT traces' sensitivity to block misalignment, we modulate DCT feature contributions via predicted alignment scores, enhancing resilience to distortions like resizing and cropping. A hierarchical content disentanglement method boosts localization by reducing text-background disparities. Leveraging pristine background regions, we build a untampered prototype to improve accuracy and robustness.

## TODO

- [ ] General inference pipline for images outside DocTamper
- [ ] Update better OCR model
- [ ] Evaluate ADCD-Net on [ForensicHub](https://github.com/scu-zjz/ForensicHub) benchmark (cross-domain document IML)

## Environment Setup

**ADCD-Net is trained on 4 NVIDIA GeForce RTX 4090 24G GPUs which takes about 33 hours**

Install dependencies: python 3.8, pytorch 1.11, albumentations 1.3.0

## Data Preparation

Download the DocTamper dataset from [DocTamper](https://github.com/qcf-568/DocTamper) (```qt_table.pk``` and files in ```pks``` can be also found from the DocTamper repository) and the ocr mask and model checkpoints from [ADCD-Net](https://drive.google.com/file/d/10m7v0RrmI68UbfaWCwAN0nfR2y7DWS_4/view?usp=sharing) (the data can be accessed without request from now on).
The files from ADCD-Net is organized as follows:

```
ADCD-Net.pth # ADCD-Net checkpoint
docres.pkl # DocRes checkpoint
DocTamperOCR/ # OCR mask directory
    ├── TrainingSet # Training set directory
    ├── TestingSet # Testing set directory
    ├── FCD # FCD dataset directory
    └── SCD # SCD dataset directory
```

## Get OCR masks of images not in DocTamper

We only provide the ocr mask of the DocTamper dataset. For other document images, you can use the script in ```seg_char``` to get the ocr mask."
First, download the OCR model CRAFT model checkpoint from [CRAFT](https://github.com/clovaai/CRAFT-pytorch). Then, set the checkpoint path and document image path in ```seg_char/main.py``` and run the code to get the ocr mask.

```python
ckpt_path = ''  # TODO: CRAFT model checkpoint
save_dir = '' # TODO: save dir of ocr masks
img_path = '' # TODO: input document image path

char_seger = CharSeger(ckpt_path=ckpt_path,
                       save_dir=save_dir)

char_seger.seg_char_per_img(img_path=img_path)
```

## Training

Before run the ```main.py```, please set the paths of the dataset, ocr mask and model checkpoint in ```cfg.py```. 
The DocRes checkpoint is provided in ```docres.pkl```. 

```python
mode = 'train'
root = 'path/to/root' # TODO:
ckpt = 'path/to/ADCD-Net.pth' # TODO:
docres_ckpt_path = 'path/to/docres.pkl' # TODO:
```

## Evaluation

Before run the ```main.py```, please set the paths of the dataset, distortions, ocr mask and model checkpoint in ```cfg.py```. 
The ADCD-Net checkpoint is provided in ```ADCDNet.pth```. 

```python
mode = 'val'
root = 'path/to/root' # TODO:
ckpt = 'path/to/ADCD-Net.pth' # TODO:
docres_ckpt_path = 'path/to/docres.pkl' # TODO:

multi_jpeg_val = False  # able to use multi jpeg distortion
jpeg_record = False  # manually set multi jpeg distortion record
min_qf = 75  # minimum jpeg quality factor
shift_1p = False  # shift 1 pixel for evaluation
val_aug = None # other distortions can be added here
```
