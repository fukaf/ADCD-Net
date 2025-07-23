import os.path as op
import pickle
import tempfile
from copy import deepcopy
from random import randint
from random import random

import albumentations as A
import albumentations.augmentations.crops.functional as F
import cv2
import lmdb
import numpy as np
import six
import torch
import torchvision.transforms as T
from PIL import Image
from albumentations import CropNonEmptyMaskIfExists
from albumentations.pytorch import ToTensorV2
from jpeg2dct.numpy import load
from torch.utils.data import Dataset, DataLoader

import cfg


def load_qt(qt_path):
    with open(qt_path, 'rb') as fpk:
        pks_ = pickle.load(fpk)
    pks = {}
    for k, v in pks_.items():
        pks[k] = torch.LongTensor(v)
    return pks


def load_data(idx, lmdb):
    img_key = 'image-%09d' % idx
    img_buf = lmdb.get(img_key.encode('utf-8'))
    buf = six.BytesIO()
    buf.write(img_buf)
    buf.seek(0)
    img = Image.open(buf)
    lbl_key = 'label-%09d' % idx
    lbl_buf = lmdb.get(lbl_key.encode('utf-8'))
    mask = (cv2.imdecode(np.frombuffer(lbl_buf, dtype=np.uint8), 0) != 0).astype(np.uint8)
    return img, mask


def load_jpeg_record(record_path):
    with open(record_path, 'rb') as f:
        record = pickle.load(f)
    return record


def bbox_2_mask(bbox, ori_h, ori_w, expand_ratio=0.1):
    ocr_mask = np.zeros([ori_h, ori_w])
    for char_bbox in bbox:
        x1, y1, x2, y2 = char_bbox
        w = x2 - x1
        h = y2 - y1
        x1 = int(max(0, x1 - w * expand_ratio))
        y1 = int(max(0, y1 - h * expand_ratio))
        x2 = int(min(ori_w, x2 + w * expand_ratio))
        y2 = int(min(ori_h, y2 + h * expand_ratio))
        ocr_mask[int(y1):int(y2), int(x1):int(x2)] = 1
    return ocr_mask


def multi_jpeg(img, num_jpeg, min_qf, upper_bound, jpeg_record=None):
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
            dct_y, _, _ = load(tmp.name, normalized=False)
        except:
            with tempfile.NamedTemporaryFile(delete=True) as tmp1:
                qf = 100
                qf_record = [100]
                im_ori.save_ckpt(tmp1, "JPEG", quality=qf)
                img = Image.open(tmp1)
                img = img.convert('RGB')
                dct_y, _, _ = load(tmp1.name, normalized=False)

    # dct_y [h, w, nb]
    rows, cols, _ = dct_y.shape
    dct = np.empty(shape=(8 * rows, 8 * cols))
    for j in range(rows):
        for i in range(cols):
            dct[8 * j: 8 * (j + 1), 8 * i: 8 * (i + 1)] = dct_y[j, i].reshape(8, 8)
    # dct to int32
    dct = np.int32(dct)
    return dct, img, qf_record


class AlignCrop(CropNonEmptyMaskIfExists):
    def apply(self, img, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        # x_min, y_min, x_max, y_max // 8 = 0
        x_diff = x_min % 8
        x_min, x_max = x_min - x_diff, x_max - x_diff
        y_diff = y_min % 8
        y_min, y_max = y_min - y_diff, y_max - y_diff
        return F.crop(img, x_min, y_min, x_max, y_max)


class NonAlignCrop(CropNonEmptyMaskIfExists):
    def apply(self, img, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        # x_min, y_min, x_max, y_max // 8 = 0
        x_diff = x_min % 8
        y_diff = y_min % 8
        if x_diff == 0 and y_diff == 0:  # if align, then make it non-align
            # check if x_min, y_min is 0
            if x_min == 0:
                x_min, x_max = 1, x_max + 1
            if y_min == 0:
                y_min, y_max = 1, y_max + 1
            if x_max == 256:
                x_max, x_min = 255, x_min - 1
            if y_max == 256:
                y_max, y_min = 255, y_min - 1
        return F.crop(img, x_min, y_min, x_max, y_max)


def get_align_aug():
    return A.Compose([
        AlignCrop(cfg.img_size, cfg.img_size, p=1, always_apply=True),
        A.OneOf([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
        ], p=1),
        A.OneOf([
            A.GaussNoise(p=1),
            A.ISONoise(p=1),
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
            A.RandomToneCurve(p=1),
            A.Sharpen(p=1),
        ], p=0.5),
    ], p=1, bbox_params=A.BboxParams(format='pascal_voc',
                                     min_area=16,
                                     min_visibility=0.2,
                                     label_fields=[]))


def get_non_align_aug():
    return A.Compose([
        NonAlignCrop(cfg.img_size, cfg.img_size, p=1, always_apply=True),
        A.Downscale(scale_min=0.5, scale_max=0.99, p=0.5),
        A.GaussianBlur(blur_limit=(3, 9), sigma_limit=(0.5, 0.9), p=0.5),
        A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
        ], p=1.0),
        A.OneOf([
            A.GaussNoise(p=1),
            A.ISONoise(p=1),
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
            A.RandomToneCurve(p=1),
            A.Sharpen(p=1),
        ], p=0.5),
    ], p=1, bbox_params=A.BboxParams(format='pascal_voc',
                                     min_area=16,
                                     min_visibility=0.2,
                                     label_fields=[]))


img_totsr = T.Compose([T.ToTensor(),
                       T.Normalize(mean=(0.485, 0.455, 0.406),
                                   std=(0.229, 0.224, 0.225))])

mask_totsr = ToTensorV2()


class TrainDs(Dataset):
    def __init__(self):
        lmdb_path = op.join(cfg.data_root, 'DocTamperV1-TrainingSet')
        self.lmdb = lmdb.open(lmdb_path, max_readers=64, readonly=True, lock=False, readahead=False, meminit=False)
        with self.lmdb.begin(write=False) as txn:
            self.sample_n = int(txn.get('num-samples'.encode('utf-8')))
        self.ocr_dir = op.join(cfg.ocr_root, 'TrainingSet/char_seg')
        self.qts = load_qt(cfg.qt_path)

        self.S = cfg.init_S
        self.T = cfg.cnt_per_epoch
        self.min_qf = cfg.min_qf
        self.ds_len = cfg.ds_len

        self.align_aug = get_align_aug()
        self.non_align_aug = get_non_align_aug()
        self.mask_totsr = mask_totsr
        self.img_totsr = img_totsr

    def __len__(self):
        return self.ds_len

    def __getitem__(self, _):
        with self.lmdb.begin(write=False) as lmdb:
            index = randint(0, self.sample_n - 1)
            img_name = '%06d' % index
            img, mask = load_data(index, lmdb)

        # load char seg
        char_seg_path = op.join(self.ocr_dir, img_name + '.pkl')
        if op.exists(char_seg_path):
            with open(char_seg_path, 'rb') as f:
                c_bbox = pickle.load(f)
        else:
            c_bbox = []

        img = np.array(img)

        if random() > 0.5:  # DCT grid align sample
            aug_func = self.align_aug
            is_align = True
        else:  # non-align sample
            aug_func = self.non_align_aug
            is_align = False

        aug_out = aug_func(image=img, mask=mask, bboxes=c_bbox)
        img, mask, c_bbox = aug_out['image'], aug_out['mask'], aug_out['bboxes']
        h, w = mask.shape
        ocr_mask = bbox_2_mask(c_bbox, h, w)
        img = Image.fromarray(img)

        min_qf = max(int(round(100 - (self.S / self.T))), 75)
        num_jpeg = randint(1, 3)

        dct, img, qfs = multi_jpeg(deepcopy(img),
                                   num_jpeg=num_jpeg,
                                   min_qf=min_qf,
                                   upper_bound=100)

        qf = qfs[-1]
        qt = self.qts[qf]
        img = self.img_totsr(img)
        mask = self.mask_totsr(image=mask.copy())['image']
        ocr_mask = self.mask_totsr(image=ocr_mask.copy())['image']

        return {
            'img': img,
            'dct': np.clip(np.abs(dct), 0, 20),
            'qt': qt,
            'mask': mask.long(),
            'ocr_mask': ocr_mask.long(),
            'img_name': img_name,
            'min_qf': min_qf,
            'is_align': is_align
        }


class DtdValDs(Dataset):
    def __init__(self, val_name, is_sample=False):
        lmdb_path = op.join(cfg.data_root, f'DocTamperV1-{val_name}')
        self.lmdb = lmdb.open(lmdb_path, max_readers=64, readonly=True, lock=False, readahead=False, meminit=False)
        with self.lmdb.begin(write=False) as txn:
            self.sample_n = int(txn.get('num-samples'.encode('utf-8')))
        if is_sample:
            self.sample_n = cfg.val_sample_n

        self.qts = load_qt(cfg.qt_path)
        self.ocr_dir = op.join(cfg.ocr_root, f'{val_name}/char_seg')
        self.jpeg_record = load_jpeg_record(op.join(cfg.jpeg_record_dir, f'DocTamperV1-{val_name}_{cfg.min_qf}.pk'))
        self.mask_totsr = mask_totsr
        self.img_totsr = img_totsr

    def __len__(self):
        return self.sample_n

    def __getitem__(self, index):
        with self.lmdb.begin(write=False) as lmdb:
            img_name = '%06d' % index
            img, mask = load_data(index, lmdb)
            h, w = mask.shape

        char_seg_path = op.join(self.ocr_dir, img_name + '.pkl')
        if op.exists(char_seg_path):
            with open(char_seg_path, 'rb') as f:
                c_bbox = pickle.load(f)
        else:
            c_bbox = []

        # augment
        if cfg.val_aug is not None:
            img = np.array(img)
            aug = cfg.val_aug(image=img, mask=mask, bboxes=c_bbox)
            img, mask, c_bbox = aug['image'], aug['mask'], aug['bboxes']
            h, w = mask.shape
            ocr_mask = bbox_2_mask(c_bbox, h, w)
            img = Image.fromarray(img)
        else:
            ocr_mask = bbox_2_mask(c_bbox, h, w)

        if cfg.shift_1p:
            img = np.array(img)
            img = np.roll(img, 1, axis=0)
            img = np.roll(img, 1, axis=1)
            img = Image.fromarray(img)
            mask = np.roll(mask, 1, axis=0)
            mask = np.roll(mask, 1, axis=1)
            ocr_mask = np.roll(ocr_mask, 1, axis=0)
            ocr_mask = np.roll(ocr_mask, 1, axis=1)

        if cfg.multi_jpeg_val:
            record = list(self.jpeg_record[index])
        else:
            if cfg.jpeg_record:
                record = cfg.jpeg_record
            else:
                record = [100]

        dct, img, qfs = multi_jpeg(deepcopy(img),
                                   num_jpeg=-1,
                                   min_qf=-1,
                                   upper_bound=-1,
                                   jpeg_record=record)

        qt = self.qts[qfs[-1]]
        img = self.img_totsr(img)
        ori_img = np.array(img)
        mask = self.mask_totsr(image=mask.copy())['image']
        ocr_mask = self.mask_totsr(image=ocr_mask.copy())['image']

        return {
            'img': img,
            'dct': np.clip(np.abs(dct), 0, 20),
            'qt': qt,
            'mask': mask.long(),
            'ocr_mask': ocr_mask.long(),
            'img_name': img_name,
            'ori_img': ori_img,
        }

def get_train_dl():
    ds = TrainDs()
    dl = DataLoader(dataset=ds,
                    batch_size=cfg.train_bs,
                    num_workers=4,
                    shuffle=True)
    return dl


def get_val_dl():
    dl_list = {}
    for val_name in cfg.val_name_list:
        is_sample = False
        if 'sample' in val_name:
            val_name = val_name.replace('_sample', '')
            is_sample = True
        ds = DtdValDs(val_name, is_sample)
        dl = DataLoader(dataset=ds,
                        batch_size=cfg.val_bs,
                        num_workers=4,
                        shuffle=False)
        dl_list[val_name] = dl
    return dl_list


if __name__ == '__main__':
    ds = TrainDs()
    from tqdm import tqdm

    for i in tqdm(range(50000)):
        tmp = ds.__getitem__(i)
        i = 0
    # ds = DtdValDs(roots='/data/jesonwong47/DocTamper/DocTamperV1/DocTamperV1-FCD',
    #               minq=75)
    # ds.__getitem__(0)
