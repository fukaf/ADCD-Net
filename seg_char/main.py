import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os.path as op
import cv2
import numpy as np
from copy import deepcopy
import pickle
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from CRAFT_pytorch import imgproc
from CRAFT_pytorch.craft import CRAFT
from CRAFT_pytorch.test import copyStateDict


class CharSeger:
    def __init__(self, ckpt_path, save_dir):
        self.model = self.load_CRAFT(ckpt_path)
        self.save_dir = save_dir
        self.save_mask_dir = op.join(save_dir, 'mask')
        self.save_bbox_dir = op.join(save_dir, 'bbox')
        os.makedirs(self.save_mask_dir, exist_ok=True)
        os.makedirs(self.save_bbox_dir, exist_ok=True)

        self.w_extend = 3
        self.h_extend = 3
        self.thresh = 0.6

    def load_CRAFT(self, ckpt_path):
        model = CRAFT()
        print('Loading weights from checkpoint (' + ckpt_path + ')')
        model.load_state_dict(copyStateDict(torch.load(ckpt_path)))
        model = model.cuda()
        cudnn.benchmark = False
        model.eval()
        return model

    def seg_char_per_img(self, img_path):
        img_name = op.basename(img_path).split('.')[0]
        img = cv2.imread(img_path)

        # resize
        resize_img, tgt_ratio, _ = imgproc.resize_aspect_ratio(img=img,
                                                               square_size=280,
                                                               interpolation=cv2.INTER_LINEAR,
                                                               mag_ratio=1.5)
        ratio = 1 / tgt_ratio * 2

        # preprocess
        x = imgproc.normalizeMeanVariance(resize_img)
        x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
        x = x.cuda()

        # forward pass
        with torch.no_grad():
            pred, _ = self.model(x)

        score_map = pred[0, :, :, 0].cpu().data.numpy()

        bin_map = cv2.threshold(score_map, self.thresh, 1, 0)[1]
        bin_map = bin_map * 255
        bin_map = bin_map.astype(np.uint8)

        contours, _ = cv2.findContours(bin_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        char_bbox_img = deepcopy(img)
        seg_char_list = []

        for contour in contours:
            contour = contour * ratio
            contour = contour.astype(np.int32)
            x, pred, w, h = cv2.boundingRect(contour)

            square_side = max(w, h)

            bbox = (int(x - (square_side / self.w_extend)),
                    int(pred - square_side * (1 / 8) - (square_side / self.w_extend)),
                    int(x + square_side + (square_side / self.w_extend)),
                    int(pred + square_side * (7 / 8) + (square_side / self.w_extend)))
            x1, y1, x2, y2 = bbox
            seg_char_list.append((x1, y1, x2, y2))
            cv2.rectangle(char_bbox_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # save bbox
        with open(op.join(self.save_bbox_dir, img_name + '.pkl'), 'wb') as f:
            pickle.dump(seg_char_list, f)
        # save heatmap
        cv2.imwrite(op.join(self.save_mask_dir, img_name + '.jpg'), char_bbox_img)


if __name__ == '__main__':
    ckpt_path = ''
    save_dir = ''
    img_path = ''

    char_seger = CharSeger(ckpt_path=ckpt_path,
                           save_dir=save_dir)

    char_seger.seg_char_per_img(img_path=img_path)
