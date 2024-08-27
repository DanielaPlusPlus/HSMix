"""
mixup based on superpixels
"""

import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from networks import A1115_UNet_binary_BraTS18         as SegNet

from metrics.Metrics_BraTS import dice_coef_cc, iou_coef_cc ,ppv,sensitivity


from skimage import segmentation
import copy
from scipy import stats
from data.dataset_BraTS18 import datasets
import torch
import torch.nn as nn
import datetime
import cv2
import sys


import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random


import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
# from sklearn.externals import joblib
from skimage.io import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms


from data.dataset_BraTS18 import Dataset
from skimage.io import imread, imsave
import imageio
from hausdorff import hausdorff_distance

torch.cuda.set_device(0)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: %s" % device)



def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='UNet',
                        help='model name')
    parser.add_argument('--mode', default='Calculate',
                        help='GetPicture or Calculate')
    parser.add_argument("--batch_size", type=int, default=1,
                        help='batch size (default: 24)')
    parser.add_argument("--h", type=int, default=224,
                        help='height of image')
    parser.add_argument("--w", type=int, default=224,
                        help='width of image')
    return parser

results_out_dir = './Results_out/'
if not os.path.exists(results_out_dir):
    os.makedirs(results_out_dir)

results_out_dir = './checkpoints/'
if not os.path.exists(results_out_dir):
    os.makedirs(results_out_dir)

def main():
    test_opts = get_argparser().parse_args()

    # Data loading code
    test_img_paths = glob(r'/home/dy/PycharmProjects/BraTS/data/BraTS2018_split/test/image/*')
    test_mask_paths = glob(r'/home/dy/PycharmProjects/BraTS/data/BraTS2018_split/test/mask/*')
    print("test_num:%s"%str(len(test_img_paths)))
    # Data Load
    test_dataset = Dataset(test_opts, test_img_paths, test_mask_paths)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_opts.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True)

    model_name = './checkpoints/A0313A_BraTS18_UNet_baseline_7307_6372.pth'
    model = SegNet.U_Net(img_ch=4)
    model.load_state_dict(torch.load(model_name))
    model = model.cuda()


    if test_opts.mode == "GetPicture":

        """
        获取并保存模型生成的标签图
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            with torch.no_grad():
                for i, (input, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
                    input = input.cuda()
                    #target = target.cuda()
                    output = model(input)
                    # output = F.sigmoid(output)
                    # # compute output
                    # if args.deepsupervision:
                    #     output = model(input)[-1]
                    # else:
                    #     output = model(input)
                    #print("img_paths[i]:%s" % img_paths[i])
                    # output = torch.sigmoid(output).data.cpu().numpy()
                    img_paths = test_img_paths[test_opts.batch_size*i:test_opts.batch_size*(i+1)]
                    #print("output_shape:%s"%str(output.shape))
                    # print(img_paths)


                    for j in range(output.shape[0]):
                        """
                        生成灰色圖片
                        wtName = os.path.basename(img_paths[i])
                        overNum = wtName.find(".npy")
                        wtName = wtName[0:overNum]
                        wtName = wtName + "_WT" + ".png"
                        imsave('output/%s/'%args.name + wtName, (output[i,0,:,:]*255).astype('uint8'))
                        tcName = os.path.basename(img_paths[i])
                        overNum = tcName.find(".npy")
                        tcName = tcName[0:overNum]
                        tcName = tcName + "_TC" + ".png"
                        imsave('output/%s/'%args.name + tcName, (output[i,1,:,:]*255).astype('uint8'))
                        etName = os.path.basename(img_paths[i])
                        overNum = etName.find(".npy")
                        etName = etName[0:overNum]
                        etName = etName + "_ET" + ".png"
                        imsave('output/%s/'%args.name + etName, (output[i,2,:,:]*255).astype('uint8'))
                        """
                        npName = os.path.basename(img_paths[j])
                        overNum = npName.find(".npy")
                        rgbName = npName[0:overNum]
                        rgbName = rgbName  + ".png"
                        # print(rgbName)
                        rgbPic = np.zeros([test_opts.h, test_opts.w, 3], dtype=np.uint8)
                        for idx in range(output.shape[2]):
                            for idy in range(output.shape[3]):
                                if output[j,0,idx,idy] > 0.5:
                                    rgbPic[idx, idy, 0] = 0
                                    rgbPic[idx, idy, 1] = 128
                                    rgbPic[idx, idy, 2] = 0
                                if output[j,1,idx,idy] > 0.5:
                                    rgbPic[idx, idy, 0] = 255
                                    rgbPic[idx, idy, 1] = 0
                                    rgbPic[idx, idy, 2] = 0
                                if output[j,2,idx,idy] > 0.5:
                                    rgbPic[idx, idy, 0] = 255
                                    rgbPic[idx, idy, 1] = 255
                                    rgbPic[idx, idy, 2] = 0

                        imsave('output/%s/'%test_opts.name + rgbName,rgbPic)

            torch.cuda.empty_cache()


        """
        将验证集中的GT numpy格式转换成图片格式并保存
        """
        print("Saving GT,numpy to picture")
        test_gt_path = 'output/%s/'%test_opts.name + "GT/"
        if not os.path.exists(test_gt_path):
            os.mkdir(test_gt_path)
        for idx in tqdm(range(len(test_mask_paths))):
            mask_path = test_mask_paths[idx]
            name = os.path.basename(mask_path)
            overNum = name.find(".npy")
            name = name[0:overNum]
            rgbName = name + ".png"

            npmask = np.load(mask_path)

            GtColor = np.zeros([npmask.shape[0],npmask.shape[1],3], dtype=np.uint8)
            for idx in range(npmask.shape[0]):
                for idy in range(npmask.shape[1]):
                    #坏疽(NET,non-enhancing tumor)(标签1) 红色
                    if npmask[idx, idy] == 1:
                        GtColor[idx, idy, 0] = 255
                        GtColor[idx, idy, 1] = 0
                        GtColor[idx, idy, 2] = 0
                    #浮肿区域(ED,peritumoral edema) (标签2) 绿色
                    elif npmask[idx, idy] == 2:
                        GtColor[idx, idy, 0] = 0
                        GtColor[idx, idy, 1] = 128
                        GtColor[idx, idy, 2] = 0
                    #增强肿瘤区域(ET,enhancing tumor)(标签4) 黄色
                    elif npmask[idx, idy] == 4:
                        GtColor[idx, idy, 0] = 255
                        GtColor[idx, idy, 1] = 255
                        GtColor[idx, idy, 2] = 0

            #imsave(val_gt_path + rgbName, GtColor)
            imageio.imwrite(test_gt_path + rgbName, GtColor)
            """
            mask_path = val_mask_paths[idx]
            name = os.path.basename(mask_path)
            overNum = name.find(".npy")
            name = name[0:overNum]
            wtName = name + "_WT" + ".png"
            tcName = name + "_TC" + ".png"
            etName = name + "_ET" + ".png"

            npmask = np.load(mask_path)

            WT_Label = npmask.copy()
            WT_Label[npmask == 1] = 1.
            WT_Label[npmask == 2] = 1.
            WT_Label[npmask == 4] = 1.
            TC_Label = npmask.copy()
            TC_Label[npmask == 1] = 1.
            TC_Label[npmask == 2] = 0.
            TC_Label[npmask == 4] = 1.
            ET_Label = npmask.copy()
            ET_Label[npmask == 1] = 0.
            ET_Label[npmask == 2] = 0.
            ET_Label[npmask == 4] = 1.

            imsave(val_gt_path + wtName, (WT_Label * 255).astype('uint8'))
            imsave(val_gt_path + tcName, (TC_Label * 255).astype('uint8'))
            imsave(val_gt_path + etName, (ET_Label * 255).astype('uint8'))
            """
        print("Save Prediction and GT images Done!")



    if test_opts.mode == "Calculate":
        """
        计算各种指标:Dice, IoU, Sensitivity, PPV, HD
        """
        wt_dices = []
        tc_dices = []
        et_dices = []
        wt_ious = []
        tc_ious = []
        et_ious = []
        wt_sensitivities = []
        tc_sensitivities = []
        et_sensitivities = []
        wt_ppvs = []
        tc_ppvs = []
        et_ppvs = []
        wt_Hausdorf = []
        tc_Hausdorf = []
        et_Hausdorf = []

        maskPath = glob("output/%s/" % test_opts.name + "GT/*.png")
        pbPath = glob("output/%s/" % test_opts.name + "*.png")
        if len(maskPath) == 0:
            print("请先生成图片!")
            return

        for myi in tqdm(range(len(maskPath))):
            mask = imread(maskPath[myi])
            pb = imread(pbPath[myi])

            wtmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            wtpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)

            tcmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            tcpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)

            etmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            etpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)

            for idx in range(mask.shape[0]):
                for idy in range(mask.shape[1]):
                    # 只要这个像素的任何一个通道有值,就代表这个像素不属于前景,即属于WT区域
                    if mask[idx, idy, :].any() != 0:
                        wtmaskregion[idx, idy] = 1
                    if pb[idx, idy, :].any() != 0:
                        wtpbregion[idx, idy] = 1
                    # 只要第一个通道是255,即可判断是TC区域,因为红色和黄色的第一个通道都是255,区别于绿色
                    if mask[idx, idy, 0] == 255:
                        tcmaskregion[idx, idy] = 1
                    if pb[idx, idy, 0] == 255:
                        tcpbregion[idx, idy] = 1
                    # 只要第二个通道是128,即可判断是ET区域
                    if mask[idx, idy, 1] == 128:
                        etmaskregion[idx, idy] = 1
                    if pb[idx, idy, 1] == 128:
                        etpbregion[idx, idy] = 1
            # 开始计算WT
            dice = dice_coef_cc(wtpbregion, wtmaskregion)
            iou = iou_coef_cc(wtpbregion, wtmaskregion)
            wt_dices.append(dice)
            wt_ious.append(iou)
            ppv_n = ppv(wtpbregion, wtmaskregion)
            wt_ppvs.append(ppv_n)
            Hausdorff = hausdorff_distance(wtmaskregion, wtpbregion)
            wt_Hausdorf.append(Hausdorff)
            sensitivity_n = sensitivity(wtpbregion, wtmaskregion)
            wt_sensitivities.append(sensitivity_n)
            # 开始计算TC
            dice = dice_coef_cc(tcpbregion, tcmaskregion)
            iou = iou_coef_cc(tcpbregion, tcmaskregion)
            tc_dices.append(dice)
            tc_ious.append(iou)
            ppv_n = ppv(tcpbregion, tcmaskregion)
            tc_ppvs.append(ppv_n)
            Hausdorff = hausdorff_distance(tcmaskregion, tcpbregion)
            tc_Hausdorf.append(Hausdorff)
            sensitivity_n = sensitivity(tcpbregion, tcmaskregion)
            tc_sensitivities.append(sensitivity_n)
            # 开始计算ET
            dice = dice_coef_cc(etpbregion, etmaskregion)
            iou = iou_coef_cc(etpbregion, etmaskregion)
            et_dices.append(dice)
            et_ious.append(iou)
            ppv_n = ppv(etpbregion, etmaskregion)
            et_ppvs.append(ppv_n)
            Hausdorff = hausdorff_distance(etmaskregion, etpbregion)
            et_Hausdorf.append(Hausdorff)
            sensitivity_n = sensitivity(etpbregion, etmaskregion)
            et_sensitivities.append(sensitivity_n)

        print('WT Dice: %.4f' % np.mean(wt_dices))
        print('TC Dice: %.4f' % np.mean(tc_dices))
        print('ET Dice: %.4f' % np.mean(et_dices))
        print('Avg. Dice: %.4f' % np.mean(np.mean(wt_dices),np.mean(tc_dices),np.mean(et_dices)))
        print("=============")
        print('WT IoU: %.4f' % np.mean(wt_ious))
        print('TC IoU: %.4f' % np.mean(tc_ious))
        print('ET IoU: %.4f' % np.mean(et_ious))
        print('Avg. IoU: %.4f' % np.mean(np.mean(wt_ious),np.mean(tc_ious),np.mean(et_ious)))
        print("=============")
        print('WT PPV: %.4f' % np.mean(wt_ppvs))
        print('TC PPV: %.4f' % np.mean(tc_ppvs))
        print('ET PPV: %.4f' % np.mean(et_ppvs))
        print("=============")
        print('WT sensitivity: %.4f' % np.mean(wt_sensitivities))
        print('TC sensitivity: %.4f' % np.mean(tc_sensitivities))
        print('ET sensitivity: %.4f' % np.mean(et_sensitivities))
        print("=============")
        print('WT Hausdorff: %.4f' % np.mean(wt_Hausdorf))
        print('TC Hausdorff: %.4f' % np.mean(tc_Hausdorf))
        print('ET Hausdorff: %.4f' % np.mean(et_Hausdorf))
        print("=============")

if __name__ == '__main__':
    main()
