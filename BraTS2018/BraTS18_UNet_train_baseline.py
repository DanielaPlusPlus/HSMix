"""
WT Dice: 0.8436
TC Dice: 0.8022
ET Dice: 0.7235
=============
WT IoU: 0.7758
TC IoU: 0.7496
ET IoU: 0.6207
=============
WT PPV: 0.8776
TC PPV: 0.8861
ET PPV: 0.7426
=============
WT sensitivity: 0.8606
TC sensitivity: 0.8337
ET sensitivity: 0.7840
=============
WT Hausdorff: 2.6254
TC Hausdorff: 1.9212
ET Hausdorff: 2.9762
"""

import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from networks import A1115_UNet_binary_BraTS18         as SegNet


from metrics.stream_metrics import StreamSegMetrics, dice_binary_class, IoU_binary_class
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
from losses.DiceLoss import BinaryDiceLoss

torch.cuda.set_device(0)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: %s" % device)

def make_print_to_file(path='./'):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    import sys
    import os
    import sys
    import datetime

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = datetime.datetime.now().strftime('DAY' + '%Y_%m_%d_')
    sys.stdout = Logger(
        fileName + 'A0313A_BraTS18_UNet_baseline.log', path=path)
make_print_to_file(path='./logs')

def get_argparser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data", type=str, default="complete",
    #                     help="Label data type.")
    # parser.add_argument("--img_root", type=str, default="./data/BraTS2018_split/train/image",
    #                     help="The directory containing the training image dataset.")
    # parser.add_argument("--label_root", type=str, default="./data/BraTS2018_split/train/mask",
    #                     help="The directory containing the training label datgaset")
    # Train Options
    parser.add_argument("--RESUME", type=bool, default=False)
    parser.add_argument("--START_EPOCH", type=int, default=0,
                        help="epoch number (default: 30k)")
    parser.add_argument("--NB_EPOCH", type=int, default=70,
                        help="epoch number (default: 30k)")
    parser.add_argument("--warmup_itrs", type=int, default=5,
                        help="epoch number (default: 1k)")
    parser.add_argument("--LR", type=float, default=0.001,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=True, #修改！！！
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch size (default: 24)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')


    parser.add_argument("--gpu_id", type=str, default='0',help="GPU ID")
    parser.add_argument("--gpu_ids", type=list, default=[0],help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1234,
                        help="random seed (default: 1)")



    return parser

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def validate(model, loader, device):
    model.eval()
    with torch.no_grad():
        iou, dice, count = 0, 0, 0
        for i, (input, target) in tqdm(enumerate(loader), total=len(loader)):

            images = input
            labels = target

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            count += images.shape[0]

            outputs = model(images)
            outputs = F.sigmoid(outputs)

            dice += dice_binary_class(outputs, labels)
            iou += IoU_binary_class(outputs, labels)

        dice = dice / count
        iou = iou / count
        # score = metrics.get_results()

    return iou, dice



results_out_dir = './Results_out/'
if not os.path.exists(results_out_dir):
    os.makedirs(results_out_dir)

results_out_dir = './checkpoints/'
if not os.path.exists(results_out_dir):
    os.makedirs(results_out_dir)

def main():
    opts = get_argparser().parse_args()

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Data loading code
    img_paths = glob(r'/home/dy/PycharmProjects/BraTS/data/BraTS2018_split/train/image/*')
    mask_paths = glob(r'/home/dy/PycharmProjects/BraTS/data/BraTS2018_split/train/mask/*')

    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)
    print("train_num:%s"%str(len(train_img_paths)))
    print("val_num:%s"%str(len(val_img_paths)))


    # Data Load
    train_dataset = Dataset(opts, train_img_paths, train_mask_paths)
    val_dataset = Dataset(opts, val_img_paths, val_mask_paths)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    model_name = './checkpoints/A0313A_BraTS18_UNet_baseline.pth'
    if len(opts.gpu_ids) > 1:
        if opts.RESUME:
            model = SegNet.U_Net(img_ch=4)
            model.load_state_dict(torch.load(model_name))
            model = model.cuda()
        else:
            model = SegNet.U_Net(img_ch=4)
            model = model.cuda()
        model = torch.nn.DataParallel(model)
    else:
        if opts.RESUME:
            model = SegNet.U_Net(img_ch=4)
            model.load_state_dict(torch.load(model_name))
            model = model.cuda()
        else:
            model = SegNet.U_Net(img_ch=4)
            model = model.cuda()

    criterion_seg =  nn.BCELoss(reduction='mean')
    criterion_dice = BinaryDiceLoss()

    tm = datetime.datetime.now().strftime('T' + '%m%d%H%M')
    results_file_name = results_out_dir + tm + 'A0313A_BraTS18_UNet_baseline_results.txt'

    best_dice = 0.0
    for epoch in range(opts.START_EPOCH, opts.NB_EPOCH):
        if epoch <= 4:
            if opts.LR >= 0.00001:
                warmup_lr = opts.LR * ((epoch + 1) / 5)
                lr = warmup_lr
            else:
                lr = opts.LR
        elif 4 < epoch <= 19:
            lr = opts.LR
        elif 19 < epoch <= 39:  # 50
            lr = opts.LR / 10
        elif 39 < epoch <= 59:  # 40
            lr = opts.LR / 100
        elif 59 < epoch <= 70:  # 30
            lr = opts.LR / 1000
        print("current epoch:", epoch, "current lr:", lr)

        optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=opts.weight_decay)

        list_loss, list_loss_seg, list_loss_dice, list_loss_mse_a, list_loss_mse_b, list_loss_local = [], [], [], [], [], []
        for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            images = input
            labels = target

            optimizer.zero_grad()

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            model.train()
            out_seg = model(images)
            out_seg = F.sigmoid(out_seg)
            loss_seg = criterion_seg(out_seg, labels)
            loss_dice = criterion_dice(out_seg, labels)
            loss = loss_seg + loss_dice

            list_loss.append(loss)
            list_loss_seg.append(loss_seg)
            list_loss_dice.append(loss_dice)

            loss.backward()
            optimizer.step()

        iou, dice = validate(model=model, loader=val_loader, device=device)
        if len(list_loss) > 0:
            list_loss = torch.stack(list_loss).mean()
        else:
            list_loss = 0

        if len(list_loss_seg) > 0:
            list_loss_seg = torch.stack(list_loss_seg).mean()
        else:
            list_loss_seg = 0

        if len(list_loss_dice) > 0:
            list_loss_dice = torch.stack(list_loss_dice).mean()
        else:
            list_loss_dice = 0

        if len(list_loss_mse_a) > 0:
            list_loss_mse_a = torch.stack(list_loss_mse_a).mean()
        else:
            list_loss_mse_a = 0

        if len(list_loss_mse_b) > 0:
            list_loss_mse_b = torch.stack(list_loss_mse_b).mean()
        else:
            list_loss_mse_b = 0

        if len(list_loss_local) > 0:
            list_loss_local = torch.stack(list_loss_local).mean()
        else:
            list_loss_local = 0

        if dice > best_dice:  # save best model
            best_dice, best_iou = dice, iou
            if len(opts.gpu_ids) > 1:
                torch.save(model.module.state_dict(), model_name)
            else:
                torch.save(model.state_dict(), model_name)
            print(
                "Epoch %d, Loss=%f, Loss_seg=%f, Loss_dice=%f,Loss_MSE_A=%f, Loss_MSE_B=%f,Loss_LOCAL=%f, mIoU=%f, Dice=%f" %
                (
                epoch, list_loss, list_loss_seg, list_loss_dice, list_loss_mse_a, list_loss_mse_b, list_loss_local, iou,
                dice))
        with open(results_file_name, 'a') as file:
            file.write(
                'Epoch %d, Loss=%f, Loss_seg=%f, Loss_dice=%f,Loss_MSE_A=%f, Loss_MSE_B=%f,Loss_LOCAL=%f, mIoU=%f, Dice=%f \n ' % (
                    epoch, list_loss, list_loss_seg, list_loss_dice, list_loss_mse_a, list_loss_mse_b, list_loss_local,
                    iou, dice))

    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
