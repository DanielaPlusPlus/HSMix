
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from networks.A1220_TransUnet_vit_seg_modeling_binary_base import VisionTransformer as ViT_seg
from networks.A1220_TransUnet_vit_seg_modeling_binary_base import CONFIGS as Configs_ViT_Seg
import utils
import os
import random
import argparse
import numpy as np


from torch.utils.data import DataLoader
from skimage import segmentation
import copy
from scipy import stats
from datasets.Glas_dataset import RandomGenerator,ValGenerator,ImageToImage2D_kfold
from utils import ext_transforms as et
from metrics.stream_metrics import StreamSegMetrics, dice_binary_class, IoU_binary_class, IoU_multi_class, dice_multi_class,iou_on_batch

import torch
import torch.nn as nn
import datetime
from losses.DiceLoss import BinaryDiceLoss

from sklearn.model_selection import KFold
torch.cuda.set_device(0)

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
        fileName + 'A0313_MoNuSeg_TransUnet_SuperpixelCutMixMixup_2Mixup_2Aug.log', path=path)
make_print_to_file(path='./logs')

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='MoNuSeg',
                        choices=[ 'ISIC2018T1', 'MoNuSeg', 'ISIC2017T1'], help='Name of dataset')
    parser.add_argument("--train_dataset", type=str, default='./datasets/data/MoNuSeg/train',
                        help="path to Dataset")
    parser.add_argument("--test_dataset", type=str, default='./datasets/data/MoNuSeg/test',
                        help="path to Dataset")


    # Train Options
    parser.add_argument("--RESUME", type=bool, default=False)
    parser.add_argument("--START_EPOCH", type=int, default=0,
                        help="epoch number (default: 30k)")
    parser.add_argument("--NB_EPOCH", type=int, default=420,
                        help="epoch number (default: 30k)")
    parser.add_argument("--warmup_itrs", type=int, default=5,
                        help="epoch number (default: 1k)")
    parser.add_argument("--LR", type=float, default=0.002,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--crop_val", action='store_true', default=True, 
    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch size (default: 4)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--kfold", type=int, default=5)

    parser.add_argument("--loss_type", type=str, default='BCE',
                        choices=['cross_entropy', 'focal_loss', 'BCE'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--gpu_ids", type=list, default=[0])
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1234,
                        help="random seed (default: 1)")
 ########################################
    parser.add_argument("--Beta_mixup", type=float, default=1)
    parser.add_argument("--Beta_cutmix", type=float, default=0.5)
    parser.add_argument("--cutmix_prob", type=float, default=1)
    parser.add_argument("--N_min", type=int, default=300)
    parser.add_argument("--N_max", type=int, default=500)

    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")
    parser.add_argument("--model", type=str, default='Unet',help='model name')

    #
    # parser.add_argument("--loss_weight_seg", type=float, default=1)
    # # parser.add_argument("--loss_weight_dice", type=float, default=0.0)
    # parser.add_argument("--loss_weight_local", type=float, default=0.2)
    # parser.add_argument("--temp_local_contrast", type=float, default=0.07)
    # parser.add_argument("--kappa", type=float, default=8)

    return parser

def rand_bnl(size, p_binom=0.5):
    brl = stats.bernoulli.rvs(p_binom, size=size, random_state=None)  # random_state=None指每次生成随机
    # print(brl)
    (zero_idx,) = np.where(brl == int(1))
    return zero_idx

def SuperpixelMixup_LambdaMask(images, labels, N_superpixels_mim, N_superpixels_max, Beta_mixup, Beta_cutmix):
    bsz, C, W, H = images.shape
    mixup_mask_3d, cutmix_mask_3d, lb_batch_mixup, lb_batch_cutmix = [], [], [], []
    rand_index = torch.randperm(bsz)
    img_a, img_b = images, images[rand_index]
    lb_a, lb_b = labels, labels[rand_index]
    for sp in range(bsz):
        """superpixel for image B"""
        img_seg_b = img_b[sp].reshape(W, H, -1)  # (W,H,C)
        N_b = random.randint(N_superpixels_mim, N_superpixels_max)
        SuperP_map_b = segmentation.slic(img_seg_b.cpu().numpy(), n_segments=N_b, compactness=8, start_label=1)
        SuperP_map_b_value = np.unique(SuperP_map_b)

        """superpixel-mixup with random lam """
        # binary_mask_sp_mixup = np.zeros((W, H)) # for superpixel genertion
        lam_mixup_mask_sp = np.zeros((W, H), dtype=np.float32) # for image and mask mixup
        for v in range(SuperP_map_b_value.shape[0]):
            bool_v = (SuperP_map_b == SuperP_map_b_value[v])
            lam = np.random.beta(Beta_mixup, Beta_mixup)
            lam_mixup_mask_sp[bool_v == True] = lam

        lam_mixup_mask_sp = torch.tensor(lam_mixup_mask_sp)
        labels_sp_mixup =  lb_a[sp] * (1- lam_mixup_mask_sp) + lb_b[sp] * lam_mixup_mask_sp
        lb_batch_mixup.append(labels_sp_mixup)

        lam_mixup_mask_ch_sp = copy.deepcopy(lam_mixup_mask_sp)
        lam_mixup_mask_ch_sp = lam_mixup_mask_ch_sp.expand(C, -1, -1)  # torch.Size([3, 32, 32])
        mixup_mask_3d.append(lam_mixup_mask_ch_sp)

        """superpixel-cutmix with random selection"""
        SuperP_map_b_value = np.unique(SuperP_map_b)
        sel_region_idx_cutmix = rand_bnl(p_binom=Beta_cutmix, size=SuperP_map_b_value.shape[0])
        binary_mask_sp_cutmix = np.zeros((W, H))
        for v in range(SuperP_map_b_value.shape[0]):
            if v in sel_region_idx_cutmix:
                bool_v = (SuperP_map_b == SuperP_map_b_value[v])
                binary_mask_sp_cutmix[bool_v == True] = 1  
            else:
                pass
        labels_sp_cutmix = lb_a[sp] * (1 - binary_mask_sp_cutmix) + lb_b[sp] * binary_mask_sp_cutmix
        lb_batch_cutmix.append(labels_sp_cutmix)

        binary_mask_sp_cutmix = torch.tensor(binary_mask_sp_cutmix)
        binary_mask_ch_sp_cutmix = copy.deepcopy(binary_mask_sp_cutmix)
        binary_mask_ch_sp_cutmix = binary_mask_ch_sp_cutmix.expand(C, -1, -1)  # torch.Size([3, 32, 32])
        cutmix_mask_3d.append(binary_mask_ch_sp_cutmix)

    lb_batch_mixup = torch.stack(lb_batch_mixup)
    mixup_mask = torch.stack(mixup_mask_3d)
    mixup_mask = mixup_mask.float()
    images_mixup = img_a * (1- mixup_mask) + img_b * mixup_mask   ### genarate the final mixup image

    lb_batch_cutmix = torch.stack(lb_batch_cutmix)
    cutmix_mask = torch.stack(cutmix_mask_3d)
    cutmix_mask = cutmix_mask.float()
    images_cutmix = img_a * (1- cutmix_mask) + img_b * cutmix_mask   ### genarate the final cutmix image


    return images_cutmix, images_mixup, lb_batch_cutmix, lb_batch_mixup


def validate(model, loader, device):
    model.eval()
    with torch.no_grad():
        iou, dice, count = 0, 0, 0
        for i, (samples,_) in tqdm(enumerate(loader)):

            images = samples['image'].to(device, dtype=torch.float32)
            labels = samples['label'].to(device, dtype=torch.long)

            count += images.shape[0]
            outputs = model(images)
            dice += dice_binary_class(outputs, labels)
            iou += IoU_binary_class(outputs, labels)

        dice = dice / count
        iou = iou / count
    return iou, dice


results_out_dir = './Results_out/'
if not os.path.exists(results_out_dir):
    os.makedirs(results_out_dir)

def main():
    opts = get_argparser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    if opts.dataset == "Synapse":
        filelists = os.listdir(opts.train_dataset)
    else:
        filelists = os.listdir(opts.train_dataset+"/img")
    filelists = np.array(filelists)
    kfold = opts.kfold
    kf = KFold(n_splits=kfold, shuffle=True, random_state=opts.random_seed)

    for fold, (train_index, val_index) in enumerate(kf.split(filelists)):
        train_filelists = filelists[train_index]
        val_filelists = filelists[val_index]
        print("Fold: {}, Total Nums: {}, train: {}, val: {}".format(fold, len(filelists), len(train_filelists),
                                                                    len(val_filelists)))

        train_tf = RandomGenerator(output_size=[opts.img_size, opts.img_size])
        val_tf = ValGenerator(output_size=[opts.img_size, opts.img_size])
        train_dataset = ImageToImage2D_kfold(opts.train_dataset,
                                             train_tf,
                                             image_size=opts.img_size,
                                             filelists=train_filelists,
                                             task_name=opts.dataset)
        val_dataset = ImageToImage2D_kfold(opts.train_dataset,
                                           val_tf,
                                           image_size=opts.img_size,
                                           filelists=val_filelists,
                                           task_name=opts.dataset)
        train_loader = DataLoader(train_dataset,
                                  batch_size=opts.batch_size,
                                  shuffle=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=opts.val_batch_size,
                                shuffle=True)

        model_name = './checkpoints/A0313_MoNuSeg_TransUnet_SuperpixelCutMixMixup_2Mixup_2Aug_%sFold.pth' %  (fold)
        config_vit = Configs_ViT_Seg['R50-ViT-B_16']
        config_vit.n_classes = 1
        config_vit.n_skip = 3

        config_vit.patches.grid = (int(224 / 16), int(224 / 16))
        if len(opts.gpu_ids) > 1:
            if opts.RESUME:
                model = ViT_seg(config_vit, img_size=224, num_classes=1).cuda()
                model.load_from(weights=np.load(config_vit.pretrained_path))
                model.load_state_dict(torch.load(model_name))
            else:
                model = ViT_seg(config_vit, img_size=224, num_classes=1).cuda()
                model.load_from(weights=np.load(config_vit.pretrained_path))
            model = torch.nn.DataParallel(model)
        else:
            if opts.RESUME:
                model = ViT_seg(config_vit, img_size=224, num_classes=1).cuda()
                model.load_from(weights=np.load(config_vit.pretrained_path))
                model.load_state_dict(torch.load(model_name))
            else:
                model = ViT_seg(config_vit, img_size=224, num_classes=1).cuda()
                model.load_from(weights=np.load(config_vit.pretrained_path))

        if opts.loss_type == 'focal_loss':
            criterion_seg = utils.FocalLoss(ignore_index=255, size_average=True)
        elif opts.loss_type == 'cross_entropy':
            criterion_seg = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        elif opts.loss_type == 'BCE':
            criterion_seg = nn.BCELoss(reduction='mean')
        criterion_dice = BinaryDiceLoss()


        tm = datetime.datetime.now().strftime('T' + '%m%d%H%M')
        results_file_name = results_out_dir + tm + 'A0313_MoNuSeg_TransUnet_SuperpixelCutMixMixup_2Mixup_2Aug_%sFold_results.txt' %  (fold)

        best_dice, best_iou = 0, 0
        for epoch in range(opts.START_EPOCH, opts.NB_EPOCH):
            if epoch <= 4:
                if opts.LR >= 1:
                    warmup_lr = opts.LR * ((epoch + 1) / 5)
                    lr = warmup_lr
                else:
                    lr = opts.LR
            elif 4 < epoch <= 149:
                lr = opts.LR
            elif 149 < epoch <= 199:  # 50
                lr = opts.LR / 2
            elif 199 < epoch <= 249:  # 40
                lr = opts.LR / 4
            elif 249 < epoch <= 279:  # 30
                lr = opts.LR / 8
            elif 279 < epoch <= 309:  # 40
                lr = opts.LR / 10
            elif 309 < epoch <= 329:  # 40
                lr = opts.LR / 20
            elif 329 < epoch <= 349:  # 40
                lr = opts.LR / 50
            elif 349 < epoch <= 369:  # 40
                lr = opts.LR / 80
            elif 369 < epoch <= 399:  # 40
                lr = opts.LR / 100
            elif 399 < epoch <= 420:  # 40
                lr = opts.LR / 1000
            # print("current epoch:", epoch, "current lr:", lr)

            optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=opts.weight_decay)

            list_loss, list_loss_seg, list_loss_dice, list_loss_mse_a, list_loss_mse_b, list_loss_local = [], [], [], [], [], []
            for i, (samples, _) in tqdm(enumerate(train_loader)):
                images = samples['image']
                labels = samples['label']

                r = np.random.rand(1) 
                if r < opts.cutmix_prob:
                    # rand_index = torch.randperm(images.shape[0]).cuda()
                    # img_a, img_b = images, images[rand_index]
                    # lb_a, lb_b = labels, labels[rand_index]
                    optimizer.zero_grad()

                    images_SuperPCutMix, images_SuperPMixup, labels_SuperPCutMix, labels_SuperPMixup = \
                        SuperpixelMixup_LambdaMask(images, labels, opts.N_min, opts.N_max, Beta_mixup=opts.Beta_mixup, Beta_cutmix=opts.Beta_cutmix)

                    imgs_aug = torch.cat((images_SuperPCutMix, images_SuperPMixup), dim=0)
                    lbs_aug = torch.cat((labels_SuperPCutMix, labels_SuperPMixup), dim=0)

                    imgs_aug = imgs_aug.to(device, dtype=torch.float32)
                    lbs_aug = lbs_aug.to(device, dtype=torch.float32)
                    model.train()
                    out_seg = model(imgs_aug)
                    loss_seg = criterion_seg(out_seg, lbs_aug)
                    loss_dice = criterion_dice(out_seg, lbs_aug)
                    loss = loss_seg + loss_dice

                    list_loss.append(loss)
                    list_loss_seg.append(loss_seg)
                    list_loss_dice.append(loss_dice)

                    # print("loss: ", loss)
                    # print("loss_seg: ", loss_seg)
                    # print("loss_dice: ",  + loss_dice)

                else:
                    optimizer.zero_grad()
                    model.train()
                    images = images.to(device, dtype=torch.float32)
                    labels = labels.to(device, dtype=torch.float32)
                    outputs = model(images)
                    loss_seg = criterion_seg(outputs, labels)
                    loss_dice = criterion_dice(outputs, labels)
                    # loss_dice, loss_MSE = torch.zeros(1).cuda(), torch.zeros(1).cuda()
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
                    "Fold %d, Epoch %d, Loss=%f, Loss_seg=%f, Loss_dice=%f,Loss_MSE_A=%f, Loss_MSE_B=%f,Loss_LOCAL=%f, mIoU=%f, Dice=%f" %
                    (
                    fold, epoch, list_loss, list_loss_seg, list_loss_dice, list_loss_mse_a, list_loss_mse_b, list_loss_local, iou,
                    dice))
            with open(results_file_name, 'a') as file:
                file.write(
                    'Fold %d,Epoch %d, Loss=%f, Loss_seg=%f, Loss_dice=%f,Loss_MSE_A=%f, Loss_MSE_B=%f,Loss_LOCAL=%f, mIoU=%f, Dice=%f \n ' % (
                        fold, epoch, list_loss, list_loss_seg, list_loss_dice, list_loss_mse_a, list_loss_mse_b, list_loss_local,
                        iou, dice))

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
