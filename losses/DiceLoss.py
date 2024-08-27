import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        # 获取每个批次的大小 N
        N = targets.size()[0]
        # 平滑变量
        smooth = 1
        # 将宽高 reshape 到同一纬度
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)
        # 计算交集
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        # 计算一个批次中平均每张图的损失
        loss = 1 - N_dice_eff.sum() / N
        return loss


class MultiClassDiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(MultiClassDiceLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.kwargs = kwargs
    def forward(self, input, target):
        """
        input tesor of shape = (N, C, H, W)
        target tensor of shape = (N, H, W)
        """
        # 先将 target 进行 one-hot 处理，转换为 (N, C, H, W)
        # target[target==255.] = 0.
        nclass = input.shape[1]
        target = F.one_hot(target.long(), nclass)
        target = target.reshape(input.shape[0],input.shape[1],input.shape[2],-1)

        assert input.shape == target.shape, "predict & target shape do not match"
        binaryDiceLoss = BinaryDiceLoss()
        total_loss = 0
        # 归一化输出
        logits = F.softmax(input, dim=1)
        C = target.shape[1]
        # 遍历 channel，得到每个类别的二分类 DiceLoss
        for i in range(C):
            dice_loss = binaryDiceLoss(logits[:, i], target[:, i])
            total_loss += dice_loss
            # 每个类别的平均 dice_loss
        return total_loss / C