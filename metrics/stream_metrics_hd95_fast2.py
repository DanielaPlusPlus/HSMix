import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from sklearn.metrics import roc_auc_score,jaccard_score
import numpy as np
# import scipy
# from scipy.spatial.distance import directed_hausdorff
from medpy import metric

class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()      

class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )
    
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %f\n"%(k, v)
        
        #string+='Class IoU:\n'
        #for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

class AverageMeter(object):
    """Computes average values"""
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()
    
    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0]+=val
            record[1]+=1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]

def dice_coef(y_true, y_pred):
    """

    :param y_true: arbitraty shape, binary
    :param y_pred: arbitraty shape, binary
    :return: dice of two masks in binary
    """
    smooth = 1e-5
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    # print(y_true_f.shape, y_pred_f.shape)
    intersection = np.sum(y_true_f * y_pred_f)

    # a = (2. * intersection + smooth)
    # b = (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    # print(np.sum(y_true_f * y_pred_f), np.sum(y_true_f), np.sum(y_pred_f), np.sum(y_true_f)+np.sum(y_pred_f))
    # print("dice-------------------------------:", (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth))
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)




def dice_multi_class(decoded_fea, target):
    """
    :param decoded_fea: [batch_size, num_class, h,w], final decoded feature
    :param target: [batch_size, h,w]
    :return: average dice across the classes in a batch(sum of each sample in a batch)
    """
    out = torch.argmax(decoded_fea, dim=1)
    # print(out.shape)
    classes = decoded_fea.shape[1]
    metric_batch = 0
    for i in range(decoded_fea.shape[0]):
        pred_sp = out[i].cpu().detach().numpy()
        tg_sp = target[i].cpu().detach().numpy()
        # print(tg_sp)
        metric_class = []
        # print(pred_tp.shape, tg_tp.shape)
        for j in range(0, classes):
            pred_tp = np.zeros_like(pred_sp)
            tg_tp = np.zeros_like(tg_sp)
            bool_pred_j = (pred_sp == j)
            bool_tg_j = (tg_sp == j)
            pred_tp[bool_pred_j==True] = 1
            tg_tp[bool_tg_j==True] = 1

            # dice_class.append(dice_coef(pred_tp, tg_tp))
            if np.sum(pred_tp)>0 or np.sum(tg_tp)>0:  #only when there's corresponding class in GT, we make average across this class; otherwise, pass
                dice = dice_coef(tg_tp , pred_tp)
                metric_class.append(dice)

        # print(metric_class)
        metric_batch += (np.mean(metric_class))
    # print(metric_batch)
    return metric_batch


def dice_binary_class(decoded_fea, target):
    """
    :param decoded_fea: [batch_size, h,w], final decoded feature after sigmoid function
    :param target: [batch_size, h,w]
    :return: sum of dice in a batch(sum of each sample in a batch)
    """
    metric_batch = 0
    for i in range(decoded_fea.shape[0]):
        pred_sp = decoded_fea[i].cpu().detach().numpy()
        tg_sp = target[i].cpu().detach().numpy()
        # print(tg_sp)
        pred_tp = np.zeros_like(pred_sp)
        tg_tp = np.zeros_like(tg_sp)
        bool_pred_j = (pred_sp >= 0.5)
        bool_tg_j = (tg_sp == 1)
        pred_tp[bool_pred_j==True] = 1
        tg_tp[bool_tg_j==True] = 1

        # dice_class.append(dice_coef(pred_tp, tg_tp))
        # if np.sum(pred_tp)>0 or np.sum(tg_tp)>0:  #only when there's corresponding class in GT, we make average across this class; otherwise, pass
        #     dice = dice_coef(tg_tp , pred_tp)
        #     metric_batch += dice
        # if np.sum(pred_tp)==0 and np.sum(tg_tp)==0:  #only when there's corresponding class in GT, we make average across this class; otherwise, pass
        #     dice = 1
        #     metric_batch += dice

        dice = dice_coef(tg_tp , pred_tp)
        metric_batch += dice

    return metric_batch







def IoU(y_true, y_pred):
    """

    :param y_true: arbitraty shape, binary
    :param y_pred: arbitraty shape, binary
    :return: dice of two masks in binary
    """
    smooth = 1e-5
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    # print(y_true_f.shape, y_pred_f.shape)
    intersection = np.sum(y_true_f * y_pred_f)

    # a = (2. * intersection + smooth)
    # b = (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    # print(np.sum(y_true_f * y_pred_f), np.sum(y_true_f), np.sum(y_pred_f), np.sum(y_true_f)+np.sum(y_pred_f))
    # print("dice-------------------------------:", (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth))
    return (intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection + smooth)




def IoU_multi_class(decoded_fea, target):
    """
    :param decoded_fea: [batch_size, num_class, h,w], final decoded feature
    :param target: [batch_size, h,w]
    :return: average dice across the classes in a batch(sum of each sample in a batch)
    """
    out = torch.argmax(decoded_fea, dim=1)
    # print(out.shape)
    classes = decoded_fea.shape[1]
    metric_batch = 0
    for i in range(decoded_fea.shape[0]):
        pred_sp = out[i].cpu().detach().numpy()
        tg_sp = target[i].cpu().detach().numpy()
        # print(tg_sp)
        metric_class = []
        # print(pred_tp.shape, tg_tp.shape)
        for j in range(0, classes):
            pred_tp = np.zeros_like(pred_sp)
            tg_tp = np.zeros_like(tg_sp)
            bool_pred_j = (pred_sp == j)
            bool_tg_j = (tg_sp == j)
            pred_tp[bool_pred_j==True] = 1
            tg_tp[bool_tg_j==True] = 1

            # dice_class.append(dice_coef(pred_tp, tg_tp))
            if np.sum(pred_tp)>0 or np.sum(tg_tp)>0:  #only when there's corresponding class in GT, we make average across this class; otherwise, pass
                iou = IoU(tg_tp , pred_tp)
                metric_class.append(iou)
        # print(metric_class)
        metric_batch += (np.mean(metric_class))
    # print(metric_batch)
    return metric_batch


def IoU_binary_class(decoded_fea, target):
    """
    :param decoded_fea: [batch_size, h,w], final decoded feature after sigmoid function
    :param target: [batch_size, h,w]
    :return: sum of dice in a batch(sum of each sample in a batch)
    """
    metric_batch = 0
    for i in range(decoded_fea.shape[0]):
        pred_sp = decoded_fea[i].cpu().detach().numpy()
        tg_sp = target[i].cpu().detach().numpy()
        # print(tg_sp)
        pred_tp = np.zeros_like(pred_sp)
        tg_tp = np.zeros_like(tg_sp)
        bool_pred_j = (pred_sp >= 0.5)
        bool_tg_j = (tg_sp == 1)
        pred_tp[bool_pred_j==True] = 1
        tg_tp[bool_tg_j==True] = 1

        # dice_class.append(dice_coef(pred_tp, tg_tp))
        # if np.sum(pred_tp)>0 or np.sum(tg_tp)>0:  #only when there's corresponding class in GT, we make average across this class; otherwise, pass
        #     iou = IoU(tg_tp , pred_tp)
        #     metric_batch += iou
        # if np.sum(pred_tp)==0 and np.sum(tg_tp)==0:  #only when there's corresponding class in GT, we make average across this class; otherwise, pass
        #     iou = 1
        #     metric_batch += iou

        iou = IoU(tg_tp , pred_tp)
        metric_batch += iou

    return metric_batch


def iou_on_batch(masks, pred):
    ious = []

    for i in range(pred.shape[0]):
        pred_tmp = pred[i][0].cpu().detach().numpy()
        # print("www",np.max(prediction), np.min(prediction))
        mask_tmp = masks[i].cpu().detach().numpy()
        pred_tmp[pred_tmp>=0.5] = 1
        pred_tmp[pred_tmp<0.5] = 0
        # print("2",np.sum(tmp))
        mask_tmp[mask_tmp>0] = 1
        mask_tmp[mask_tmp<=0] = 0
        # print("rrr",np.max(mask), np.min(mask))
        ious.append(jaccard_score(mask_tmp.reshape(-1), pred_tmp.reshape(-1)))
    return np.mean(ious)


def compute_mad_distance(predicted_mask, ground_truth_mask):
    predicted_mask = predicted_mask.cpu().detach().numpy()
    ground_truth_mask = ground_truth_mask.cpu().detach().numpy()

    absolute_diff = np.abs(predicted_mask - ground_truth_mask)
    mad_distance = np.mean(absolute_diff)

    return mad_distance

# """-------------------------------------------ChatGPT--------------------------------------"""
# ###Convert the binary masks into arrays of coordinates (pixel positions) representing the locations of non-zero pixels.
# def mask_to_coordinates(mask):
#     return np.column_stack(np.where(mask > 0)) #(num,3)
#
# def compute_hd95_distance(predicted_mask, ground_truth_mask):
#     predicted_coordinates = mask_to_coordinates(predicted_mask.cpu().detach().numpy())
#     ground_truth_coordinates = mask_to_coordinates(ground_truth_mask.cpu().detach().numpy())
#
#     distance, _,_ = directed_hausdorff(predicted_coordinates, ground_truth_coordinates)
#
#     bounding_box = np.vstack([predicted_coordinates, ground_truth_coordinates])
#     diagonal_length = np.max(bounding_box, axis=0) - np.min(bounding_box, axis=0)
#     diagonal_length = np.linalg.norm(diagonal_length)
#
#     normalized_hd95_distance = distance / diagonal_length
#
#     return normalized_hd95_distance


"""----------https://zhuanlan.zhihu.com/p/674695167------------"""

def transform_image_data(predict: np.ndarray, label: np.ndarray):
    predict = predict.astype(np.bool_).astype(np.int_)
    label = label.astype(np.bool_).astype(np.int_)
    return predict, label

# def dice_coef(predict: np.ndarray, label: np.ndarray, epsilon: float = 1e-5) -> float:
#     predict, label = transform_image_data(predict, label)
#     intersection = (predict * label).sum()
#     return (2. * intersection + epsilon) / (predict.sum() + label.sum() + epsilon)

def iou_score(predict: np.ndarray, label: np.ndarray, epsilon: float = 1e-5) -> float:
    predict, label = transform_image_data(predict, label)
    intersection = (predict & label).sum()
    union = (predict | label).sum()
    return (intersection + epsilon) / (union + epsilon)

def sensitivity(predict: np.ndarray, label: np.ndarray, epsilon: float = 1e-5) -> float:
    predict, label = transform_image_data(predict, label)
    intersection = (predict * label).sum()
    return (intersection + epsilon) / (label.sum() + epsilon)

def ppv(predict: np.ndarray, label: np.ndarray, epsilon: float = 1e-5) -> float:
    predict, label = transform_image_data(predict, label)
    intersection = (predict * label).sum()
    return (intersection + epsilon) / (predict.sum() + epsilon)


"""---------------代码没错,运行时间很长"""
def hd95(pred, label):
    dist = 0
    for i in range(pred.shape[0]):
        pred_sp = pred[i].cpu().detach().numpy()
        tg_sp = label[i].cpu().detach().numpy()
        # print(tg_sp)
        pred_tp = np.zeros_like(pred_sp)
        tg_tp = np.zeros_like(tg_sp)
        bool_pred_j = (pred_sp >= 0.5)
        bool_tg_j = (tg_sp == 1)
        pred_tp[bool_pred_j == True] = 1
        tg_tp[bool_tg_j == True] = 1
        distance = metric.binary.hd95(pred_tp, tg_tp)
        dist += (distance *  0.95)
        # print("HD95_image:", distance *  0.95)
    return dist