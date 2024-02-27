import numpy as np

# #二值分割图是一个波段的黑白图，正样本值为1，负样本值为0
# #通过矩阵的逻辑运算分别计算出tp,tn,fp,fn
# seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
# true_pos = float(np.logical_and(premask, groundtruth).sum())  # float for division
# true_neg = np.logical_and(seg_inv, gt_inv).sum()
# false_pos = np.logical_and(premask, gt_inv).sum()
# false_neg = np.logical_and(seg_inv, groundtruth).sum()
#
# #然后根据公式分别计算出这几种指标
# prec = true_pos / (true_pos + false_pos + 1e-6)
# rec = true_pos / (true_pos + false_neg + 1e-6)
# accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg + 1e-6)
# F1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
# IoU = true_pos / (true_pos + false_neg + false_pos + 1e-6)


class Metric:
    def __init__(self, premask, groundtruth):
        # premask ,groundtruth 是 0,1矩阵
        self.seg_inv = np.logical_not(premask)
        self.gt_inv = np.logical_not(groundtruth)
        self.true_pos = float(np.logical_and(premask, groundtruth).sum())  # float for division
        self.true_neg = np.logical_and(self.seg_inv, self.gt_inv).sum()
        self.false_pos = np.logical_and(premask, self.gt_inv).sum()
        self.false_neg = np.logical_and(self.seg_inv, groundtruth).sum()

    def Precision(self):
        # precision = tp / (tp + fp)
        prec = self.true_pos / (self.true_pos + self.false_pos + 1e-6)
        return prec

    def Recall(self):
        # recall = tp / (tp + fn)
        rec = self.true_pos / (self.true_pos + self.false_neg + 1e-6)
        return rec

    def Accuracy(self):
        # accuracy = (tp + tn) / (tp + tn + fp + fn)
        accuracy = (self.true_pos + self.true_neg) / (self.true_pos + self.true_neg + self.false_pos + self.false_neg + 1e-6)
        return accuracy

    def F1(self):
        # F1 = (2 * Recall * Precision) / (Recall + Precision)
        F1 = 2 * self.true_pos / (2 * self.true_pos + self.false_pos + self.false_neg + 1e-6)
        return F1

    def IoU(self):
        # iou = tp / (tp + fn + fp)
        IoU = self.true_pos / (self.true_pos + self.false_neg + self.false_pos + 1e-6)
        return IoU