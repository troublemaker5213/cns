from torch import nn
import torch
from torch.nn import functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True, activation='sigmoid'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.activation = activation

    def forward(self, inputs, targets):

        predict = inputs.float()
        gt = targets.float()
        if self.activation == 'softmax':
            predict = F.softmax(predict, dim=1)
        elif self.activation == 'sigmoid':
            predict = F.sigmoid(predict)
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(predict, gt, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(predict, gt, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss