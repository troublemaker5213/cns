import torch
import torch.nn as nn


class softIoULoss(nn.Module):

    def __init__(self):
        super(softIoULoss,self).__init__()
    def forward(self, label, pred, sw=None, recall=False):
        costs = softIoU(label, pred, recall).view(-1, 1)
        #pdb.set_trace()
        if sw and (sw.data > 0).any():
            costs = torch.mean(torch.masked_select(costs,sw.byte()))
        else:
            costs = torch.mean(costs)
        return costs

def softIoU(target, out, e=1e-6, recall=False):

    """
    Args:
        target: A Variable containing a LongTensor of size
            (batch, N) which contains the true binary mask.
        out: A Variable containing a FloatTensor of size
            (batch, N) which contains the logits for each pixel in the output mask.
        sw: A Variable containing a LongTensor of size (batch,)
            which contains the mask to apply to each element in a batch.
    Returns:
        loss: Sum of losses with applied sample weight
    """


    out = torch.sigmoid(out)
    if not recall:     # .sum(-1,True).sum(2,True)
        num = (out*target).sum((1,2,3),True)
        den = (out+target-out*target).sum((1,2,3),True)+ e
        iou = num / den
    else:
        # num = (out*target).sum(1,True).sum(-1,True)
        # den = target.sum(1,True).sum(-1,True) + e
        num = (out*target).sum((1,2,3),True)
        den = target.sum((1,2,3),True) + e
        iou = num / den

    cost = (1 - iou)

    return cost.squeeze()
