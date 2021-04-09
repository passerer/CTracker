import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, background_class, alpha=0.25, gamma=2, reduce=True, ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.background_class = background_class
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
        self.ignore_index = ignore_index

    def forward(self, input, target):
        pos_indices = target!=self.ignore_index
        target = target[pos_indices]
        input = input[pos_indices]
        ce_loss = F.cross_entropy(input, target, reduction ='none')
        pt = torch.exp(-ce_loss)
        alpha_weights = torch.full_like(pt, self.alpha)
        alpha_weights[target!=self.background_class] = 1 - self.alpha
        F_loss = alpha_weights * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class GiouLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(GiouLoss, self).__init__()
        self.eps = eps

    def forward(self, bboxes1, bboxes2):
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
                bboxes1[..., 3] - bboxes1[..., 1])
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
                bboxes2[..., 3] - bboxes2[..., 1])
        lt = torch.max(bboxes1[..., :2],
                       bboxes2[..., :2])
        rb = torch.min(bboxes1[..., 2:],
                       bboxes2[..., 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]
        union = area1 + area2 - overlap
        eps = union.new_tensor([self.eps])
        union = torch.max(union, eps)
        ious = overlap / union

        enclosed_lt = torch.min(bboxes1[..., :2],
                                bboxes2[..., :2])
        enclosed_rb = torch.max(bboxes1[..., 2:],
                                bboxes2[..., 2:])
        enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        enclose_area = torch.max(enclose_area, eps)
        gious = ious - (enclose_area - union) / enclose_area
        giou_loss = torch.mean(1-gious)
        return giou_loss
