import numpy as np
import torch.nn as nn
import torch
import math
import time

import torch.nn.functional as F
from utils import  distance2bbox, bbox2distance, maxpool_nms
from losses import FocalLoss, GiouLoss
from lib.nms import cython_soft_nms_wrapper, nms
from ResNet import build_resnet, ResNet
from RepVgg import build_repvgg



class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, with_bn=True, **kwargs):
        super(ConvModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.with_bn = with_bn

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=kernel_size, padding=padding)
        if self.with_bn:
            self.bn = nn.BatchNorm2d(self.out_channels)
        self.activate = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        if self.with_bn:
            out = self.bn(out)
        out = self.activate(out)
        return out


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels, out_indices, **kwargs):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_indices = out_indices
        assert isinstance(self.out_indices, list)

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(0, len(self.in_channels)):
            self.lateral_convs.append(ConvModule(in_channels[i], self.out_channels))
            self.fpn_convs.append(ConvModule(self.out_channels,self.out_channels))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        for i in range(len(inputs)-1, 0, -1):
            laterals[i-1] += F.interpolate(laterals[i], scale_factor=2, mode='bilinear',align_corners=True)
        fpn_outputs = [self.fpn_convs[i](laterals[i]) for i in self.out_indices]
        return fpn_outputs


class CNNHead(nn.Module):
    def __init__(self, in_channels, out_channels, stack_convs = 2, with_bn=True, **kwargs):
        super(CNNHead, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stack_convs = stack_convs
        if stack_convs <=2:
            scale = 4
        else:
            scale = 2

        self.convs = nn.ModuleList()
        chn = self.in_channels
        for i in range(self.stack_convs):
            self.convs.append(ConvModule(chn, chn//scale))
            chn = chn//scale
        self.last_convs = nn.Conv2d(chn, self.out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.last_convs(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_backbone(name, pretrained=False, **kwargs):
    if 'resnet' in name:
        return build_resnet(name, pretrained=pretrained, **kwargs)
    elif 'RepVGG' in name:
        return build_repvgg(name, pretrained=pretrained, **kwargs)
    else:
        raise NotImplementedError


class FCOSTracker(nn.Module):

    def __init__(self,
                 backbone,
                 neck=None,
                 cls_head=None,
                 reg_head = None,
                 train_cfg=None,
                 test_cfg=None,
                 stride=None,
                 regress_range=((-1,64),(64,256),(256,1e8)),
                 margin = 2,
                 center_sampling=True,
                 center_sampling_radius=3,
                 cls_attention=False,
                 use_stride_channel=True):
        super(FCOSTracker, self).__init__()
        self.backbone = build_backbone(**backbone)
        if neck is not None:
            self.neck = FPN(**neck)
        else:
            self.neck = None
        self.cls_head = CNNHead(**cls_head)
        self.reg_head = CNNHead(**reg_head)
        self.num_classes = cls_head.get('out_channels', 2) #include BG.
        self.background_class = self.num_classes -1
        self.use_stride_channel = use_stride_channel
        if not use_stride_channel:
            self.reg_channels = reg_head.get('out_channels', 8)
        else:
            self.reg_channels = reg_head.get('out_channels', 9)
        self.cls_ignore_index = cls_head.get('ignore_index', -1)
        self.reg_ignore_index = reg_head.get('ignore_index', 0)
        #self.cls_loss = nn.CrossEntropyLoss(ignore_index=self.cls_ignore_index)
        self.cls_loss = FocalLoss(background_class=self.background_class,ignore_index=self.cls_ignore_index)
        #self.reg_loss = GiouLoss()
        self.reg_loss = nn.SmoothL1Loss()
        self.stride = stride
        self.reg_scale = nn.Parameter(torch.tensor([math.log(stride) for stride in self.stride], dtype=torch.float)) #A learnable scale parameter.
        self.loss_scale = torch.nn.Parameter(torch.tensor([0.33,3.33],dtype=torch.float))
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.regress_range = regress_range
        self.margin = margin
        self.center_sampling = center_sampling
        self.center_sampling_radius = center_sampling_radius
        self.cls_attention = cls_attention
        self.init_weights(backbone.get('pretrained',False))

    def init_weights(self,pretrained=False):
        """Initialize the weights in tracker.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if not pretrained:
            self.backbone.init_weights()
        if self.neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.cls_head.init_weights()
        self.reg_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.neck:
            x = self.neck(x)
            return x
        else:
            return (x,)

    def freeze_backbone_bn(self):
        self.backbone.freeze_bn()

    def forward_dummy(self, img):
        """Used for computing network flops.
        """
        raise NotImplementedError

    def forward(self, inputs, train_mode, **kwargs):
        if train_mode:
            return self.forward_train(inputs, **kwargs)
        else:
            return self.forward_test(inputs, **kwargs)

    def forward_train(self,
                      inputs):
        img_batch_1, annotations_1, img_batch_2, annotations_2 = inputs
        img_batch = torch.cat([img_batch_1, img_batch_2], 0)
        annotations = torch.cat([annotations_1, annotations_2], 0)
        features = self.extract_feat(img_batch)
        track_features = []
        for ind, featmap in enumerate(features):
            featmap_t, featmap_t1 = torch.chunk(featmap, chunks=2, dim=0)
            track_features.append(torch.cat((featmap_t, featmap_t1), dim=1))

        reg_features = []
        cls_features = []
        mlvl_points = []
        regress_ranges = []
        for i, feature in enumerate(track_features):
            cls_feat = self.cls_head(feature)
            if self.cls_attention:
                reg_in = feature*(1-torch.softmax(cls_feat,dim=1)[:,self.background_class])
            else:
                reg_in = feature
            reg_feat = self.reg_head(reg_in)

            cls_feat = cls_feat.permute(0, 2, 3, 1)
            batch_size, height, width, _ = cls_feat.shape
            cls_feat = cls_feat.contiguous().view(batch_size, -1, self.num_classes)

            reg_feat = reg_feat.permute(0, 2, 3, 1)
            reg_feat = reg_feat.contiguous().view(batch_size, -1, self.reg_channels)
            if not self.use_stride_channel:
                reg_feat = reg_feat*torch.exp(self.reg_scale[i])
            else:
                reg_feat = reg_feat[..., :8]*torch.exp(reg_feat[..., 8:])

            points = self.get_points(width, height, self.stride[i], reg_feat[0].dtype, reg_feat[0].device)

            mlvl_points.append(points)
            regress_ranges.append(points.new_tensor(self.regress_range[i])[None].expand_as(points))
            reg_features.append(reg_feat)
            cls_features.append(cls_feat)

        regression = torch.cat(reg_features, dim=1) #shape: (batch_size, all_points_num, 8)
        classification = torch.cat(cls_features, dim=1) #shape: (batch_size, all_points_num, num_classes)
        mlvl_points = torch.cat(mlvl_points, dim=0) #shape: (all_points_num, 2)
        regress_ranges = torch.cat(regress_ranges, dim=0)
        batch_size = regression.size(0)
        cls_losses = []
        reg_losses = []
        for j in range(batch_size):
            reg = regression[j, ...]
            cls = classification[j, ...]

            ann_1 = annotations_1[j]
            ann_1 = ann_1[ann_1[:, 4] != self.cls_ignore_index]
            gt_labels = ann_1[:, 4] #shape: (num_gts, )
            assert(gt_labels ==0).sum()==gt_labels.size(0), annotations_1
            gt_bboxes = ann_1[:, :4] #shape: (num_gts, 4)
            gt_ids = ann_1[:, 5] #shape: (num_gts, )

            ann_2 = annotations_2[j]
            ann_2 = ann_2[ann_2[:, 4] != self.cls_ignore_index]
            gt_labels_next = ann_2[:, 4]
            gt_bboxes_next = ann_2[:, :4]
            gt_ids_next = ann_2[:, 5]
            labels, bbox_target, ids = self.get_targets(gt_bboxes, gt_labels, gt_ids, mlvl_points, regress_ranges)# labels: (num_points,) bbox_target: (num_points, 4)
            cls_losses.append(self.cls_loss(cls, labels.long()))
            pos_indices = ((labels != self.background_class) & (labels != self.cls_ignore_index))
            if pos_indices.sum() > 0:
                reg = reg[pos_indices]
                bbox_target = bbox_target[pos_indices]
                ids = ids[pos_indices]
                pos_points = mlvl_points[pos_indices]
                reg_loss = self.reg_loss(bbox_target, reg[..., :4])

                match_indices = (ids[:, None] == gt_ids_next[None, :]) #shape: (num_pos_points, num_gts_next)
                pos_indices_next = (match_indices.sum(-1) == 1)
                if pos_indices_next.sum() > 0:
                    bbox_target_next = bbox2distance(pos_points, gt_bboxes_next) #shape: (num_pos_points, num_gts_next, 4)
                    bbox_target_next = bbox_target_next[match_indices] #shape: (num_pos_points, 4)
                    reg_loss += self.reg_loss(bbox_target_next, reg[pos_indices_next][..., 4:])
                reg_losses.append(reg_loss)
            else:
                reg_losses.append(torch.tensor(0, dtype=reg_feat[0].dtype, device=reg_feat[0].device))
        return torch.stack(cls_losses).mean(dim=0, keepdim=True), torch.stack(reg_losses).mean(
            dim=0, keepdim=True)

    def get_points(self, width, height, stride, dtype, device):
        x_range = torch.arange(width, dtype=dtype, device=device)
        y_range = torch.arange(height, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    def get_targets(self, gt_bboxes, gt_labels, gt_ids, points, regress_ranges):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.background_class),\
                   gt_bboxes.new_zeros((num_points, 4)),\
                   gt_ids.new_zeros((num_points,))
        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1])

        areas = areas[None].repeat(num_points, 1)
        bbox_targets = bbox2distance(points, gt_bboxes)

        stride = self.stride[0]
        if self.center_sampling:
            gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
            xs, ys = points[:, 0], points[:, 1]
            xs = xs[:, None].expand(num_points, num_gts)
            ys = ys[:, None].expand(num_points, num_gts)
            # condition1: inside a `center bbox`
            radius = self.center_sampling_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)

            x_mins = center_xs - radius*stride
            y_mins = center_ys - radius*stride*2
            x_maxs = center_xs + radius*stride
            y_maxs = center_ys + radius*stride*2
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            inside_gt_bbox_mask = (bbox_targets[:, :, :2].max(-1)[0] < 0) & (bbox_targets[:, :, 2:].min(-1)[0] > 0)
        max_regress_distance = bbox_targets.abs().max(-1)[0]
        #min_regress_distance = bbox_targets.abs().min(-1)[0]
        # inside_regress_range = (
        #         (max_regress_distance >= regress_ranges[..., 0][:,None])
        #         & (max_regress_distance <= regress_ranges[..., 1][:,None]))
        inside_mask = inside_gt_bbox_mask #shape:(num_points, num_gts)
        areas[inside_mask == 0] = 1e8
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        ids = gt_ids[min_area_inds]
        labels[min_area == 1e8] = self.background_class  # set as BG
        ids[min_area == 1e8] = -1

        # #count id num
        # dic_ = {}
        # for id in ids:
        #     a = id.cpu().item()
        #     if a==-1:
        #         continue
        #     if a not in dic_:
        #         dic_[a]=1
        #     else:
        #         dic_[a] +=1
        # num_med = int(np.median(list(dic_.values())))
        # print(dic_.values(),num_med)

        bbox_targets[~inside_mask] = 0  # ignore
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets, ids


    def forward_test(self, inputs, last_feats=None,return_hm=False, **kwargs):
        # only one imge per test.
        img = inputs
        feats = self.extract_feat(img)
        if last_feats is None:
            return torch.zeros(0), torch.zeros(0, 4), feats, None
        track_features = []
        for ind, featmap in enumerate(feats):
            track_features.append(torch.cat((last_feats[ind], featmap), dim=1))
        reg_features = []
        cls_scores = []
        mlvl_points = []
        for i, feature in enumerate(track_features):
            cls_feat = self.cls_head(feature)
            if self.cls_attention:
                reg_in = feature * (1 - torch.softmax(cls_feat, dim=1)[:,self.background_class])
            else:
                reg_in = feature
            reg_feat = self.reg_head(reg_in)

            cls_feat = cls_feat.permute(0, 2, 3, 1)
            batch_size, height, width, _ = cls_feat.shape
            cls_feat = cls_feat.contiguous().view(batch_size, -1, self.num_classes)
            score = cls_feat.softmax(dim=-1)
            score, indices = torch.max(score, dim=-1, keepdim=True)
            score[indices == self.background_class] = 0
            score = maxpool_nms(score,kernel=5)

            reg_feat = reg_feat.permute(0, 2, 3, 1)
            reg_feat = reg_feat.contiguous().view(batch_size, -1, self.reg_channels)
            if not self.use_stride_channel:
                reg_feat = reg_feat*torch.exp(self.reg_scale[i])
            else:
                reg_feat = reg_feat[..., :8]*torch.exp(reg_feat[..., 8:])

            points = self.get_points(width, height, self.stride[i], reg_feat[0].dtype, reg_feat[0].device)

            mlvl_points.append(points)
            reg_features.append(reg_feat)
            cls_scores.append(score)
        regression = torch.cat(reg_features, dim=1) #shape:(1, num_points, 8)
        cls_scores = torch.cat(cls_scores, dim=1) #shape:(1, num_points, 1)
        mlvl_points = torch.cat(mlvl_points, dim=0)

        conf_feat = cls_scores[0][:,0].view(feats[0].shape[-2],feats[0].shape[-1])
        pos_indices = cls_scores > 0.6
        pos_indices = pos_indices[0, :, 0] #squeeze the batc_size dim(batch_size always is 1)
        pos_num = pos_indices.sum()
        if pos_num == 0:
            # no boxes to NMS, just return
            return torch.zeros(0), torch.zeros(0, 4), feats, conf_feat

        cls_scores = cls_scores[..., pos_indices, :] #shape:(1, num_pos_points)
        mlvl_points = mlvl_points[pos_indices] #shape:(num_pos_points, 2)
        regression = regression[..., pos_indices, :] #shape:(1, num_pos_points, 8)

        bboxes_1 = distance2bbox(mlvl_points, regression[...,:4])
        bboxes_2 = distance2bbox(mlvl_points, regression[...,4:])
        bboxes = torch.cat([bboxes_1,bboxes_2],dim=-1)
        final_bboxes, index = cython_soft_nms_wrapper(0.7, method='gaussian')(
           torch.cat([bboxes.contiguous(), cls_scores], dim=2)[0].cpu().numpy())
        #final_bboxes, index = nms(torch.cat([bboxes.contiguous(), cls_scores], dim=2)[0].cpu().numpy(),0.5)

        return final_bboxes[:, -1], final_bboxes, feats, conf_feat

