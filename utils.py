import torch
import torch.nn as nn
import numpy as np

from ResNet import conv3x3, BasicBlock, Bottleneck

class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
        else:
            self.std = std

    def forward(self, boxes, deltas):

        # print(boxes.shape)
        # print(deltas.shape)

        # print(deltas[0,0,:])
        # print(deltas[0,1,:])
        # print(deltas[0,2,:])
        # print(deltas[0,3,:])
        # print(deltas[0,4,:])
        # print(deltas[0,5,:])
        # print(deltas[0,6,:])
        # print(deltas[0,7,:])


        widths  = boxes[:, :, 2::4] - boxes[:, :, 0::4]
        heights = boxes[:, :, 3::4] - boxes[:, :, 1::4]
        ctr_x   = boxes[:, :, 0::4] + 0.5 * widths
        ctr_y   = boxes[:, :, 1::4] + 0.5 * heights

        dx = deltas[:, :, 0::4] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1::4] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2::4] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3::4] * self.std[3] + self.mean[3]

        # print(ctr_x.shape)
        # print(dx.shape)
        # print(widths.shape)

        pred_ctr_x = ctr_x + dx * widths

        pred_ctr_y = ctr_y + dy * heights
        pred_w     = torch.exp(dw) * widths
        pred_h     = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes_x1 = pred_boxes_x1[:, :, :, np.newaxis]
        pred_boxes_y1 = pred_boxes_y1[:, :, :, np.newaxis]
        pred_boxes_x2 = pred_boxes_x2[:, :, :, np.newaxis]
        pred_boxes_y2 = pred_boxes_y2[:, :, :, np.newaxis]

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=3).reshape(boxes.shape)

        #pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes

def distance2bbox(points, distance):
    """Decode distance prediction to bounding box.
    
    Args:
        points (Tensor): Shape (B, N, 2) or (N, 2).
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom). Shape (B, N, 4) or (N, 4)
    Returns:
        Tensor: Boxes with shape (N, 4) or (B, N, 4)
    """
    x1 = points[..., 0] + distance[..., 0]
    y1 = points[..., 1] + distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]

    bboxes = torch.stack([x1, y1, x2, y2], -1)
    return bboxes

def bbox2distance(points, bboxes):
    num_points = points.size(0)
    num_bboxes = bboxes.size(0)
    bboxes = bboxes[None].expand(num_points, num_bboxes, 4)
    xs, ys = points[:, 0], points[:, 1]
    xs = xs[:, None].expand(num_points, num_bboxes)
    ys = ys[:, None].expand(num_points, num_bboxes)
    left = bboxes[..., 0] - xs
    right = bboxes[..., 2] - xs
    top = bboxes[..., 1] - ys
    bottom = bboxes[..., 3] - ys
    bbox_targets = torch.stack((left, top, right, bottom), -1)
    return bbox_targets

def maxpool_nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):

        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0::4] = torch.clamp(boxes[:, :, 0::4], min=0)
        boxes[:, :, 1::4] = torch.clamp(boxes[:, :, 1::4], min=0)

        boxes[:, :, 2::4] = torch.clamp(boxes[:, :, 2::4], max=width)
        boxes[:, :, 3::4] = torch.clamp(boxes[:, :, 3::4], max=height)
      
        return boxes


# import torch
# import torch.nn as nn
# import numpy as np

# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out

# class BBoxTransform(nn.Module):

#     def __init__(self, mean=None, std=None):
#         super(BBoxTransform, self).__init__()
#         if mean is None:
#             self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
#         else:
#             self.mean = mean
#         if std is None:
#             self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
#         else:
#             self.std = std

#     def forward(self, boxes, deltas):

#         widths  = boxes[:, :, 2] - boxes[:, :, 0]
#         heights = boxes[:, :, 3] - boxes[:, :, 1]
#         ctr_x   = boxes[:, :, 0] + 0.5 * widths
#         ctr_y   = boxes[:, :, 1] + 0.5 * heights

#         dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
#         dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
#         dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
#         dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

#         pred_ctr_x = ctr_x + dx * widths
#         pred_ctr_y = ctr_y + dy * heights
#         pred_w     = torch.exp(dw) * widths
#         pred_h     = torch.exp(dh) * heights

#         pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
#         pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
#         pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
#         pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

#         pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

#         return pred_boxes


# class ClipBoxes(nn.Module):

#     def __init__(self, width=None, height=None):
#         super(ClipBoxes, self).__init__()

#     def forward(self, boxes, img):

#         batch_size, num_channels, height, width = img.shape

#         boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
#         boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

#         boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
#         boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)
      
#         return boxes
