from __future__ import print_function, division
import copy
import os
import torch
import numpy as np
import random
import json
import csv
from six import raise_from
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler

import skimage.io
import skimage.transform
import skimage.color
import skimage
from PIL import Image, ImageEnhance

RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]

class MOTDataset(Dataset):
    """CoCo dataset."""

    def __init__(self, root_path, train_file, transform=None):
        """
        Args:

        """
        super(MOTDataset, self).__init__()
        self.train_file = train_file
        self.transform = transform
        self.root_path = root_path

        # load annotations
        print('loading annotations into memory')
        with open(self.train_file, 'r') as fr:
            coco_json = json.load(fr)
        assert type(coco_json) == dict, 'annotation file format {} not supported'.format(type(coco_json))
        self.classes = coco_json['categories']
        self.image_data = self._read_annotations(coco_json)

        self.image_names = list(self.image_data.keys())
        self.name2video_frames = dict()
        self.image_name_prefix = list()
        for image_name in self.image_names:
            self.image_name_prefix.append(image_name[0:-len(image_name.split('/')[-1].split('_')[-1])])
        self.image_name_prefix = set(self.image_name_prefix)
        print('total video count: {}'.format(len(self.image_name_prefix)))
        for image_name in self.image_names:
            cur_prefix = image_name[0:-len(image_name.split('/')[-1])]
            if cur_prefix not in self.name2video_frames:
                self.name2video_frames[cur_prefix] = 1
            else:
                self.name2video_frames[cur_prefix] += 1

    def _get_random_surroud_name(self, image_name, max_diff=3, ignore_equal=True, pos_only=True):
        suffix_name = image_name.split('/')[-1].split('_')[-1]
        prefix = image_name[0:-len(suffix_name)]
        cur_index = int(float(suffix_name.split('.')[0]))
        total_number = self.name2video_frames[prefix]
        if total_number < 2: return image_name
        next_index = cur_index
        while True:
            range_low = max(1, cur_index - max_diff)
            range_high = min(cur_index + max_diff, total_number)
            if pos_only:
                range_low = cur_index
                if ignore_equal:
                    range_low = range_low + 1
                    if cur_index == total_number:
                        return image_name

            next_index = random.randint(range_low, range_high)
            if ignore_equal:
                if next_index == cur_index:
                    continue
            break

        return prefix + '{0:06}.'.format(next_index) + suffix_name.split('.')[-1]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        while True:
            try:
                img = self.load_image(idx)
                next_name = self._get_random_surroud_name(self.image_names[idx])
                img_next = self.load_image(next_name)
                annot = self.load_annotations(idx)
                annot_next = self.load_annotations(next_name)

                if (annot.shape[0] < 1) or (annot_next.shape[0] < 1):
                    idx = random.randrange(0, len(self.image_names))
                    continue
            except FileNotFoundError:
                print('FileNotFoundError in process image.')
                idx = random.randrange(0, len(self.image_names))
                continue
            break

        if np.random.rand() < 0.5:
            sample = {'img': img, 'annot': annot, 'img_next': img_next, 'annot_next': annot_next}
        else:
            sample = {'img': img_next, 'annot': annot_next, 'img_next': img, 'annot_next': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample


    def load_image(self, image_index_or_name):
        if isinstance(image_index_or_name, int):
            img = skimage.io.imread(self.image_names[image_index_or_name])
        elif isinstance(image_index_or_name, str):
            img = skimage.io.imread(image_index_or_name)
        else:
            raise ValueError('Expected int or str, but get {}'.format(type(image_index_or_name)))

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img

    def load_annotations(self, image_index_or_name):
        # get ground truth annotations
        if isinstance(image_index_or_name, int):
            annotation_list = self.image_data[self.image_names[image_index_or_name]]
        elif isinstance(image_index_or_name, str):
            annotation_list = self.image_data[image_index_or_name]
        else:
            raise ValueError('Expected int or str, but get {}'.format(type(image_index_or_name)))

        annotations = np.zeros((0, 6))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']
            obj_id = a['obj_id']
            category_id = a['class']
            if category_id > 0:
                category_id -= 1

            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue

            annotation = np.zeros((1, 6))

            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4] = category_id
            annotation[0, 5] = obj_id
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, coco_json):
        result = {}
        img_id2path = {}
        for img in coco_json['images']:
            img_full_path = os.path.join(self.root_path, img['file_name'])
            result[img_full_path] = []
            img_id2path[img['id']] = img_full_path
        for ann in coco_json['annotations']:
            result[img_id2path[ann['image_id']]].append({\
                'x1': float(ann['bbox'][0]), 'y1': float(ann['bbox'][1]), \
                'x2':float(ann['bbox'][0]+ann['bbox'][2]), 'y2':float(ann['bbox'][1]+ann['bbox'][3]) , \
                'class': int(ann['category_id']), 'obj_id': int(ann['track_id'])})
        return result

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


class CrowdHumanDataset(Dataset):
    """CoCo dataset."""
    def __init__(self, root_path, train_file, transform=None):
        """
        Args:

        """
        super(CrowdHumanDataset, self).__init__()
        self.train_file = train_file
        self.transform = transform
        self.root_path = root_path

        # load annotations
        print('loading annotations into memory')
        with open(self.train_file, 'r') as fr:
            coco_json = json.load(fr)
        assert type(coco_json) == dict, 'annotation file format {} not supported'.format(type(coco_json))
        self.classes = coco_json['categories']
        self.image_data = self._read_annotations(coco_json)

        self.image_names = list(self.image_data.keys())

        print('total video count: {}'.format(len(self.image_names)))

    def gen_fake_img_ann(self, img, annots, debug=False):
        img_next, annots_next = copy.deepcopy(img), copy.deepcopy(annots)
        height, width, depth = img.shape
        # step 1: crop
        crop_range = (0.9,1.0)
        min_iou=0.4
        crop_ratio = np.random.uniform(crop_range[0],crop_range[1])
        crop_h = int(height*crop_ratio)
        crop_w = int(width*crop_ratio)
        crop_success = False
        for _ in range(10):
            left = np.random.uniform(0, width - crop_w)
            top = np.random.uniform(0, height - crop_h)
            # convert to integer rect x1,y1,x2,y2
            rect = np.array([int(left), int(top), int(left + crop_w), int(top + crop_h)])
            # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
            overlap = overlap_numpy(annots[:, :4], rect)
            if overlap.max() < min_iou:
                continue
            crop_success = True
            break
        if not crop_success:
            return img_next, annots_next
        img_next = img_next[rect[1]:rect[3], rect[0]:rect[2], :]
        annots_next = annots_next[overlap > min_iou, :]
        annots_next[:, :2] -= rect[:2]
        annots_next[:, 2:4] -= rect[:2]
        # step 2: resize slightly
        resize_range = (0.95,1.05)
        resize_ratio = np.random.uniform(resize_range[0],resize_range[1])
        resize_h = int(img_next.shape[0]*resize_ratio)
        resize_w = int(img_next.shape[1]*resize_ratio)
        img_next = (255.0 * skimage.transform.resize(img_next, (resize_h, resize_w))).astype(np.uint8)
        annots_next[:, :4] *= resize_ratio
        ## step 3: bbox shift and rescale slightly
        # for i in range(len(annots_next)):
        #     r = np.random.randn(4)/40
        #     r = np.clip(r,a_min=-0.05,a_max=0.05)
        #     annots_next[i][:4] += r*(annots_next[i][0]-annots_next[i][2])
        #step4: padding img_next
        pad_img_next = np.zeros((height,width,depth),dtype=img.dtype)
        min_width = min(width, img_next.shape[1])
        min_height = min(height,img_next.shape[0])
        pad_img_next[:min_height, :min_width, :] = img_next[:min_height, :min_width, :]
        if debug:
            from matplotlib import pyplot as plt
            import cv2
            plt.figure(figsize=(32,16))
            for ann in annots:
                cv2.rectangle(img,(int(ann[0]),int(ann[1])),(int(ann[2]),int(ann[3])),color=(0,200,0),thickness=2)
            for ann in annots_next:
                cv2.rectangle(pad_img_next,(int(ann[0]),int(ann[1])),(int(ann[2]),int(ann[3])),color=(0,200,0),thickness=2)
            plt.subplot(211)
            plt.imshow(img)
            plt.subplot(212)
            plt.imshow(pad_img_next)
            plt.show()
            exit(0)

        return pad_img_next,annots_next

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        while True:
            try:
                img = self.load_image(idx)
                annot = self.load_annotations(idx)
                img_next,annot_next = self.gen_fake_img_ann(img, annot)

                if (annot.shape[0] < 1) or (annot_next.shape[0] < 1):
                    idx = random.randrange(0, len(self.image_names))
                    continue
            except FileNotFoundError:
                print('FileNotFoundError in process image.')
                idx = random.randrange(0, len(self.image_names))
                continue
            break

        if np.random.rand() < 0.5:
            sample = {'img': img, 'annot': annot, 'img_next': img_next, 'annot_next': annot_next}
        else:
            sample = {'img': img_next, 'annot': annot_next, 'img_next': img, 'annot_next': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample


    def load_image(self, image_index_or_name):
        if isinstance(image_index_or_name, int):
            img = skimage.io.imread(self.image_names[image_index_or_name])
        elif isinstance(image_index_or_name, str):
            img = skimage.io.imread(image_index_or_name)
        else:
            raise ValueError('Expected int or str, but get {}'.format(type(image_index_or_name)))

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img

    def load_annotations(self, image_index_or_name):
        # get ground truth annotations
        if isinstance(image_index_or_name, int):
            annotation_list = self.image_data[self.image_names[image_index_or_name]]
        elif isinstance(image_index_or_name, str):
            annotation_list = self.image_data[image_index_or_name]
        else:
            raise ValueError('Expected int or str, but get {}'.format(type(image_index_or_name)))

        annotations = np.zeros((0, 6))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']
            obj_id = a['obj_id']
            category_id = a['class']
            if category_id > 0:
                category_id -= 1

            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue

            annotation = np.zeros((1, 6))

            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4] = category_id
            annotation[0, 5] = obj_id
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, coco_json):
        result = {}
        img_id2path = {}
        for img in coco_json['images']:
            img_full_path = os.path.join(self.root_path, img['file_name'])
            result[img_full_path] = []
            img_id2path[img['id']] = img_full_path
        for ann in coco_json['annotations']:
            result[img_id2path[ann['image_id']]].append({\
                'x1': float(ann['bbox'][0]), 'y1': float(ann['bbox'][1]), \
                'x2':float(ann['bbox'][0]+ann['bbox'][2]), 'y2':float(ann['bbox'][1]+ann['bbox'][3]) , \
                'class': int(ann['category_id']), 'obj_id': int(ann['id'])})
        return result

    def num_classes(self):
        return max(self.classes.values())

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


class CaltechPedestrianDataset(CrowdHumanDataset):
    def __init__(self, root_path, train_file, transform=None):
        super(CaltechPedestrianDataset,self).__init__(root_path,train_file,transform)

def resize_collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    imgs_next = [s['img_next'] for s in data]
    annots_next = [s['annot_next'] for s in data]
        
    widths = [int(s.shape[1]) for s in imgs]
    heights = [int(s.shape[0]) for s in imgs]
    batch_size = len(imgs)

    mean_width = int(np.array(widths).mean())
    mean_height = int(np.array(heights).mean())
    mean_width += (32 - mean_width % 32)
    mean_height += (32 - mean_height % 32)

    align_imgs = []
    align_imgs_next = []

    for i in range(batch_size):
        img = imgs[i]
        height, width = img.shape[0], img.shape[1]
        if annots[i].shape[0]>0:
            annots[i][:, 0] *= (mean_width / width)
            annots[i][:, 1] *= (mean_height / height)
            annots[i][:, 2] *= (mean_width / width)
            annots[i][:, 3] *= (mean_height / height)
        img = img.view(1, height, width, -1).permute(0, 3, 1, 2)
        img = F.interpolate(img, size=(mean_height, mean_width), mode='bilinear', align_corners=True)
        align_imgs.append(img.view(-1,mean_height,mean_width))

        img_next = imgs_next[i]
        height, width = img_next.shape[0], img_next.shape[1]
        if annots_next[i].shape[0] > 0:
            annots_next[i][:, 0] *= (mean_width / width)
            annots_next[i][:, 1] *= (mean_height / height)
            annots_next[i][:, 2] *= (mean_width / width)
            annots_next[i][:, 3] *= (mean_height / height)
        img_next = img_next.view(1, height, width, -1).permute(0, 3, 1, 2)
        img_next = F.interpolate(img_next, size=(mean_height, mean_width), mode='bilinear', align_corners=True)
        align_imgs_next.append(img_next.view(-1, mean_height, mean_width))
    align_imgs = torch.stack(align_imgs)
    align_imgs_next = torch.stack(align_imgs_next)

    max_num_annots = max(annot.shape[0] for annot in annots)
    max_num_annots_next = max(annot.shape[0] for annot in annots_next)
    max_num_annots = max(max_num_annots, max_num_annots_next)
    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 6)) * -1
        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 6)) * -1
    
    if max_num_annots > 0:
        annot_padded_next = torch.ones((len(annots_next), max_num_annots, 6)) * -1
        for idx, annot in enumerate(annots_next):
            if annot.shape[0] > 0:
                annot_padded_next[idx, :annot.shape[0], :] = annot
    else:
        annot_padded_next = torch.ones((len(annots_next), 1, 6)) * -1

    return {'img': align_imgs, 'annot': annot_padded, 'img_next': align_imgs_next, 'annot_next': annot_padded_next}


def pad_collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    imgs_next = [s['img_next'] for s in data]
    annots_next = [s['annot_next'] for s in data]

    widths = [int(s.shape[1]) for s in imgs]
    heights = [int(s.shape[0]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_height, max_width, 3)
    padded_imgs_next = torch.zeros(batch_size, max_height, max_width, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

        img_next = imgs_next[i]
        padded_imgs_next[i, :int(img_next.shape[0]), :int(img_next.shape[1]), :] = img_next

    max_num_annots = max(annot.shape[0] for annot in annots)
    max_num_annots_next = max(annot.shape[0] for annot in annots_next)
    max_num_annots = max(max_num_annots, max_num_annots_next)
    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 6)) * -1
        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 6)) * -1

    if max_num_annots > 0:
        annot_padded_next = torch.ones((len(annots_next), max_num_annots, 6)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots_next):
                if annot.shape[0] > 0:
                    annot_padded_next[idx, :annot.shape[0], :] = annot
    else:
        annot_padded_next = torch.ones((len(annots_next), 1, 6)) * -1

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)
    padded_imgs_next = padded_imgs_next.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'img_next': padded_imgs_next, 'annot_next': annot_padded_next}

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy + 1), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0] + 1) *
              (box_a[:, 3]-box_a[:, 1] + 1))  # [A,B]
    area_b = ((box_b[2]-box_b[0] + 1) *
              (box_b[3]-box_b[1] + 1))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def overlap_numpy(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0] + 1) *
              (box_a[:, 3]-box_a[:, 1] + 1))  # [A,B]
    return inter / area_a  # [A,B]


class RandomFlip(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.3):

        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image_next, annots_next = sample['img_next'], sample['annot_next']
            image = image[:, ::-1, :]
            image_next = image_next[:, ::-1, :]

            height, width, _ = image.shape
            height_next, width_next, _ = image_next.shape
            assert (height == height_next) and (width == width_next), 'size must be equal between adjacent images pair.'

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            
            x_tmp = x1.copy()

            annots[:, 0] = width - x2
            annots[:, 2] = width - x_tmp

            # for next
            x1 = annots_next[:, 0].copy()
            x2 = annots_next[:, 2].copy()
            
            x_tmp = x1.copy()

            annots_next[:, 0] = width - x2
            annots_next[:, 2] = width - x_tmp

            sample = {'img': image, 'annot': annots, 'img_next': image_next, 'annot_next': annots_next}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[RGB_MEAN]])
        self.std = np.array([[RGB_STD]])

    def __call__(self, sample):

        image, annots = sample['img'], sample['annot']
        image_next, annots_next = sample['img_next'], sample['annot_next']

        return {'img':torch.from_numpy((image.astype(np.float32) / 255.0 - self.mean) / self.std), 'annot': torch.from_numpy(annots), 'img_next':torch.from_numpy((image_next.astype(np.float32) / 255.0-self.mean)/self.std), 'annot_next': torch.from_numpy(annots_next)}


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = RGB_MEAN
        else:
            self.mean = mean
        if std == None:
            self.std = RGB_STD
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def random_brightness(img, img_next):
    prob = np.random.uniform(0, 1)
    if prob < 0.5:
        delta = np.random.uniform(-0.125, 0.125) + 1
        img = ImageEnhance.Brightness(img).enhance(delta)
        img_next = ImageEnhance.Brightness(img_next).enhance(delta)
    return img, img_next


def random_contrast(img, img_next):
    prob = np.random.uniform(0, 1)
    if prob < 0.5:
        delta = np.random.uniform(-0.5, 0.5) + 1
        img = ImageEnhance.Contrast(img).enhance(delta)
        img_next = ImageEnhance.Contrast(img_next).enhance(delta)
    return img, img_next


def random_saturation(img, img_next):
    prob = np.random.uniform(0, 1)
    if prob < 0.5:
        delta = np.random.uniform(-0.5, 0.5) + 1
        img = ImageEnhance.Color(img).enhance(delta)
        img_next = ImageEnhance.Color(img_next).enhance(delta)
    return img, img_next


def random_hue(img, img_next):
    prob = np.random.uniform(0, 1)
    if prob < 0.5:
        delta = np.random.uniform(-18, 18)
        img_hsv = np.array(img.convert('HSV'))
        img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
        img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')

        img_next_hsv = np.array(img_next.convert('HSV'))
        img_next_hsv[:, :, 0] = img_next_hsv[:, :, 0] + delta
        img_next = Image.fromarray(img_next_hsv, mode='HSV').convert('RGB')
    return img, img_next



class PhotometricDistort(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image, annots, image_next, annots_next = sample['img'], sample['annot'], sample['img_next'], sample['annot_next']
        # Apply different distort order
        if np.random.uniform(0, 1) > 0.5:
            img = Image.fromarray(image)
            img_next = Image.fromarray(image_next)
            if np.random.uniform(0, 1) > 0.5:
                img, img_next = random_brightness(img, img_next)
                img, img_next = random_contrast(img, img_next)
                img, img_next = random_saturation(img, img_next)
                img, img_next = random_hue(img, img_next)
            else:
                img, img_next = random_brightness(img, img_next)
                img, img_next = random_saturation(img, img_next)
                img, img_next = random_hue(img, img_next)
                img, img_next = random_contrast(img, img_next)
            image = np.array(img)
            image_next = np.array(img_next)

        return {'img': image, 'annot': annots, 'img_next': image_next, 'annot_next': annots_next}

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, image_next):
        for t in self.transforms:
            img, image_next = t(img, image_next)
        return img, image_next


class RandomResize(object):
    def __init__(self, resize_range=(0.4, 1.0)):
        self.resize_range=resize_range
        assert isinstance(resize_range, tuple) and resize_range[0] < resize_range[1]

    def __call__(self, sample):
        image, annots, image_next, annots_next = sample['img'], sample['annot'], sample['img_next'], sample[
            'annot_next']
        height, width, _ = image.shape
        resize_h = int(np.random.uniform(self.resize_range[0] * height, self.resize_range[1] * height))
        resize_w =  int(resize_h/height*width)
        resize_h += (32 - resize_h % 32)
        resize_w += (32 - resize_w % 32)

        image = (255.0 * skimage.transform.resize(image, (resize_h, resize_w))).astype(np.uint8)
        image_next = (255.0 * skimage.transform.resize(image_next, (resize_h, resize_w))).astype(np.uint8)
        annots[:, 0:4:2] *= (resize_w / width)
        annots_next[:, 0:4:2] *= (resize_w / width)
        annots[:, 1:4:2] *= (resize_h / height)
        annots_next[:, 1:4:2] *= (resize_h / height)
        return {'img': image, 'annot': annots, 'img_next': image_next, 'annot_next': annots_next}


class RandomSampleCrop(object):
    def __init__(self, min_iou=0.25, crop_range=(0.3, 0.9)):
        self.min_iou = min_iou
        self.crop_range = crop_range
        assert isinstance(crop_range, tuple) and crop_range[0]<crop_range[1]

    def __call__(self, sample):
        image, annots, image_next, annots_next = sample['img'], sample['annot'], sample['img_next'], sample['annot_next']
        if np.random.uniform(0, 1) > 0.3:
            height, width, _ = image.shape
            crop_h = np.random.uniform(self.crop_range[0] * height, self.crop_range[1] * height)
            crop_w = crop_h/height*width

            crop_success = False
            # max trails (10)
            for _ in range(10):
                left = np.random.uniform(0, width - crop_w)
                top = np.random.uniform(0, height - crop_h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + crop_w), int(top + crop_h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = overlap_numpy(annots[:, :4], rect)
                overlap_next = overlap_numpy(annots_next[:, :4], rect)

                if overlap.max() < self.min_iou or overlap_next.max() < self.min_iou:
                    continue
                crop_success = True
                break
            if not crop_success:
                rect = np.array([0, 0, width, height])

            image = image[rect[1]:rect[3], rect[0]:rect[2], :]
            image_next = image_next[rect[1]:rect[3], rect[0]:rect[2], :]

            annots = annots[overlap > self.min_iou, :].copy()
            annots_next = annots_next[overlap_next > self.min_iou, :].copy()

            annots[:, :2] -= rect[:2]
            annots[:, 2:4] -= rect[:2]
            annots_next[:, :2] -= rect[:2]
            annots_next[:, 2:4] -= rect[:2]

        return {'img': image, 'annot': annots, 'img_next': image_next, 'annot_next': annots_next}


class AspectRatioBasedSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]
