import time
import os
random_seed = 20200804
os.environ['PYTHONHASHSEED'] = str(random_seed)
import copy
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
import collections
import sys
import random
from tqdm import tqdm
random.seed(random_seed)
import numpy as np
np.random.seed(random_seed)

import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
import torch.optim as optim
from torchvision import datasets, models, transforms
import torchvision

import model
from dataloader import MOTDataset, CrowdHumanDataset, CaltechPedestrianDataset, \
    pad_collater, RandomFlip, AspectRatioBasedSampler, RandomResize, Normalizer, PhotometricDistort, RandomSampleCrop
from torch.utils.data import Dataset, DataLoader

#assert torch.__version__.split('.')[1] == '4'
use_gpu = torch.cuda.is_available()
print('use gpu:', use_gpu)

def main(args=None):

    parser     = argparse.ArgumentParser(description='Simple training script for training a CTracker network.')

    parser.add_argument('--dataset', default='MOT', type=str, help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--model_dir', default='./trained_model/', type=str, help='Path to save the model.')
    parser.add_argument('--root_path', default='/dockerdata/home/changanwang/Dataset/Tracking/MOT17Det/', type=str, help='Path of the directory containing both label and images')
    parser.add_argument('--train_file', default='half_train_annots.json', type=str, help='Path to file containing training annotations (see readme)')
    #parser.add_argument('--csv_classes', default='train_labels.csv', type=str, help='Path to file containing class list (see readme)')
    parser.add_argument('--backbone', help='Backbone.', type=str, default='resnet18')
    parser.add_argument('--batch_size', help='Training batch size', type=int, default=8)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--show_interval', help='Show loss every N iters', type=int, default=40)
    parser.add_argument('--save_interval', help='save trained models every N epoch', type=int, default=20)
    parser.add_argument('--resume_from', help='Whether to resume trained model', type=str, default=None)

    parser = parser.parse_args(args)
    print(parser)

    print(parser.model_dir)
    if not os.path.exists(parser.model_dir):
       os.makedirs(parser.model_dir)

    # Create the data loaders
    transform = transforms.Compose([RandomSampleCrop(),
                                    RandomResize(),
                                    PhotometricDistort(),
                                    RandomFlip(),
                                    Normalizer()])
    if parser.dataset == 'MOT':
        parser.root_path = './data/mot17'
        parser.train_file = parser.root_path+'/half_train_annots.json'
        dataset_train = MOTDataset(parser.root_path, train_file=parser.train_file, transform=transform)#

    elif parser.dataset == 'CrowdHuman':
        parser.root_path = './data/crowdhuman/CrowdHuman_train/Images'
        parser.train_file = './data/crowdhuman/annotations/train.json'
        dataset_train = CrowdHumanDataset(parser.root_path, train_file=parser.train_file, transform=transform)

    elif parser.dataset == 'CaltechPedestrian':
        parser.root_path = './data/caltech/Images'
        parser.train_file = './data/caltech/annotations/annotations.json'
        dataset_train = CaltechPedestrianDataset(parser.root_path, train_file=parser.train_file, transform=transform)
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    # sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=32, collate_fn=pad_collater, batch_sampler=sampler)

    # Create the model
    if parser.resume_from is not None:
        print('resume')
        fcosTracker = torch.load(os.path.join(parser.model_dir, parser.resume_from))
        if isinstance(fcosTracker, torch.nn.DataParallel):
            fcosTracker = fcosTracker.module
    else:
        fcosTracker = model.FCOSTracker(backbone=dict(name=parser.backbone,pretrained=True),
                                        neck=dict(in_channels=[128,256,512], out_channels=128, out_indices=[0]),
                                        cls_head=dict(in_channels=128*2,out_channels=2,stack_convs=3),
                                        reg_head=dict(in_channels=128*2,out_channels=9,stack_convs=3),
                                        stride=[8],
                                        test_cfg=dict(score_thr=0.65),
                                        regress_range=((-1,1e8),),
                                        margin=1)

    if use_gpu:
        fcosTracker = torch.nn.DataParallel(fcosTracker).cuda()

    fcosTracker.training = True

    optimizer = optim.Adam(fcosTracker.parameters(), lr=5*1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.5,cooldown=1,verbose=False)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=5,eta_min=1e-7)

    loss_hist = collections.deque(maxlen=len(dataset_train)//parser.batch_size)

    fcosTracker.train()
    #fcosTracker.module.freeze_backbone_bn()

    print('Num training images: {}'.format(len(dataset_train)))
    total_iter = 0
    for epoch_num in range(parser.epochs):

        epoch_loss = []
        for iter_num, data in enumerate(dataloader_train):
            total_iter = total_iter + 1
            optimizer.zero_grad()
            if use_gpu:
                input = [data['img'].cuda().float(), data['annot'], data['img_next'].cuda().float(), data['annot_next']]
            else:
                input = [data['img'].float(), data['annot'], data['img_next'].float(),
                         data['annot_next']]
            (classification_loss, regression_loss) = \
                fcosTracker(input, train_mode=True)

            classification_loss = (classification_loss.mean())
            regression_loss = (regression_loss.mean())
            loss_scale = fcosTracker.module.loss_scale
            # loss = classification_loss/(loss_scale[0]**2) + regression_loss/(loss_scale[1]**2) \
            #        +torch.log(torch.clamp((loss_scale[0]**2)*(loss_scale[1]**2),min=1))
            loss = classification_loss*20 + regression_loss/20
            if bool(loss == 0):
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(fcosTracker.parameters(), 0.1)

            optimizer.step()

            loss_hist.append(float(loss))
            epoch_loss.append(float(loss))

            if iter_num % parser.show_interval == 0:
                print('loss_scale:',loss_scale,' lr:', optimizer.param_groups[0]['lr'])
                print('Epoch: {} | Iter: {} | Cls loss: {:1.3f}  | Reg loss: {:1.3f} | Running loss: {:1.3f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

        if (epoch_num+1) % parser.save_interval == 0:
            torch.save(fcosTracker, os.path.join(parser.model_dir, '{}_{}_{}.pt'.format(parser.backbone, parser.dataset,epoch_num+1)))
        scheduler.step(np.mean(epoch_loss))

    fcosTracker.eval()
    #torch.save(fcosTracker, os.path.join(parser.model_dir, 'ChainTracker_{}.pt'.format(parser.epochs)))

if __name__ == '__main__':
    main()
