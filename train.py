import time
import os
random_seed = 20200804
os.environ['PYTHONHASHSEED'] = str(random_seed)
import copy
import argparse
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="7"
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
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import model
from dataloader import CSVDataset, collater, RandomFlip, AspectRatioBasedSampler, RandomResize, Normalizer, PhotometricDistort, RandomSampleCrop
from torch.utils.data import Dataset, DataLoader

#assert torch.__version__.split('.')[1] == '4'
use_gpu = torch.cuda.is_available()
print('use gpu:', use_gpu)

def main(args=None):

    parser     = argparse.ArgumentParser(description='Simple training script for training a CTracker network.')

    parser.add_argument('--dataset', default='csv', type=str, help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--model_dir', default='./ctracker/', type=str, help='Path to save the model.')
    parser.add_argument('--root_path', default='/dockerdata/home/changanwang/Dataset/Tracking/MOT17Det/', type=str, help='Path of the directory containing both label and images')
    parser.add_argument('--csv_train', default='train_annots.csv', type=str, help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', default='train_labels.csv', type=str, help='Path to file containing class list (see readme)')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--batch_size', help='Training batch size', type=int, default=8)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--show_interval', help='Show loss every N iters', type=int, default=20)
    parser.add_argument('--save_interval', help='save trained models every N epoch', type=int, default=10)

    parser = parser.parse_args(args)
    print(parser)

    print(parser.model_dir)
    if not os.path.exists(parser.model_dir):
       os.makedirs(parser.model_dir)

    # Create the data loaders
    if parser.dataset == 'csv':
        if (parser.csv_train is None) or (parser.csv_train == ''):
            raise ValueError('Must provide --csv_train when training on COCO,')

        if (parser.csv_classes is None) or (parser.csv_classes == ''):
            raise ValueError('Must provide --csv_classes when training on COCO,')
        transform = transforms.Compose([RandomSampleCrop(),
                                        RandomResize(),
                                        PhotometricDistort(),
                                        RandomFlip(),
                                        Normalizer()])
        #transform = transforms.Compose([Normalizer(), Augmenter(), Resizer()])
        dataset_train = CSVDataset(parser.root_path, train_file=os.path.join(parser.root_path, parser.csv_train), class_list=os.path.join(parser.root_path, parser.csv_classes), \
            transform=transform)#

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    # sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=32, collate_fn=collater, batch_sampler=sampler)

    # Create the model
    fcosTracker = model.FCOSTracker(backbone=dict(name='resnet'+str(parser.depth),pretrained=True),
                                    neck=dict(in_channels=[128,256,512], out_channels=128, out_indices=[0]),
                                    cls_head=dict(in_channels=128*2,out_channels=2,stack_convs=3),
                                    reg_head=dict(in_channels=128*2,out_channels=8,stack_convs=3),
                                    stride=[8],
                                    test_cfg=dict(score_thr=0.65),
                                    regress_range=((-1,1e8),),
                                    margin=1)
    if use_gpu:
        fcosTracker = torch.nn.DataParallel(fcosTracker).cuda()

    fcosTracker.training = True

    optimizer = optim.Adam(fcosTracker.parameters(), lr=1*1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    fcosTracker.train()

    print('Num training images: {}'.format(len(dataset_train)))
    total_iter = 0
    for epoch_num in range(parser.epochs):

        fcosTracker.train()
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

            classification_loss = 20*(classification_loss.mean())
            regression_loss = 0.05*(regression_loss.mean())

            loss = classification_loss + regression_loss
            if bool(loss == 0):
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(fcosTracker.parameters(), 0.1)

            optimizer.step()

            loss_hist.append(float(loss))
            epoch_loss.append(float(loss))

            if iter_num % parser.show_interval == 0:
                print(fcosTracker.module.reg_scale.data)
                print('Epoch: {} | Iter: {} | Cls loss: {:1.3f}  | Reg loss: {:1.3f} | Running loss: {:1.3f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

        if (epoch_num+1) % parser.save_interval == 0:
            torch.save(fcosTracker, os.path.join(parser.model_dir, 'ChainTracker_{}.pt'.format(epoch_num)))
        scheduler.step(np.mean(epoch_loss))

    fcosTracker.eval()
    torch.save(fcosTracker, os.path.join(parser.model_dir, 'ChainTracker_{}.pt'.format(parser.epochs)))

if __name__ == '__main__':
    main()
