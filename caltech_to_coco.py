import os.path as osp
import os
from scipy.io import loadmat
import shutil
import json
import numpy as np

DATAPATH='./data/caltech/'


def open_seq_save_jpg(file, savepath):
    # read .seq file, and save the images into the savepath

    f = open(file, 'rb')
    string = f.read().decode('latin-1')
    splitstring = "\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46"
    # split .seq file into segment with the image prefix
    strlist = string.split(splitstring)
    f.close()
    count = 0
    # delete the image folder path if it exists
    if os.path.exists(savepath):
        shutil.rmtree(savepath)
    # create the image folder path
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    # deal with file segment, every segment is an image except the first one
    for img in strlist:
        filename = '{:06d}.jpg'.format(count)
        filenamewithpath = os.path.join(savepath, filename)
        # abandon the first one, which is filled with .seq header
        if count > 0:
            i = open(filenamewithpath, 'wb+')
            i.write(splitstring.encode('latin-1'))
            i.write(img.encode('latin-1'))
            i.close()
        count += 1

def gen_jpgs():
    img_root = osp.join(DATAPATH, 'Images')
    for s in os.listdir(img_root):
        if not s.startswith('set'):
            continue
        s_path = osp.join(img_root, s)
        for seq in os.listdir(s_path):
            if not seq.endswith('.seq'):
                continue
            seq_path = osp.join(s_path, seq)
            save_dir = osp.join(s_path, seq.split('.')[0])
            print(seq_path, save_dir)
            open_seq_save_jpg(seq_path, save_dir)

def vbb2coco():
    ann_root = osp.join(DATAPATH, 'annotations')
    out = {'images': [], 'annotations': [],
           'categories': [{'id': 1, 'name': 'person'}]}

    img_cnt = 0
    ann_cnt = 0

    for s in os.listdir(ann_root):
        if not s.startswith('set'):
            continue
        s_path = osp.join(ann_root, s)
        for vbb in os.listdir(s_path):
            if not vbb.endswith('.vbb'):
                continue
            vbb_path = osp.join(s_path, vbb)
            vbb_file = loadmat(vbb_path)
            objLists = vbb_file['A'][0][0][1][0]
            objLbl = [str(v[0]) for v in vbb_file['A'][0][0][4][0]]
            person_index_list = np.where(np.array(objLbl) == "person")[0]
            for frame_id, obj in enumerate(objLists):
                if len(obj) == 0:
                    continue
                file_name = osp.join(osp.join(s,vbb.split('.')[0]), '{:06d}.jpg'.format(frame_id+1))
                img_info = dict(file_name=file_name,
                                id=img_cnt)
                anns = []
                for id, pos, occl in zip(obj['id'][0], obj['pos'][0], obj['occl'][0]):
                    id = int(id[0][0]) - 1  # for matlab start from 1 not 0
                    if not id in person_index_list:  # only use bbox whose label is person
                        continue
                    pos = pos[0].tolist()
                    occl = int(occl[0][0])
                    is_ignore = True if occl==1 else False
                    # skip ignore
                    if is_ignore:
                        continue
                    ann = {'id': ann_cnt,
                           'category_id': 1 if not is_ignore else -1,
                           'image_id': img_cnt,
                           'bbox': pos,
                           'iscrowd': 1 if is_ignore else 0}
                    anns.append(ann)
                    ann_cnt += 1
                img_cnt += 1
                if len(anns) >4:
                    out['annotations'].extend(anns)
                    out['images'].append(img_info)
    print(len(out['images']),len(out['annotations']))
    out_path = osp.join(ann_root,'annotations.json')
    json.dump(out, open(out_path, 'w'), indent=2)

def random_show():
    ann_root = osp.join(DATAPATH, 'annotations')
    out_path = osp.join(ann_root, 'annotations.json')
    info = json.load(open(out_path))
    img_root = osp.join(DATAPATH, 'Images')
    img_path = osp.join(img_root,info['images'][1000]['file_name'])
    from matplotlib import pyplot as plt
    import cv2
    img = cv2.imread(img_path)
    plt.figure(figsize=(32, 16))
    for ann in info['annotations']:
        if ann['image_id'] == info['images'][1000]['id']:
            print(ann)
            cv2.rectangle(img, (int(ann['bbox'][0]), int(ann['bbox'][1])), (int(ann['bbox'][0]+ann['bbox'][2]), int(ann['bbox'][1]+ann['bbox'][3])), color=(0, 200, 0), thickness=1)
    plt.imshow(img)
    plt.show()

if __name__ =='__main__':
    # gen_jpgs()
    vbb2coco()
    #random_show()
