import os
import numpy as  np
import json

DATA_PATH='./data/mot17/'
OUT_PATH=None
SPLITS = ['train_half', 'val_half']

def mot_to_coco():
    train_path = os.path.join(DATA_PATH, 'train')
    seqs = os.listdir(train_path)

    half_train_dic = {'images': [], 'annotations': [],
                      'categories': [{'id': 1, 'name': 'person'}],
                      'videos': []}
    img_cnt = 0
    video_cnt = 0
    ann_cnt = 0

    for seq in sorted(seqs):
        if 'MOT17' not in seq:
            continue

        video_cnt += 1
        half_train_dic['videos'].append({'id': video_cnt})

        seq_path = os.path.join(train_path, seq)
        img_path = os.path.join(seq_path, 'img1')
        gt_path = os.path.join(seq_path, 'gt/gt.txt')
        imgs = os.listdir(img_path)
        num_imgs = len([img for img in imgs if 'jpg' in img])
        half_train_range = [0, num_imgs // 2]

        for i in range(half_train_range[0], half_train_range[1]):
            img_info = dict(file_name='train/{}/img1/{:06d}.jpg'.format(seq, i + 1),
                            id=img_cnt + i + 1,
                            frame_id=i + 1 - half_train_range[0],
                            video_id=video_cnt)
            half_train_dic['images'].append(img_info)
        print('{}:{} train imgs'.format(seq, len(half_train_dic['images'])))

        half_val_range = [num_imgs // 2, num_imgs]
        fw = open(os.path.join(seq_path, 'gt/half_val_image.txt'), 'w')
        for i in range(half_val_range[0], half_val_range[1]):
            fw.write('{:0>6d}.jpg\n'.format(i))
        fw.close()

        gts = np.loadtxt(gt_path, dtype=np.float32, delimiter=',')

        def output_half_gt(name, half_range):
            half_gts = np.array([
                gt for gt in gts if \
                int(gt[0]) - 1 >= half_range[0] and \
                int(gt[0]) - 1 < half_range[1]
            ], np.float32)
            print('{}-{}:{} anns'.format(seq, name, len(half_gts)))

            half_gts[:, 0] -= half_range[0]
            half_gts_out = os.path.join(seq_path, 'gt/gt_{}.txt'.format(name))
            fw = open(half_gts_out, 'w')
            for o in half_gts:
                fw.write(
                    '{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n'.format(
                        int(o[0]), int(o[1]), int(o[2]), int(o[3]), int(o[4]), int(o[5]),
                        int(o[6]), int(o[7]), o[8]))
            fw.close()
            return half_gts

        half_train_gts = output_half_gt('half_train', half_train_range)
        half_val_gts = output_half_gt('half_val', half_val_range)

        for o in half_train_gts:
            if float(o[8]) < 0.2:  # poor visibility
                continue
            if not (int(o[6]) == 1):  # uncertain object
                continue
            if int(o[7]) in [3, 4, 5, 6, 9, 10, 11]:  # not person
                continue
            if int(o[7]) in [2, 7, 8, 12] or float(o[8]) < 0.25:  # ignore person
                category_id = -1
            else:
                category_id = 1
            ann_cnt += 1
            ann = dict(id=ann_cnt,
                       category_id=category_id,
                       image_id=img_cnt + int(o[0]),
                       track_id=int(o[1]),
                       bbox=o[2:6].tolist())
            half_train_dic['annotations'].append(ann)
        img_cnt = len(half_train_dic['images'])

    json.dump(half_train_dic, open(os.path.join(DATA_PATH, 'half_train_annots.json'), 'w'), indent=2)

def check():
    info = json.load(open(os.path.join(DATA_PATH, 'half_train_annots.json')))
    id1 = [img['id'] for img in info['images']]
    id2 = [ann['image_id'] for ann in info['annotations']]
    id1 = set(id1)
    id2=set(id2)
    cnt = 0
    for id in id1:
        if id not in id2:
            cnt += 1
    print(cnt)

if __name__ == '__main__':
    mot_to_coco()






