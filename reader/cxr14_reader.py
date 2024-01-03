import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

import os
from os.path import join
import numpy as np
import json

from reader.transforms import build_transforms


LABEL_MAPPING_WITH_HEALTY = {
    'No Finding': 0,
    'Atelectasis': 1,
    'Cardiomegaly': 2,
    'Effusion': 3, 
    'Infiltration': 4, 
    'Mass': 5, 
    'Nodule': 6, 
    'Pneumonia': 7, 
    'Pneumothorax': 8, 
    'Consolidation': 9,
    'Edema':10, 
    'Emphysema': 11, 
    'Fibrosis': 12, 
    'Pleural_Thickening': 13, 
    'Hernia': 14, 
}



LABEL_MAPPING = {
    'Atelectasis': 0,
    'Cardiomegaly': 1,
    'Effusion': 2, 
    'Infiltration': 3, 
    'Mass': 4, 
    'Nodule': 5, 
    'Pneumonia': 6, 
    'Pneumothorax': 7, 
    'Consolidation': 8,
    'Edema':9, 
    'Emphysema': 10, 
    'Fibrosis': 11, 
    'Pleural_Thickening': 12, 
    'Hernia': 13, 
}


# Setup function to create dataloaders for image datasets
def cxr14(root, mode, meta, produce=None, augs=None, add_healty_as_label=False, only_bbox_label=False):
    
    label_json_file = join(root, 'test.json') if mode == 'test' else join(root, 'train_val.json')
    bboxes = get_bboxes(join(root, 'BBox_List_2017.csv'))
    
    with open(label_json_file, 'r') as jfile:
        labels = json.load(jfile)

    image_list = []
    for pid in sorted(labels.keys()):
        if only_bbox_label:
            if pid in bboxes:
                image_list.append({
                    'image':os.path.join(root, meta['image_folder'], pid),
                    'labels': labels[pid]['labels'],
                    'pid': pid,
                    'bboxes': bboxes[pid]
                })
        else:
            image_list.append({
                'image':os.path.join(root, meta['image_folder'], pid),
                'labels': labels[pid]['labels'],
                'pid': pid
            })

    add_healthy = False if mode == 'test' else add_healty_as_label


    if augs is None:
        xform = build_transforms([
            {'name':'ToTensor'},
            {'name':'Normalize', 'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
        ])
    elif mode in augs:
        xform = build_transforms(augs[mode])
    else:
        xform = build_transforms([
            {'name':'ToTensor'},
            {'name':'Normalize', 'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
        ])
        
    dset = CXR8Dataset(image_list, add_healty_as_label=add_healthy, xform=xform)

    return dset


def get_bboxes(bbox_file):
    bboxes = {}
    with open(bbox_file, 'r') as bfile:
        next(bfile)
        for l in bfile:
            fname, label_name, x,y,w,h = l.replace('\n', '').split(',')
            x,y,w,h = float(x), float(y), float(w), float(h)
            if fname not in bboxes:
                bboxes[fname] = {label_name: [x,y,w,h]}
            else:
                bboxes[fname][label_name] = [x,y,w,h]
    return bboxes


def multi_hot_encoding(label_set, num_classes):
    ml = np.zeros((num_classes,), dtype=float)
    ml[list(label_set)] = 1
    return ml

class CXR8Dataset(Dataset):
    def __init__(self, dataarr, add_healty_as_label=False, xform=None):
        super().__init__()

        self.data = dataarr
        self.add_healty_as_label = add_healty_as_label
        self.transform = xform if xform is not None else transforms.ToTensor()

        # Count class frequiency
        self.class_freq = {}
        for img_entry in self.data:
            labels = img_entry['labels']
            for l in labels:  
                if l in self.class_freq: 
                    self.class_freq[l] += 1
                else:
                    self.class_freq[l] = 1

    def __len__(self):
        return len(self.data)

    def read_img(self, path):
        img = Image.open(path).convert("RGB")
        return img

    def __getitem__(self, index):
        img_path, label_list, pid = self.data[index]['image'], self.data[index]['labels'], self.data[index]['pid']
        bboxes = self.data[index]['bboxes'] if 'bboxes' in self.data[index] else None
        # Read Image
        img = self.read_img(img_path)
        if self.transform is not None:
            img = self.transform(img)

        # Multi class label
        if self.add_healty_as_label:
            label = multi_hot_encoding(set([LABEL_MAPPING_WITH_HEALTY[l] for l in label_list]), num_classes=len(LABEL_MAPPING_WITH_HEALTY))
        else:
            if 'No Finding' in label_list: label_list = []
            label = multi_hot_encoding(set([LABEL_MAPPING[l] for l in label_list]), num_classes=len(LABEL_MAPPING))

        out = {'img': img, 'label': label, 'pid':pid}
        if bboxes is not None: out['bboxes'] = bboxes
        return out
        #return img, label
