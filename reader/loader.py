from reader.transforms import build_transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from os.path import join
import numpy as np

from reader.cxr14_reader import cxr14

def build_loader(name, split, batch_size, workers, balanced=False, **kwargs):
    if name == 'cub200':
        dset = get_cub200(split, **kwargs)
    elif name == 'cxr-14':
        dset = cxr14(root='_datasets/cxr-14/', mode=split, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset name is given:', str(name))

    if split == 'train' and balanced:
        class_freq = np.array([dset.class_freq[c] for c in sorted(dset.class_freq)]).astype(np.float32)
        class_weights = class_freq.sum() / class_freq
        sample_weights = [class_weights[int(dset.data[i][-1])] for i in range(len(dset.data))]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        loader = DataLoader(dset, batch_size=batch_size, num_workers=workers, sampler=sampler)
    else:
        loader = DataLoader(dset, batch_size=batch_size, shuffle=(split=='train'), num_workers=workers)

    return loader


def get_cub200(split, meta, produce, scale_size, crop_size, scale=None, rotation=None, hflip=0, vflip=0):
    if split == 'train':
        xforms = build_transforms(scale_size=scale_size, crop_size=crop_size, scale=scale, rotation=rotation, hflip=hflip, vflip=vflip)
        dset = datasets.ImageFolder(join('_datasets', 'cub200-2011', meta['train_folder']), xforms)
    else:
        xforms = build_transforms(scale_size=scale_size, crop_size=crop_size)
        dset = datasets.ImageFolder(join('_datasets', 'cub200-2011', meta['test_folder']), xforms)

    return dset