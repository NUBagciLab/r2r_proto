import timm
from torchvision import transforms


def build_transforms(augs=[]):

    tlist = []
    for t in augs:
        name = t.pop('name')
        if name in transforms.__dict__:
            transform_func = transforms.__dict__[name](**t)
        elif name in timm.data.transforms.__dict__:
            transform_func = timm.data.transforms.__dict__[name](**t)
        else:
            raise NotImplementedError('Unknown augmentation method')

        tlist.append(transform_func)

    xform = transforms.Compose(tlist)

    return xform