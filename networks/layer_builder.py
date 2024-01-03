import torch.nn as nn
from networks.feature import resnet, densenet, swin_transformer, maxvit, vgg

from networks.last_layers.multi_branch_output import MBOLayer
from networks.last_layers.cls_proto_2_logits import ClassWisePrototype2Logits


def build_feat_extractor(**backbone):
    # Build feature extractor
    if backbone['arch'].startswith('resnet') or backbone['arch'].startswith('wide_resnet'):
        feat_extractor, last_in_feat, last_out_feat = build_resnet(**backbone)
        del feat_extractor.fc
    elif backbone['arch'].startswith('densenet'):
        feat_extractor, last_in_feat, last_out_feat = build_densenet(**backbone)
        del feat_extractor.classifier
    elif backbone['arch'].startswith('vgg'):
        feat_extractor, last_in_feat, last_out_feat = build_vgg(**backbone)
        del feat_extractor.classifier
    elif backbone['arch'].startswith('swin'):
        feat_extractor, last_in_feat, last_out_feat = build_swin_transformer(**backbone)
        del feat_extractor.head
    elif backbone['arch'].startswith('maxvit'):
        feat_extractor, last_in_feat, last_out_feat = build_maxvit(**backbone)
        del feat_extractor.classifier
    else:
        raise NotImplementedError('Unknown backbone name ' + str(backbone['arch']))
    
    return feat_extractor, last_in_feat, last_out_feat


def build_layer(arch, **kwargs):
    if arch == 'fc':
        cls_module = build_fc_layer(**kwargs)
    elif arch == 'mbo':
        cls_module = MBOLayer(**kwargs)
    elif arch == 'class_wise_proto2logit':
        cls_module = ClassWisePrototype2Logits(**kwargs)
    else:
        raise NotImplementedError('Unknown cls layer architecture')


    return cls_module


def build_fc_layer(in_channels, num_classes, bias=True, backbone_arch=None):
    if backbone_arch is None:
        new_cls = nn.Linear(in_channels, num_classes, bias=bias)
    elif backbone_arch.startswith('resnet'):
        new_cls = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(in_channels, num_classes, bias=bias)
        )
    else:
        raise Exception("Unknown arch type for new fc build")
    
    return new_cls


def build_resnet(arch, **kwargs):
    if arch == 'resnet18':
        feat_extractor = resnet.resnet18(pretrained=True, progress=False, **kwargs)
    elif arch == 'resnet34':
        feat_extractor = resnet.resnet34(pretrained=True, progress=False, **kwargs)
    elif arch == 'resnet50':
        feat_extractor = resnet.resnet50(pretrained=True, progress=False, **kwargs)
    elif arch == 'resnet101':
        feat_extractor = resnet.resnet101(pretrained=True, progress=False, **kwargs)
    elif arch == 'resnet152':
        feat_extractor = resnet.resnet152(pretrained=True, progress=False, **kwargs)
    elif arch == 'resnext50_32x4d':
        feat_extractor = resnet.resnext50_32x4d(pretrained=True, progress=False, **kwargs)
    elif arch == 'resnext101_32x8d':
        feat_extractor = resnet.resnext101_32x8d(pretrained=True, progress=False, **kwargs)
    elif arch == 'wide_resnet50_2':
        feat_extractor = resnet.wide_resnet50_2(pretrained=True, progress=False, **kwargs)
    elif arch == 'wide_resnet101_2':
        feat_extractor = resnet.wide_resnet101_2(pretrained=True, progress=False, **kwargs)
    else:
        raise NotImplementedError('Unknown backbone arch ' + str(arch))

    last_in_feat = feat_extractor.fc.in_features
    last_out_feat = feat_extractor.fc.out_features

    return feat_extractor, last_in_feat, last_out_feat


def build_densenet(arch, **kwargs):
    if arch == 'densenet121':
        feat_extractor = densenet.densenet121(pretrained=True, progress=False, **kwargs)
    elif arch == 'densenet161':
        feat_extractor = densenet.densenet161(pretrained=True, progress=False, **kwargs)
    elif arch == 'densenet169':
        feat_extractor = densenet.densenet169(pretrained=True, progress=False, **kwargs)
    elif arch == 'densenet201':
        feat_extractor = densenet.densenet201(pretrained=True, progress=False, **kwargs)
    else:
        raise NotImplementedError('Unknown backbone arch ' + str(arch))

    last_in_feat = feat_extractor.classifier.in_features
    last_out_feat = feat_extractor.classifier.out_features

    return feat_extractor, last_in_feat, last_out_feat


def build_vgg(arch, **kwargs):
    if arch in vgg.__dict__:
        feat_extractor = vgg.__dict__[arch](pretrained=True, progress=False, **kwargs)
    else:
        raise NotImplementedError('Unknown backbone arch ' + str(arch))

    for layer in feat_extractor.features[::-1]:
        if type(layer) == nn.Conv2d:
            last_in_feat = layer.out_channels
            break

    #last_in_feat = feat_extractor.classifier[0].in_features
    last_out_feat = feat_extractor.classifier[-1].out_features

    return feat_extractor, last_in_feat, last_out_feat


def build_swin_transformer(arch, **kwargs):
    if arch == 'swin_t':
        feat_extractor = swin_transformer.swin_t(**kwargs)
    elif arch == 'swin_b':
        feat_extractor = swin_transformer.swin_b(**kwargs)
    elif arch == 'swin_s':
        feat_extractor = swin_transformer.swin_s(**kwargs)
    elif arch == 'swin_v2_t':
        feat_extractor = swin_transformer.swin_v2_t(**kwargs)
    elif arch == 'swin_v2_b':
        feat_extractor = swin_transformer.swin_v2_b(**kwargs)
    elif arch == 'swin_v2_s':
        feat_extractor = swin_transformer.swin_v2_s(**kwargs)
    else:
        raise NotImplementedError('Unknown backbone arch ' + str(arch))


    last_in_feat = feat_extractor.head.in_features
    last_out_feat = feat_extractor.head.out_features

    return feat_extractor, last_in_feat, last_out_feat


def build_maxvit(arch, **kwargs):
    if arch == 'maxvit_t':
        feat_extractor = maxvit.maxvit_t(**kwargs)
    else:
        raise NotImplementedError('Unknown backbone name: ' + str(arch))

    last_in_feat = feat_extractor.classifier[-1].in_features
    last_out_feat = feat_extractor.classifier[-1].out_features

    return feat_extractor, last_in_feat, last_out_feat

