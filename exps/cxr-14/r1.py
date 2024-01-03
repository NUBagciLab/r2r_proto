import torch.nn as nn

opt = {
    'version': 1, 

    # Dataset
    'dataset': {
        'name': 'cxr-14',
        'meta': {'image_folder':'images_256'},
        'produce': ['pid', 'img', 'label'], # orig|label_bc|label_mc|label_onehot
        'add_healty_as_label':False,

        'augs':{
            'train':[
                {'name':'RandomRotation', 'degrees':10},
                {'name':'RandomResizedCropAndInterpolation', 'size':256, 'scale':(0.8, 1.2), 'ratio':(1.0, 1.0), 'interpolation':'bicubic'},
                #{'name':'RandomResizedCrop', 'size':512, 'scale':(0.8, 1.2), 'ratio':(1.0, 1.0)},
                #{'name':'CenterCrop', 'size': 512},
                #{'name':'RandomHorizontalFlip', 'p':0.5},
                #{'name':'RandomVerticalFlip', 'p':0.5},
                #{'name':'ColorJitter', 'brightness':0.5, 'contrast':0.5, 'saturation':0.5, 'hue':0.5},
                {'name':'ToTensor'},
                {'name':'Normalize', 'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
            ],
            'test':[
                {'name':'Resize', 'size': 224},
                {'name':'ToTensor'},
                {'name':'Normalize', 'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
            ]
        },

        'workers':8,
        'batch_size': 128
    },

    # Model definition
    'model': {
        'arch': 'r2r_proto',
        'num_classes':14,

        'context_encoder': {
            'arch': 'r2r_cvt',
            'in_chans':3, 'act_layer':nn.GELU, 'norm_layer':nn.LayerNorm, 'init':'trunc_norm',
            'spec':{
                'INIT': 'trunc_norm',
                'NUM_STAGES': 3,
                
                #'ATTN_LAYER': ['Attention', 'Attention_v5','Attention'],
                'ATTN_LAYER': [['Attention'], ['Attention_v5']*2, ['Attention']*10],
                'PATCH_SIZE': [7, 3, 3],
                'PATCH_STRIDE': [4, 2, 2],
                'PATCH_PADDING': [2, 1, 1],
                'DIM_EMBED': [64, 192, 384],
                'NUM_HEADS': [1, 3, 6],
                'DEPTH': [1, 2, 10],
                'MLP_RATIO': [4.0, 4.0, 4.0],
                'ATTN_DROP_RATE': [0.0, 0.0, 0.0],
                'DROP_RATE': [0.0, 0.0, 0.0],
                'DROP_PATH_RATE': [0.0, 0.0, 0.1],
                'QKV_BIAS': [True, True, True],
                'CLS_TOKEN': [False, False, True],
                'POS_EMBED': [False, False, False],
                'QKV_PROJ_METHOD': ['dw_bn', 'dw_bn', 'dw_bn'],
                'KERNEL_QKV': [3, 3, 3],
                'PADDING_KV': [1, 1, 1],
                'STRIDE_KV': [2, 2, 2],
                'PADDING_Q': [1, 1, 1],
                'STRIDE_Q': [1, 1, 1]
            },
            'pretrain': 'weights/cvt/CvT-13-224x224-IN-1k.pth'
        },
    
        'last_layer': {
            'arch': 'fc',
            'bias': True,
            'in_channels': 384
        }
    },
    #'resume': None,


    # Optimizers    
    'optim': {'method':'Adam', 'weight_decay':1e-4},
    'lr_schedule': {'method': 'CosineAnnealingLR', 'T_max':100},
    'lr': 1e-4,

    # Loss Functions
    'loss':[
        ({'method':'WeightedBalanceLoss', 'gamma':2.0, 'apply_sigmoid':True}, 1.0),
    ],


    # Training schedule
    'start_epoch':0, 
    'epochs': 100,
    'warmup_epochs': 0,

    #'clip_grad_norm': 1.0,

    # Best model selection criteria
    # track {dict} --> accuracy:1 | macro avg_{precision,recall,f1-score}:1 | weighted avg_{precision,recall,f1-score}:1 | <label_id>_{precision,recall,f1-score}
    # {<metric>:1|0} 1: higher is better, 0: lower is better
    'track': {'auc':1},

    # Logging frequencies
    'print_freq': 100, # [iter]
    'image_freq': 40, # [iter]
    'val_print_freq': 100, # [iter]
    #'log_freq': 1, # Save network weigth distrubutinons [epoch]
    'save_freq': 5 # Model saving frequency [epoch]

}


def get():
    global opt
    return opt