import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys
from os.path import join
import time

import exps as configs
from util import utils
from reader.loader import build_loader
from optimization import builder as optim_builder
from networks import arch_builder
from util.logger import LogHandler
from util.model_tracker import ModelSaveTracker
from util.accuracy import AUCMeter


def main(opt):
    # Initialize logger
    log = LogHandler(opt)

    # Data loaders
    train_loader = build_loader(split='train', **opt['dataset'])
    eval_loader = build_loader(split='test', **opt['dataset'])

    #clip_grad_norm = opt.get('clip_grad_norm', 0.0)

    # Initialize network
    dev = torch.device('cuda:0')
    model = arch_builder.build(**opt['model'])
    model.to(dev)
    log.i(model)

    # Model Tracker
    save_tracker = ModelSaveTracker(freq=opt['save_freq'], track=opt['track'], prefix='v_')
    if 'resume' in opt: save_tracker.load(opt['netA']['resume'])

    # Build loss functions
    loss_seq = optim_builder.build_loss_seq(*opt['loss'])

    parameters = model.parameters()
    optimizer = optim_builder.build_optim(opt['optim']['method'], parameters, lr=opt['lr'])
    lr_scheduler = optim_builder.build_lr_scheduler(optimizer, **opt['lr_schedule']) if 'lr_schedule' in opt else None

    # Main training loop
    for epoch in range(opt['start_epoch'], opt['epochs']):
        train_logs = train(epoch, opt, model=model, dev=dev, dataloader=train_loader, optimizer=optimizer, loss_seq=loss_seq, log=log)
        if lr_scheduler is not None: lr_scheduler.step()

        val_logs = validation(epoch, opt, model=model, dev=dev, dataloader=eval_loader, loss_seq=loss_seq, log=log)
        save_tracker.update(epoch, model, model_root=opt['modeldir'], filename='chk_{}.pth.tar', **{**train_logs, **val_logs})



def train(epoch, opt, model, dev, dataloader, optimizer, loss_seq, log, use_l1_mask=True):
    start = time.time()
    model.train()
    auc_meter = AUCMeter()
    for i, batch in enumerate(dataloader):
        x, label = batch['img'].to(dev), batch['label'].float().to(dev)

        # Forward pass
        output, model_info = model(x)

        # Total Loss
        loss, loss_info = calc_total_loss(loss_seq, output, label, model_info)
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Apply Sigmoid for performance calculation
        output = torch.sigmoid(output)

        # Logging
        auc_meter.add(output.detach().cpu().numpy(), label.detach().cpu().numpy())
        
        # Visualization
        if i % opt['image_freq'] == -1:
            utils.save_images(epoch, i, tag=join(opt['outdir'], 'train'), r=(0.0, 1.0), img=x)

        # Print logs
        if (i+1) % opt['print_freq'] == 0:
            # Calc performance
            perf_map = {'t_'+k:v for k,v in auc_meter().items()}
            logmap = {'t_'+k:v for k,v in loss_info.items()}

            # Add log
            logmap = {**logmap, **perf_map}
            log.add_hist(epoch, logmap)

            # Print loss and performance info
            msg = '[{e:3d}/{nepoch}] T {i:3d}/{niter} '.format(e=epoch, nepoch=opt['epochs'], i=i, niter=len(dataloader))
            msg += ''.join(['[{k}: {v:.4f}]'.format(k=k[2:], v=v) for k,v in logmap.items()])
            log.i(msg)

    log.save()
    perf_map = auc_meter()
    perf_map = {'t_'+k:v for k,v in perf_map.items()}
    duration_str = '[t_time: {:.2f}]'.format(time.time() - start)
    log.i('[Epoh {}] Train '.format(epoch) + generate_msg(opt, perf_map, key_prefix='t_') + duration_str)
    return perf_map


def validation(epoch, opt, model, dev, dataloader, loss_seq, log, use_l1_mask=True):
    start = time.time()
    model.eval()
    auc_meter = AUCMeter()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x, label = batch['img'].to(dev), batch['label'].float().to(dev)

            # Forward pass
            output, model_info = model(x, mode='mean') if opt['model']['last_layer']['arch'].startswith('var_') else model(x)

            # Total Loss
            _, loss_info = calc_total_loss(loss_seq, output, label, model_info)

            # Apply Sigmoid for performance calculation
            output = torch.sigmoid(output)

            # Performance measurement and logging
            auc_meter.add(output.detach().cpu().numpy(), label.detach().cpu().numpy())

            # Visualization
            if i % opt['image_freq'] == -1:
                utils.save_images(epoch, i, tag=join(opt['outdir'], 'eval'), r=(0.0, 1.0), img=x)

            # Print logs
            if (i+1) % opt['val_print_freq'] == 0:
                # Calc performance
                perf_map = {'v_'+k:v for k,v in auc_meter().items()}
                logmap = {'v_'+k:v for k,v in loss_info.items()}

                # Add log
                logmap = {**logmap, **perf_map}
                log.add_hist(epoch, logmap)

                # Print loss and performance info
                msg = '[{e:3d}/{nepoch}] V {i:3d}/{niter} '.format(e=epoch, nepoch=opt['epochs'], i=i, niter=len(dataloader))
                msg += ''.join(['[{k}: {v:.4f}]'.format(k=k[2:], v=v) for k,v in logmap.items()])
                log.i(msg)

    log.save()
    perf_map = auc_meter()
    perf_map = {'v_'+k:v for k,v in perf_map.items()}
    duration_str = '[v_time: {:.2f}]'.format(time.time() - start)
    log.i('[Epoh {}] Test '.format(epoch) + generate_msg(opt, perf_map, key_prefix='v_') + duration_str)
    return perf_map


def calc_total_loss(loss_seq, output, label, model_info):
    loss = 0
    loss_info = {}

    for loss_conf in loss_seq:
        # Get loss function and weight
        loss_name, loss_func, wt = loss_conf['method'], loss_conf['f'], loss_conf['wt']

        # Calculate the loss
        if loss_name.startswith('BCE') or loss_name == 'WeightedBalanceLoss':
            loss_val = loss_func(output, label)
        else:
            loss_val = loss_func(output, label, model_info)

        # Apply loss weight
        loss_val_weighted = loss_val * wt
        loss += loss_val_weighted

        # Log
        loss_info[loss_name] = loss_val_weighted.item()

    loss_info['loss'] = loss.item()

    return loss, loss_info



def generate_msg(opt, logmap, key_prefix):
    msg, msgmap = '', {}
    for k in opt['track']:
        if (key_prefix + k) in logmap:
            msg += '['+ k +':{'+ k +':.4f}] '
            msgmap[k] = logmap[key_prefix+k]

    msg = msg.format(**msgmap)
    return msg


if __name__ == "__main__":
    assert len(sys.argv) == 2
    exp_name = sys.argv[1]
    #exp_name = 'cxr-14.r7'
    
    # Apply Configurations
    opt = configs.parse(exp_name)
    utils.mkdirs(opt['outdir'], opt['modeldir'], opt['logdir'], opt['slurmdir'])
    main(opt)