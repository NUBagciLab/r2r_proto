import torch
import torch.nn as nn
import torch.nn.functional as F
from optimization import wbalance_loss

def build_loss(method, **kwargs):
    if method == 'WeightedBalanceLoss':
        loss = wbalance_loss.WeightedBalanceLoss(**kwargs)
    elif method in globals():
        # all of the methods expect args: (output, label, model_info)
        loss = globals()[method]
    else:
        loss = torch.nn.__dict__[method](**kwargs)
    return loss


def build_loss_seq(*loss_conf_arr):
    loss_seq = []
    for (conf, wt) in loss_conf_arr:
        loss_func = build_loss(**conf)
        loss_seq.append({'method':conf['method'], 'f': loss_func, 'wt':wt})
    return loss_seq


def build_optim(method, optim_specs, **kwargs):
    optimizer = torch.optim.__dict__[method](optim_specs, **kwargs)
    return optimizer


def build_lr_scheduler(optimizer, method, **kwargs):
    lr_scheduler = torch.optim.lr_scheduler.__dict__[method](optimizer, **kwargs)

    return lr_scheduler


###################### LOSS FUNCTIONS ######################

def hydra_vit_mlce_loss(output, label, model_info):
    branch2 = model_info['branch2']
    loss_val = F.binary_cross_entropy_with_logits(branch2, label)
    return loss_val


def hydra_vit_cl_loss(output, label, model_info):
    branch1, branch2, alpha, beta = model_info['branch1'], model_info['branch2'], model_info['alpha'], model_info['beta']
    loss_val = F.mse_loss(alpha*branch1, beta*branch2).sqrt()
    return loss_val


# Cluster Loss
def cluster_loss_v1(output, label, model_info):
    min_distances, prototype_shape, prototype_class_identity  = model_info['dist'], model_info['proto_vec'].shape, model_info['proto_idx']
    max_dist = (prototype_shape[1] * prototype_shape[2] * prototype_shape[3])
    pcid = prototype_class_identity[None,...].to(label.device)
    label_ext = label[:,None,:]#.to(dev)
    prototypes_of_correct_class = (pcid * label_ext).sum(dim=-1)#.to(dev)
    inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
    cluster_cost = torch.mean(max_dist - inverted_distances)

    return cluster_cost


def cluster_loss_v2(output, label, model_info, epsilon=1e-10):
    min_distances, prototype_shape, prototype_class_identity  = model_info['dist'], model_info['proto_vec'].shape, model_info['proto_idx']
    
    n_proto_per_class = prototype_class_identity.shape[0] // prototype_class_identity.shape[1]
    min_proto_dist_per_class = -F.max_pool1d(-min_distances[:,None,:], kernel_size=n_proto_per_class, stride=n_proto_per_class)[:,0,:]
    cluster_cost = label * min_proto_dist_per_class

    n_pos = torch.count_nonzero(label, dim=0)+epsilon
    #n_neg = torch.count_nonzero(label==0, dim=0)+epsilon

    cluster_cost = (1/n_pos[None, :]) * cluster_cost
    cluster_cost = cluster_cost.sum()

    return cluster_cost


# Seperation Loss
def separation_loss_v1(output, label, model_info):
    min_distances, prototype_shape, prototype_class_identity  = model_info['dist'], model_info['proto_vec'].shape, model_info['proto_idx']
    max_dist = (prototype_shape[1] * prototype_shape[2] * prototype_shape[3])
    pcid = prototype_class_identity[None,...].to(label.device)
    label_ext = label[:,None,:]#.to(dev)
    prototypes_of_correct_class = (pcid * label_ext).sum(dim=-1)#.to(dev)
    prototypes_of_wrong_class = 1 - prototypes_of_correct_class
    inverted_distances_to_nontarget_prototypes, _ = torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
    separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

    return separation_cost


def separation_loss_v2(output, label, model_info, epsilon=1e-10):
    min_distances, prototype_shape, prototype_class_identity  = model_info['dist'], model_info['proto_vec'].shape, model_info['proto_idx']
    
    n_proto_per_class = prototype_class_identity.shape[0] // prototype_class_identity.shape[1]
    min_proto_dist_per_class = -F.max_pool1d(-min_distances[:,None,:], kernel_size=n_proto_per_class, stride=n_proto_per_class)[:,0,:]
    separation_cost = (1 - label) * min_proto_dist_per_class

    #n_pos = torch.count_nonzero(label, dim=0)+epsilon
    n_neg = torch.count_nonzero(label==0, dim=0)+epsilon

    separation_cost = (1/n_neg[None, :]) * separation_cost
    separation_cost = separation_cost.sum()

    return separation_cost


def last_layer_l1(output, label, model_info):
    last_layer_weight = model_info['last_layer_weight']
    l1 = last_layer_weight.norm(p=1)
    return l1
    

def last_layer_l1_with_pcid(output, label, model_info):
    last_layer_weight, prototype_class_identity = model_info['last_layer_weight'], model_info['proto_idx']
    l1_mask = 1 - torch.t(prototype_class_identity).to(label.device)
    l1 = (last_layer_weight * l1_mask).norm(p=1)
    
    return l1


# Orthogonality Loss
def orthogonal_loss_v1(output, label, model_info):
    proto_presence = model_info['proto_presence']
    num_classes, _, num_descriptive = proto_presence.shape

    orthogonal_loss = torch.Tensor([0]).cuda()
    for c in range(0, proto_presence.shape[0], 1000):
        orthogonal_loss_p = torch.nn.functional.cosine_similarity(proto_presence.unsqueeze(2)[c:c+1000],
                                                                    proto_presence.unsqueeze(-1)[c:c+1000], dim=1).sum()
        orthogonal_loss += orthogonal_loss_p
    orthogonal_loss = orthogonal_loss / (num_descriptive * num_classes) - 1

    return orthogonal_loss


def orthogonal_loss_v2(output, label, model_info):
    proto_vec = model_info['proto_vec']
    num_proto, proto_dim, pH, pW = proto_vec.shape
    orthogonal_loss = torch.nn.functional.cosine_similarity(proto_vec.unsqueeze(0), proto_vec.unsqueeze(1), dim=2)
    orthogonal_loss = orthogonal_loss.sum() / (num_proto*num_proto*pH*pW) - 1

    return orthogonal_loss


# Proto Var Loss
def proto_kld(output, label, model_info):
    proto_mean, proto_logvar  = model_info['proto_vec'], model_info['proto_logvar']
    kld_loss = torch.mean(-0.5 * torch.sum(1 + proto_logvar - proto_mean ** 2 - proto_logvar.exp(), dim = 1), dim = 0).squeeze()
    
    return kld_loss


# Feature Var Loss
def feat_kld(output, label, model_info):
    bsize, f_mean, f_logvar = model_info['f_mean'].shape[0], model_info['f_mean'], model_info['f_logvar']
    f_mean, f_logvar = f_mean.view(bsize, -1), f_logvar.view(bsize, -1)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + f_logvar - f_mean ** 2 - f_logvar.exp(), dim = 1), dim = 0).squeeze()

    return kld_loss
