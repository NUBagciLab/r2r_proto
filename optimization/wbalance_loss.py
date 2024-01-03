import torch
import torch.nn as nn


class WeightedBalanceLoss(nn.Module):
    def __init__(self, gamma, apply_sigmoid=True, epsilon=1e-10) -> None:
        super(WeightedBalanceLoss, self).__init__()
        self.apply_sigmoid = apply_sigmoid
        self.gamma = gamma
        self.epsilon = epsilon


    def forward(self, pred, target):
        if self.apply_sigmoid:
            pred = torch.sigmoid(pred)

        positive_side = -(1-pred)**self.gamma * target * torch.log(pred)
        negative_side = -pred**self.gamma * (1 - target) * torch.log(1-pred)

        n_pos = torch.count_nonzero(target, dim=0)+self.epsilon
        n_neg = torch.count_nonzero(target==0, dim=0)+self.epsilon

        total_loss = (1/n_pos)*positive_side + (1/n_neg)*negative_side
        total_loss = total_loss.sum()#.sum(dim=0)

        return total_loss
