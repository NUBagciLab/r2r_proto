import torch
import torch.nn as nn


class MBOLayer(nn.Module):

    def __init__(self, in_channels, num_classes, w_c, w_A, return_the_first_brach=True):
        super().__init__()
        self.return_the_first_brach = return_the_first_brach

        self.branch1 = nn.Linear(in_channels, num_classes)
        self.branch2 = nn.Linear(in_channels, num_classes)

        self.wc = nn.Parameter(torch.FloatTensor(w_c), requires_grad=True) # [num_class]
        self.wA = nn.Parameter(torch.FloatTensor([w_A]), requires_grad=True) # single_number

        self.alpha = nn.Parameter(torch.rand(1)*5, requires_grad=True) # single_number
        self.beta = nn.Parameter(torch.rand(1)*5, requires_grad=True) # single_number
        

    def last_layer_parameters(self):
        return [self.wc, self.wA, self.alpha, self.beta] + list(self.branch1.parameters()) + list(self.branch2.parameters())

    def forward(self, x):
        o1 = self.branch1(x) * self.wc
        o2 = self.branch2(x) * self.wA

        out = o1 if self.return_the_first_brach else o2

        return out, {'branch1': o1, 'branch2':o2, 'wc':self.wc, 'wA': self.wA, 'alpha': self.alpha, 'beta':self.beta}