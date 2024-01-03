import torch
import torch.nn as nn

class ClassWisePrototype2Logits(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.num_prototypes = in_channels
        self.num_prototypes_per_class = self.num_prototypes // self.num_classes

        assert(self.num_prototypes % self.num_classes == 0)

        self.w_pc = nn.Parameter(torch.ones(1, self.num_prototypes), requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1, 1, self.num_prototypes_per_class), requires_grad=False)

    def forward(self, x):
        logits  = self.w_pc * x
        logits = nn.functional.conv1d(logits[:,None,:], weight=self.ones, stride=self.num_prototypes_per_class)[:,0,:]

        return logits, {'last_layer_weight':self.w_pc}
    

    def last_layer_parameters(self):
        return [self.w_pc]