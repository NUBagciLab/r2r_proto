import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, roc_auc_score

class Accuracy():
    def __init__(self):
        pass
# Taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    def __call__(self, output, target, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size).item())
            return res


class RunningAccuracy():
    def __init__(self):
        self.preds = None#np.array([])
        self.target = None#np.array([])

    # y     : [batch, nclass]
    # label : [batch, 1]
    def add(self, y, label):
        y = (y >=0.5).astype(label.dtype)
        if self.preds is None and self.target is None:
            self.preds = y
            self.target = label
        else:
            self.preds = np.vstack((self.preds, y))
            self.target = np.vstack((self.target, label))


    def __call__(self):
        acc = accuracy_score(self.target, self.preds)
        return acc

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AUCMeter():
    def __init__(self):
        self.preds = None#np.array([])
        self.target = None#np.array([])

    # y     : mc: [batch, nclass]
    # label : mc: [batch, nclass]        
    def add(self, y, label):
        # Add entries to history
        if self.preds is None and self.target is None:
            self.preds = y
            self.target = label
        else:
            self.preds = np.concatenate((self.preds, y), axis=0)
            self.target = np.concatenate((self.target, label), axis=0)


    def __call__(self, average=None):
        outmap = {}
        try:
            outmap['auc'] = roc_auc_score(self.target, self.preds, average=average).mean()
        except:
            outmap['auc'] = 0.0


        return outmap