import torch.nn.functional as F
import torch

"""
define loss function and printing metrics function. 
Here, loss is the combination of dice loss and binary cross entropy.
loss = a * dice loss + (1-a) * binary cross entropy.
BCE, Dice loss and total loss are appended to metrics.
"""


# define dice loss, return the mean of dice loss
def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    # pred and target shape: (batch size, 3, H, W)
    # intersection and loss shape: (batch size, 3)
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) /
                 (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()


# define the total loss, return loss and metrics
def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    # use sigmoid function: pred -> (0, 1)
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss, metrics


# print all elements in metrics
def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    print("{}: {}".format(phase, ", ".join(outputs)))

