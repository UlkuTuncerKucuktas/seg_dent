import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, pred, target):
        n, c, h, w = pred.size()
        pred = F.softmax(pred, dim=1)
        target = target.float()

        dist_maps = self.compute_dist_maps(target)
        loss = (pred * dist_maps).sum() / (dist_maps.sum() + 1e-10)
        return loss

    def compute_dist_maps(self, target):
        dist_maps = []
        for i in range(target.size(0)):
            dist_map = distance_transform_edt(target[i].cpu().numpy() == 0)
            dist_maps.append(dist_map)
        dist_maps = torch.tensor(dist_maps, dtype=torch.float32).to(target.device)
        return dist_maps


def dice_coefficient(pred, target):
    """Calculate Dice Coefficient for individual prediction and target."""
    smooth = 1.0  # Smoothing factor to avoid division by zero

    # Flatten the tensors to convert them into vectors
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    # Dice score
    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice
