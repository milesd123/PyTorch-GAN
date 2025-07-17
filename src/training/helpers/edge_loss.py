import torch

def edge_loss(pred, target):

    # How much do the pixels change from left to right and up and down
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    # Compute the mean of the differences of the x and y components
    return (torch.abs(pred_dx - target_dx).mean() + torch.abs(pred_dy - target_dy).mean())
