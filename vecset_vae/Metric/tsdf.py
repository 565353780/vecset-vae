import torch


@torch.no_grad()
def getTSDFAccPos(
    gt_tsdf: torch.Tensor,
    pred_tsdf: torch.Tensor,
    dist_max: float,
) -> float:
    """
    计算 TSDF 在 GT >= 0 区域的正确率。
    """
    valid_mask = (gt_tsdf >= 0.0) & (gt_tsdf <= 1.0)
    gt = gt_tsdf[valid_mask]
    pred = pred_tsdf[valid_mask]

    if gt.numel() == 0:
        return 0.0

    sign_match = (gt >= 0) == (pred >= 0)
    at_pos1 = gt >= 1.0
    at_mid = ~at_pos1

    correct = torch.zeros_like(gt, dtype=torch.bool)
    correct[at_mid] = (torch.abs(pred[at_mid] - gt[at_mid]) <= dist_max) & sign_match[
        at_mid
    ]
    correct[at_pos1] = (pred[at_pos1] >= 1.0) & sign_match[at_pos1]

    return correct.float().mean().item()


@torch.no_grad()
def getTSDFAccNeg(
    gt_tsdf: torch.Tensor,
    pred_tsdf: torch.Tensor,
    dist_max: float,
) -> float:
    """
    计算 TSDF 在 GT <= 0 区域的正确率。
    """
    valid_mask = (gt_tsdf <= 0.0) & (gt_tsdf >= -1.0)
    gt = gt_tsdf[valid_mask]
    pred = pred_tsdf[valid_mask]

    if gt.numel() == 0:
        return 0.0

    sign_match = (gt >= 0) == (pred >= 0)
    at_neg1 = gt <= -1.0
    at_mid = ~at_neg1

    correct = torch.zeros_like(gt, dtype=torch.bool)
    correct[at_mid] = (torch.abs(pred[at_mid] - gt[at_mid]) <= dist_max) & sign_match[
        at_mid
    ]
    correct[at_neg1] = (pred[at_neg1] <= -1.0) & sign_match[at_neg1]

    return correct.float().mean().item()
