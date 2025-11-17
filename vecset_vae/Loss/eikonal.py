import torch


def eikonal_loss_fn(s_pred, coords, s_gt, trunc):
    grads = torch.autograd.grad(
        outputs=s_pred,
        inputs=coords,
        grad_outputs=torch.ones_like(s_pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0].contiguous()  # ✅ 确保 contiguous

    mask = (s_gt.abs() < trunc).float()  # 只在截断范围内约束
    grad_norm = grads.norm(2, dim=-1)

    eik_loss = ((grad_norm - 1.0) ** 2) * mask
    if mask.sum() > 0:
        eik_loss = eik_loss.sum() / (mask.sum() + 1e-9)
    else:
        eik_loss = torch.tensor(0.0, device=s_pred.device)
    return eik_loss
