import torch
from torchvision.ops import box_iou


def assign_gt_boxes(
    proposals, gt_boxes, gt_labels
):  # N,M,proposals is 2000x4, gt_boxes is Mx4
    if len(gt_boxes) == 0:
        # 无物体的图片：所有候选框都是背景，不需要回归 GT。
        return {
            "labels": torch.zeros(len(proposals), dtype=torch.int64, device=proposals.device),
            "max_iou": proposals.new_zeros(len(proposals)),
            "max_indices": torch.zeros(len(proposals), dtype=torch.int64, device=proposals.device),
        }
    iou = box_iou(proposals, gt_boxes)  # [N,M]，每个 proposal 与每个 GT 的 IoU
    max_iou, max_indices = torch.max(
        iou, dim=1
    )  # get the index of the max iou for each proposal
    labels = gt_labels[
        max_indices
    ].clone()  # clone is used to avoid modifying the original gt_labels
    labels[(max_iou >= 0.1) & (max_iou < 0.5)] = (
        0  # bool mask to set the labels of proposals with iou between 0.1 and 0.5 to 0 (background)
    )
    labels[max_iou < 0.1] = -1
    return {
        "labels": labels,
        "max_iou": max_iou,
        "max_indices": max_indices,
    }


def sample_rois(
    labels, num_rois=64, fg_fraction=0.25, random_sample=True
):  # one image choose 64 rois, 25% foreground, 75% background
    fg_indices = torch.where(labels > 0)[0]
    bg_indices = torch.where(labels == 0)[0]

    num_fg = int(num_rois * fg_fraction)

    # 实际 foreground 不一定够
    num_fg = min(num_fg, len(fg_indices))

    num_bg = num_rois - num_fg
    num_bg = min(num_bg, len(bg_indices))

    # train 随机抽；val 按固定顺序取，避免每次验证因抽样不同而波动。
    if random_sample:
        fg_indices = fg_indices[torch.randperm(len(fg_indices), device=labels.device)]
        bg_indices = bg_indices[torch.randperm(len(bg_indices), device=labels.device)]
    fg_selected = fg_indices[:num_fg]
    bg_selected = bg_indices[:num_bg]

    selected_indices = torch.cat([fg_selected, bg_selected])  # shape: [num_fg + num_bg]
    # get the indices of the selected foreground and background proposals, concatenate them to get the final selected indices
    # example: if fg_selected is [1, 3, 5] and bg_selected is [0, 2, 4], then selected_indices will be [1, 3, 5, 0, 2, 4]
    return selected_indices


def assign_gt_anchors(anchors, gt_boxes, image_size):
    """RPN 专用：anchor [A,4] -> labels [A]，1=物体、0=背景、-1=忽略。"""
    h, w = image_size
    labels = torch.full((len(anchors),), -1, dtype=torch.int64, device=anchors.device)
    max_indices = torch.zeros(len(anchors), dtype=torch.int64, device=anchors.device)

    # RPN 训练忽略跨越真实图片边界的 anchor，也忽略 batch 的补零区域。
    inside = (
        (anchors[:, 0] >= 0) & (anchors[:, 1] >= 0)
        & (anchors[:, 2] <= w) & (anchors[:, 3] <= h)
    )
    inside_idx = torch.where(inside)[0]
    if len(inside_idx) == 0:
        return {"labels": labels, "max_indices": max_indices}
    if len(gt_boxes) == 0:
        labels[inside_idx] = 0
        return {"labels": labels, "max_indices": max_indices}

    iou = box_iou(anchors[inside_idx], gt_boxes)  # [有效anchor数,GT数]
    max_iou, matched_idx = iou.max(dim=1)  # 每个 anchor 最像哪个 GT。
    inside_labels = torch.full_like(matched_idx, -1)
    inside_labels[max_iou < 0.3] = 0
    inside_labels[max_iou >= 0.7] = 1

    # 即使某个 GT 没有 IoU>=0.7 的 anchor，也把其最佳匹配设为正样本。
    gt_best_iou, gt_best_idx = iou.max(dim=0)
    best_matches = ((iou == gt_best_iou[None, :]) & (gt_best_iou[None, :] > 0)).any(dim=1)
    inside_labels[best_matches] = 1
    inside_labels[gt_best_idx] = 1
    # 上面正样本规则有优先权：最佳匹配即使 <0.3，也不能再次变成背景。
    labels[inside_idx] = inside_labels
    max_indices[inside_idx] = matched_idx
    return {"labels": labels, "max_indices": max_indices}
