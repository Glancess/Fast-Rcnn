import torch
from torchvision.ops import box_iou


def assign_gt_boxes(
    proposals, gt_boxes, gt_labels
):  # N,M,proposals is 2000x4, gt_boxes is Mx4
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
