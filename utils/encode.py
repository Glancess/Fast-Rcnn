import torch
from torchvision.models.detection._utils import BoxCoder


def encode_boxes(
    proposals, gt_boxes
):  # compute the error between proposals and gt_boxes
    """
    proposals: [N,4]
    gt_boxes:  [N,4]

    return:
        targets: [N,4]
        每行 [tx, ty, tw, th]
    """
    # proposal is x1,y1,x2,y2
    # gt_boxes is x1,y1,x2,y2
    wp = proposals[:, 2] - proposals[:, 0]
    hp = proposals[:, 3] - proposals[:, 1]

    wg = gt_boxes[:, 2] - gt_boxes[:, 0]
    hg = gt_boxes[:, 3] - gt_boxes[:, 1]

    xp = (proposals[:, 0] + proposals[:, 2]) / 2
    yp = (proposals[:, 1] + proposals[:, 3]) / 2

    xg = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    yg = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2

    tx = (xg - xp) / wp
    ty = (yg - yp) / hp

    tw = torch.log(wg / wp)
    th = torch.log(hg / hp)
    targets = torch.stack(
        [tx, ty, tw, th], dim=1
    )  # this means the targets is a tensor with shape [N,4], each row is [tx, ty, tw, th]about the proposals and gt_boxes
    return targets
