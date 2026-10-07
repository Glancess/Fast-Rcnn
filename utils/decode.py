import math
import torch


def decode_boxes(proposals, deltas):
    """
    proposals: [N, 4]
    deltas:    [N, 4]

    return:
        boxes: [N, 4]
    """
    # proposals is x1,y1,x2,y2
    # deltas is tx,ty,tw,th
    wp = proposals[:, 2] - proposals[:, 0]
    hp = proposals[:, 3] - proposals[:, 1]

    xp = (proposals[:, 0] + proposals[:, 2]) / 2
    yp = (proposals[:, 1] + proposals[:, 3]) / 2

    tx = deltas[:, 0]
    ty = deltas[:, 1]
    # 网络刚开始训练时可能预测很大的 dw/dh，限制 exp 的输入，避免溢出。
    tw = deltas[:, 2].clamp(max=math.log(1000.0 / 16))
    th = deltas[:, 3].clamp(max=math.log(1000.0 / 16))

    xg = tx * wp + xp
    yg = ty * hp + yp

    wg = torch.exp(tw) * wp
    hg = torch.exp(th) * hp

    x1 = xg - wg / 2
    y1 = yg - hg / 2
    x2 = xg + wg / 2
    y2 = yg + hg / 2

    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    return boxes
