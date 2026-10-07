import torch
from torchvision.ops import nms, remove_small_boxes
from utils.decode import decode_boxes


def reshape_rpn_outputs(rpn_cls_logits, rpn_bbox_deltas):
    """卷积输出 -> cls [B,A,2]、delta [B,A,4]；A = H * W * 9。"""
    B, _, H, W = rpn_cls_logits.shape
    k = rpn_bbox_deltas.shape[1] // 4
    # 顺序统一为：第几行 -> 第几列 -> 该位置的第几个 anchor。
    # 每个 anchor 的两个分类通道依次是 background、object。
    cls = rpn_cls_logits.reshape(B, k, 2, H, W)
    cls = cls.permute(0, 3, 4, 1, 2).reshape(B, -1, 2)
    deltas = rpn_bbox_deltas.reshape(B, k, 4, H, W)
    deltas = deltas.permute(0, 3, 4, 1, 2).reshape(B, -1, 4)
    return cls, deltas


@torch.no_grad()
def generate_proposals(
    anchors_list,
    rpn_cls_logits,
    rpn_bbox_deltas,
    image_sizes,
    pre_nms_topk=6000,
    post_nms_topk=300,
    nms_thresh=0.7,
    min_size=1.0,
):
    """
    anchors_list:
        长度 B 的列表，每项 [A,4]，xyxy

    rpn_cls_logits:
        [B,2k,H,W]

    rpn_bbox_deltas:
        [B,4k,H,W]

    image_sizes:
        [(h0,w0), (h1,w1), ...]

    return:
        proposals_list:
        [
            [N0,4],
            [N1,4],
            ...
        ]
    """

    # =====================================================
    # 1. reshape cls
    # =====================================================

    cls, deltas = reshape_rpn_outputs(rpn_cls_logits, rpn_bbox_deltas)
    B = cls.shape[0]

    # [B,A]
    objectness = torch.softmax(cls, dim=-1)[..., 1]

    # =====================================================
    # 2. reshape bbox
    # =====================================================

    # delta 已在上面和分类一起整理好，仍然逐行对应同一个 anchor。
    # 这里只生成候选框，不算 loss；RPN loss 要用保留计算图的原始输出。

    proposals_list = []

    # =====================================================
    # 3. 一张图一张图生成 proposal
    # =====================================================

    for b in range(B):

        scores = objectness[b]  # [A]
        bbox_deltas = deltas[b]  # [A,4]

        anchors = anchors_list[b]
        if len(anchors) != len(bbox_deltas):
            raise ValueError("anchor 数量与 RPN 输出数量不一致。")
        # anchor + predicted delta
        proposals = decode_boxes(anchors, bbox_deltas)

        h, w = image_sizes[b]

        # batch 补零区域不是这张图的内容，不能让其中心落在图外的 anchor 抢名额。
        centers = (anchors[:, :2] + anchors[:, 2:]) / 2
        valid = (
            torch.isfinite(proposals).all(dim=1)
            & torch.isfinite(scores)
            & (centers[:, 0] >= 0) & (centers[:, 0] < w)
            & (centers[:, 1] >= 0) & (centers[:, 1] < h)
        )
        proposals = proposals[valid]
        scores = scores[valid]

        # ----------------------------------------------
        # clip
        # ----------------------------------------------
        """
            proposal like 
            [
            [x y x y],[x y x y],[x y x y]
            ]
            solve over images w h
        """
        proposals[:, 0::2].clamp_(0, w)
        proposals[:, 1::2].clamp_(0, h)

        # ----------------------------------------------
        # 去掉太小的框
        # ----------------------------------------------

        keep = remove_small_boxes(proposals, min_size=min_size)

        proposals = proposals[keep]
        scores = scores[keep]

        # ----------------------------------------------
        # objectness 排序
        # 先只留一部分，避免直接对几万个框做 NMS
        # ----------------------------------------------

        num_topk = min(pre_nms_topk, scores.numel())

        scores, order = scores.sort(descending=True)

        order = order[:num_topk]

        proposals = proposals[order]
        scores = scores[:num_topk]

        # ----------------------------------------------
        # NMS
        # ----------------------------------------------

        keep = nms(proposals, scores, iou_threshold=nms_thresh)

        keep = keep[:post_nms_topk]

        proposals = proposals[keep]

        proposals_list.append(proposals)

    return proposals_list
