import torch
import torch.nn as nn
from utils.assignAndsampling import assign_gt_anchors, sample_rois
from utils.encode import encode_boxes


class FastRCNNLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.classification_loss = nn.CrossEntropyLoss()
        self.bbox_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, class_logits, bbox_preds, class_labels, bbox_targets):
        cls_loss = self.classification_loss(class_logits, class_labels)
        # 每个 RoI 有 20 类的偏移：[R,80] -> [R,20,4]。
        bbox_preds = bbox_preds.reshape(-1, 20, 4)
        fg_indices = torch.where(class_labels > 0)[0]

        if len(fg_indices) > 0:
            # label 是 1~20，回归头下标是 0~19，所以要减 1。
            # 只取该 RoI 的真实类别对应的四个偏移，不训练其他类别的框。
            fg_labels = class_labels[fg_indices] - 1
            fg_preds = bbox_preds[fg_indices, fg_labels]
            fg_targets = bbox_targets[fg_indices]
            # 回归误差只来自前景；除以总 RoI 数，与分类 loss 的平均尺度对应。
            bbox_loss = self.bbox_loss(fg_preds, fg_targets) / len(class_labels)
        else:
            # 全是背景时不做框回归；这个零仍然连接在计算图里。
            bbox_loss = bbox_preds.sum() * 0.0

        loss = cls_loss + bbox_loss
        return loss, cls_loss, bbox_loss


class RPNLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.classification_loss = nn.CrossEntropyLoss()
        self.bbox_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, cls_logits, bbox_deltas, anchors_list, gt_boxes_list,
                image_sizes, random_sample=True):
        # cls_logits [B,A,2]；bbox_deltas [B,A,4]，未 softmax、未 detach。
        total_cls = cls_logits.sum() * 0.0
        total_bbox = bbox_deltas.sum() * 0.0
        num_images = 0
        for b in range(cls_logits.shape[0]):
            anchors = anchors_list[b]
            gt_boxes = gt_boxes_list[b]
            assignments = assign_gt_anchors(anchors, gt_boxes, image_sizes[b])
            selected_idx = sample_rois(
                assignments["labels"], num_rois=256, fg_fraction=0.5,
                random_sample=random_sample,
            )
            if len(selected_idx) == 0:
                raise ValueError("这张图没有有效 RPN anchor，请检查图片尺寸和 anchor scales。")
            labels = assignments["labels"][selected_idx]
            cls_loss = self.classification_loss(cls_logits[b, selected_idx], labels)
            fg_idx = selected_idx[labels == 1]
            if len(fg_idx) > 0:
                targets = encode_boxes(anchors[fg_idx], gt_boxes[assignments["max_indices"][fg_idx]])
                # 论文：lambda=10，Nreg 为特征图位置数；不足256时分类按实际样本平均。
                h, w = image_sizes[b]
                num_locations = ((h + 15) // 16) * ((w + 15) // 16)
                bbox_loss = 10.0 * self.bbox_loss(bbox_deltas[b, fg_idx], targets) / num_locations
            else:
                bbox_loss = bbox_deltas[b].sum() * 0.0
            total_cls = total_cls + cls_loss
            total_bbox = total_bbox + bbox_loss
            num_images += 1
        cls_loss = total_cls / num_images
        bbox_loss = total_bbox / num_images
        return cls_loss + bbox_loss, cls_loss, bbox_loss
