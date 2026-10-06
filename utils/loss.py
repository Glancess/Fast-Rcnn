import torch
import torch.nn as nn


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
