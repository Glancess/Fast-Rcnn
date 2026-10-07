import torch
from utils.encode import encode_boxes
from utils.assignAndsampling import assign_gt_boxes, sample_rois
from utils.loss import FastRCNNLoss, RPNLoss

criterion = FastRCNNLoss()
rpn_criterion = RPNLoss()


@torch.no_grad()  # 验证只算 loss，不记录反向传播的计算图。
def evaluate(model, data_loader, device, stage=1, proposal_model=None):
    if stage not in [1, 2, 3, 4]:
        raise ValueError("stage 只能是 1、2、3、4。")
    if stage == 2 and proposal_model is None:
        raise ValueError("第2阶段需要固定的第1阶段 RPN。")
    model.eval()  # 关闭 VGG 分类器里的 Dropout，验证结果才稳定。
    if proposal_model is not None:
        proposal_model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_bbox_loss = 0.0
    total_rois = 0

    for batch in data_loader:
        images = batch["images"].to(device)
        image_sizes = batch["image_sizes"]
        gt_boxes_list = [x.to(device) for x in batch["boxes"]]
        gt_labels_list = [x.to(device) for x in batch["labels"]]

        if stage in [1, 3]:
            _, outputs = model.forward_rpn(images, image_sizes)
            loss, cls_loss, bbox_loss = rpn_criterion(
                outputs["rpn_cls_logits"], outputs["rpn_bbox_deltas"],
                outputs["anchors"], gt_boxes_list, image_sizes, random_sample=False,
            )
            num_images = len(images)
            total_loss += loss.item() * num_images
            total_cls_loss += cls_loss.item() * num_images
            total_bbox_loss += bbox_loss.item() * num_images
            total_rois += num_images
            continue

        if stage == 2:
            _, outputs = proposal_model.forward_rpn(images, image_sizes, proposal_training=True)
            feature = model.backbone(images)
        else:
            # 验证 loss 也保持2000个候选框，再固定抽64个；预测入口才用300个。
            feature, outputs = model.forward_rpn(images, image_sizes, proposal_training=True)
        proposals_list = outputs["proposals"]

        all_sampled_proposals = []
        all_labels = []
        all_bbox_targets = []

        # 要算验证 loss，也需要 GT 给 proposal 分配类别和回归目标。
        # 这不是更新参数，更不是把 val 图片加入训练集。
        for b in range(images.shape[0]):
            proposals = proposals_list[b]
            gt_boxes = gt_boxes_list[b]
            gt_labels = gt_labels_list[b]

            if len(proposals) == 0:
                all_sampled_proposals.append(proposals.new_empty((0, 4)))
                continue

            assignments = assign_gt_boxes(proposals, gt_boxes, gt_labels)
            # 和训练采用相同的前景/背景规则，但固定取框，不随机抽。
            selected_idx = sample_rois(
                assignments["labels"],
                num_rois=64,
                fg_fraction=0.25,
                random_sample=False,
            )
            sampled_proposals = proposals[selected_idx]
            all_sampled_proposals.append(sampled_proposals)
            if len(selected_idx) == 0:
                continue

            sampled_labels = assignments["labels"][selected_idx]
            encoded_targets = torch.zeros_like(sampled_proposals)
            fg_idx = torch.where(sampled_labels > 0)[0]
            if len(fg_idx) > 0:
                matched_idx = assignments["max_indices"][selected_idx[fg_idx]]
                encoded_targets[fg_idx] = encode_boxes(sampled_proposals[fg_idx], gt_boxes[matched_idx])
            all_labels.append(sampled_labels)
            all_bbox_targets.append(encoded_targets)

        if len(all_labels) == 0:
            continue

        labels = torch.cat(all_labels, dim=0)
        bbox_targets = torch.cat(all_bbox_targets, dim=0)
        cls_logits, bbox_logits = model.forward_head(feature, all_sampled_proposals)
        loss, cls_loss, bbox_loss = criterion(
            cls_logits, bbox_logits, labels, bbox_targets
        )

        # 注意：这里没有 zero_grad、backward、step，所以不改变模型参数。
        num_rois = len(labels)
        total_loss += loss.item() * num_rois
        total_cls_loss += cls_loss.item() * num_rois
        total_bbox_loss += bbox_loss.item() * num_rois
        total_rois += num_rois

    if total_rois == 0:
        raise ValueError("验证没有可用的 RoI，请检查 proposals 和 GT。")

    return {
        "loss": total_loss / total_rois,
        "cls_loss": total_cls_loss / total_rois,
        "bbox_loss": total_bbox_loss / total_rois,
    }
