import torch
from utils.encode import encode_boxes
from utils.assignAndsampling import assign_gt_boxes, sample_rois
from utils.loss import FastRCNNLoss, RPNLoss

criterion = FastRCNNLoss()
rpn_criterion = RPNLoss()


def train_one_epoch(model, optimizer, data_loader, device, stage=1, proposal_model=None):
    if stage not in [1, 2, 3, 4]:
        raise ValueError("stage 只能是 1、2、3、4。")
    if stage == 2 and proposal_model is None:
        raise ValueError("第2阶段需要固定的第1阶段 RPN。")
    model.train()
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
            # 第1/3阶段只训练 RPN，不能把分类器 loss 混进来。
            _, outputs = model.forward_rpn(images, image_sizes)
            loss, cls_loss, bbox_loss = rpn_criterion(
                outputs["rpn_cls_logits"], outputs["rpn_bbox_deltas"],
                outputs["anchors"], gt_boxes_list, image_sizes,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_images = len(images)
            total_loss += loss.item() * num_images
            total_cls_loss += cls_loss.item() * num_images
            total_bbox_loss += bbox_loss.item() * num_images
            total_rois += num_images  # RPN 按图片平均；检测阶段按 RoI 平均。
            continue

        if stage == 2:
            # 两个独立 backbone：固定 RPN 产生框，新的 ImageNet VGG 学检测。
            with torch.no_grad():
                _, outputs = proposal_model.forward_rpn(images, image_sizes, proposal_training=True)
            feature = model.backbone(images)
        else:
            # 第4阶段已经共享 backbone，卷积和 RPN 都冻结，只让 FC/检测头学习。
            with torch.no_grad():
                feature, outputs = model.forward_rpn(images, image_sizes, proposal_training=True)
        proposals_list = outputs["proposals"]

        all_sampled_proposals = []
        all_labels = []
        all_bbox_targets = []

        # 一张图片对应一个 proposal 张量；这个顺序告诉 RoIPool 去哪张特征图取区域。
        for b in range(images.shape[0]):
            proposals = proposals_list[b]
            gt_boxes = gt_boxes_list[b]
            gt_labels = gt_labels_list[b]

            if len(proposals) == 0:
                # 即使没有可用框，也保留该图片在列表中的位置，不能让图片编号错位。
                all_sampled_proposals.append(proposals.new_empty((0, 4)))
                continue

            # 1. 每个 proposal 找 IoU 最大的 GT，得到类别与匹配的 GT 下标。
            assignments = assign_gt_boxes(proposals, gt_boxes, gt_labels)
            # 2. 每张最多抽 64 个 RoI，前景最多占 25%；不足时按实际数量取。
            selected_idx = sample_rois(
                assignments["labels"], num_rois=64, fg_fraction=0.25
            )
            sampled_proposals = proposals[selected_idx]
            all_sampled_proposals.append(sampled_proposals)
            if len(selected_idx) == 0:
                continue

            sampled_labels = assignments["labels"][selected_idx]
            # 3. proposal 与匹配 GT 算 [dx,dy,dw,dh]；背景的回归目标不参与 loss。
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

        # 训练才执行这三步：清空梯度 -> 反向传播 -> 更新参数。
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 按实际 RoI 数加权，不让最后一个小 batch 占过大的权重。
        num_rois = len(labels)
        total_loss += loss.item() * num_rois
        total_cls_loss += cls_loss.item() * num_rois
        total_bbox_loss += bbox_loss.item() * num_rois
        total_rois += num_rois

    if total_rois == 0:
        raise ValueError("训练没有可用的 RoI，请检查 proposals 和 GT。")

    return {
        "loss": total_loss / total_rois,
        "cls_loss": total_cls_loss / total_rois,
        "bbox_loss": total_bbox_loss / total_rois,
    }
