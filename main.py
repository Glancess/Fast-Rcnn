import torch
from pathlib import Path
from torch.utils.data import DataLoader

from Dataset.dataset import FastRCNNVOCDataset
from Model import Fastrcnn
from engine.train_one_epoch import train_one_epoch
from engine.eval import evaluate
from engine.test import test
from Dataset.dataloader import detection_collate_fn


def test_best_model(
    model,
    data_root,
    save_dir,
    device,
    num_images=10,
    score_thresh=0.5,
    nms_thresh=0.3,
):
    # 先检查，不自动下载 test，也不把 val 当成 test。
    best_path = save_dir / "best.pth"
    test_split = data_root / "VOCdevkit/VOC2007/ImageSets/Main/test.txt"
    if not best_path.exists():
        print("没有 best.pth，跳过 test：", best_path)
        return
    if not test_split.exists():
        print("没有 VOC2007 test 数据，跳过 test：", test_split)
        return
    test_dataset = FastRCNNVOCDataset(root=data_root, image_set="test", download=False)
    # 用 best，而不是最后一轮的模型；只加载模型权重，不恢复 SGD。
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    if checkpoint.get("stage") != 4:
        raise ValueError(
            "测试需要 Faster R-CNN 第4阶段的 best，不能直接加载旧 Fast R-CNN 权重。"
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    print("\nLoad best model:", best_path, "val loss:", checkpoint["val_loss"])
    print("Test 图片总数：", len(test_dataset), "本次最多画：", num_images)
    print("RPN 在线生成候选框，不再需要 Selective Search 的 .pt 文件。")

    return test(
        model=model,
        dataset=test_dataset,
        device=device,
        save_dir=Path("./test_results"),
        num_images=num_images,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
    )


def main():

    # =========================================================
    # 1. device
    # =========================================================
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        # 现成的 torchvision RoIPool 先使用 CPU / CUDA，不走 MPS。
        device = torch.device("cpu")

    print("device:", device)

    # =========================================================
    # 2. 路径
    # =========================================================
    # data_root 指向包含 VOCdevkit 的目录；也可以复用旧项目的 VOC 数据目录。
    # 若服务器的 VOCdevkit 就在项目根目录，这一行改成 Path(".")。
    data_root = Path("/root/Faster-Rcnn/data")
    save_dir = Path("./checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    # False：正常训练，结束后加载 best 并画 test；True：直接加载 best，只画 test。
    only_test = False
    num_test_images = 10  # 只看前 10 张；想看多少就改多少，不自动跑完整 test。
    score_thresh = 0.5  # 类别分数太低的预测框不保留。
    nms_thresh = 0.3  # 同类别预测框重叠过大时，NMS 去掉低分框。

    if only_test:
        # 缺少文件时，先返回，不为了测试去下载 VGG 权重或数据。
        best_path = save_dir / "best.pth"
        test_split = data_root / "test/VOCdevkit/VOC2007/ImageSets/Main/test.txt"
        if not best_path.exists() or not test_split.exists():
            print("只测试需要第4阶段 best.pth 和 test.txt，请检查：")
            print(best_path, test_split, sep="\n")
            return
        model = Fastrcnn(weights=None).to(device)  # 参数马上由 best.pth 完整加载。
        test_best_model(
            model,
            data_root / "test",
            save_dir,
            device,
            num_images=num_test_images,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
        )
        return

    # =========================================================
    # 3. VOC train / val
    # =========================================================
    train_dataset = FastRCNNVOCDataset(
        root=data_root,
        image_set="train",
    )

    val_dataset = FastRCNNVOCDataset(
        root=data_root,
        image_set="val",
    )
    print("train images:", len(train_dataset), "val images:", len(val_dataset))

    # =========================================================
    # 4. DataLoader
    # =========================================================
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=detection_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        collate_fn=detection_collate_fn,
    )

    # =========================================================
    # 5. 第1阶段从 ImageNet VGG 开始。类名仍保留你的 Fastrcnn。
    # =========================================================
    model = Fastrcnn().to(device)

    # =========================================================
    # 6. 四步交替训练，不是每个 epoch 交替，也不是四个 loss 相加。
    # =========================================================
    stage_epochs = [
        20,
        15,
        20,
        10,
    ]  # 学习版配置，非论文的完整迭代数；改这里控制各阶段。
    proposal_model = None

    # =========================================================
    # 7. train
    # =========================================================
    for stage in [1, 2, 3, 4]:
        if stage == 2:
            # 保留第1阶段 RPN。第2阶段检测器重新从 ImageNet 开始，不沿用 RPN backbone。
            proposal_model = model
            proposal_model.eval()
            for param in proposal_model.parameters():
                param.requires_grad_(False)
            model = Fastrcnn().to(device)
        elif stage == 3:
            # 保留第2阶段的 backbone/检测头，接上第1阶段学好的 RPN 专用层。
            model.rpn_conv.load_state_dict(proposal_model.rpn_conv.state_dict())
            model.rpn_cls.load_state_dict(proposal_model.rpn_cls.state_dict())
            model.rpn_bbox.load_state_dict(proposal_model.rpn_bbox.state_dict())
            proposal_model = None  # 后面已经共享 backbone，不再需要另一份模型。

        # 先全冻结，再打开当前阶段需要更新的层。
        for param in model.parameters():
            param.requires_grad_(False)
        model.zero_grad(set_to_none=True)  # 清掉上一阶段留下的梯度。
        if stage == 1:
            train_layers = [
                model.backbone,
                model.rpn_conv,
                model.rpn_cls,
                model.rpn_bbox,
            ]
            loss_name = "RPN"
        elif stage == 2:
            train_layers = [
                model.backbone,
                model.classifier,
                model.cls_head,
                model.bbox_head,
            ]
            loss_name = "detector"
        elif stage == 3:
            train_layers = [model.rpn_conv, model.rpn_cls, model.rpn_bbox]
            loss_name = "RPN"
        else:
            train_layers = [model.classifier, model.cls_head, model.bbox_head]
            loss_name = "detector"
        for layer in train_layers:
            for param in layer.parameters():
                param.requires_grad_(True)

        # 每个阶段重新创建 SGD，不沿用上一阶段不同任务的 momentum。
        optimizer = torch.optim.SGD(
            [param for param in model.parameters() if param.requires_grad],
            lr=1e-3,
            momentum=0.9,
            weight_decay=5e-4,
        )
        best_val_loss = float("inf")
        save_path = save_dir / (f"stage{stage}_best.pth" if stage < 4 else "best.pth")
        num_epochs = stage_epochs[stage - 1]
        if num_epochs < 1:
            raise ValueError("每个阶段至少训练1轮，不能跳过后直接拿未训练权重测试。")

        for epoch in range(num_epochs):
            print(
                f"\n========== Stage {stage}/4 {loss_name} Epoch {epoch + 1}/{num_epochs} =========="
            )
            train_losses = train_one_epoch(
                model=model,
                optimizer=optimizer,
                data_loader=train_loader,
                device=device,
                stage=stage,
                proposal_model=proposal_model,
            )
            val_losses = evaluate(
                model=model,
                data_loader=val_loader,
                device=device,
                stage=stage,
                proposal_model=proposal_model,
            )
            print(
                f"train {loss_name} loss={train_losses['loss']:.4f} "
                f"cls={train_losses['cls_loss']:.4f} bbox={train_losses['bbox_loss']:.4f}"
            )
            print(
                f"val   {loss_name} loss={val_losses['loss']:.4f} "
                f"cls={val_losses['cls_loss']:.4f} bbox={val_losses['bbox_loss']:.4f}"
            )

            if val_losses["loss"] < best_val_loss:
                best_val_loss = val_losses["loss"]
                torch.save(
                    {
                        "stage": stage,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "train_split": "train",
                        "val_split": "val",
                    },
                    save_path,
                )
                print("Save stage best:", save_path, "val loss:", best_val_loss)

        # 下一阶段从当前阶段的 best 开始，而不是最后一轮开始。
        checkpoint = torch.load(save_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint

    print("\nTraining finished.")
    print("Best val loss:", best_val_loss)

    # =========================================================
    # 8. test：第4阶段 best -> RPN proposals -> 检测器修框 -> 分类NMS -> 画图。
    #    test 不参与训练或选择 best，GT 只用于画图对照，不参与生成预测。
    # =========================================================
    test_best_model(
        model,
        data_root / "test",
        save_dir,
        device,
        num_images=num_test_images,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
    )


if __name__ == "__main__":
    main()
