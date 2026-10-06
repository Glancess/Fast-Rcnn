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
    proposal_dir,
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
    if not proposal_dir.is_dir():
        print("没有 proposal 目录，跳过 test：", proposal_dir)
        print("需要先给 test 图片生成 proposals，不是只生成 trainval。")
        return

    test_dataset = FastRCNNVOCDataset(
        root=data_root, proposal_dir=proposal_dir, image_set="test", download=False
    )
    # 用 best，而不是最后一轮的模型；只加载模型权重，不恢复 SGD。
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("\nLoad best model:", best_path, "val loss:", checkpoint["val_loss"])
    print("Test 图片总数：", len(test_dataset), "本次最多画：", num_images)
    print("如果缺少某张 test 图片的 .pt，会提示跳过；请先生成 test proposals。")

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
    # data_root 指向包含 VOCdevkit 的目录，和 utils/generate.py 的路径一致。
    # 若服务器的 VOCdevkit 就在项目根目录，这一行改成 Path(".")。
    data_root = Path("/root/Fast-Rcnn/data")
    proposal_dir = data_root / "VOCdevkit/VOC2007/SelectiveSearchProposals"

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
        test_split = data_root / "VOCdevkit/VOC2007/ImageSets/Main/test.txt"
        if (
            not best_path.exists()
            or not test_split.exists()
            or not proposal_dir.is_dir()
        ):
            print("只测试需要 best.pth、test.txt 和 test 的 proposals，请检查：")
            print(best_path, test_split, proposal_dir, sep="\n")
            return
        model = Fastrcnn(weights=None).to(device)  # 参数马上由 best.pth 完整加载。
        test_best_model(
            model,
            data_root,
            proposal_dir,
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
        proposal_dir=proposal_dir,
        image_set="train",
    )

    val_dataset = FastRCNNVOCDataset(
        root=data_root,
        proposal_dir=proposal_dir,
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
    # 5. model
    # =========================================================
    model = Fastrcnn().to(device)

    # =========================================================
    # 6. optimizer
    # =========================================================
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3,
        momentum=0.9,
        weight_decay=5e-4,
    )

    num_epochs = 10

    best_val_loss = float("inf")  # loss 越小越好。

    # =========================================================
    # 7. train
    # =========================================================
    for epoch in range(num_epochs):

        print(f"\n========== Epoch " f"{epoch + 1}/{num_epochs} ==========")

        # --------------------------
        # train
        # --------------------------
        train_losses = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
        )

        # --------------------------
        # val：只算 loss，不做反向传播，不算 mAP。
        # --------------------------
        val_losses = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
        )

        # --------------------------
        # 分开打印，方便判断是分类还是框回归出现问题。
        # --------------------------
        print(
            f"train loss={train_losses['loss']:.4f} "
            f"cls={train_losses['cls_loss']:.4f} "
            f"bbox={train_losses['bbox_loss']:.4f}"
        )
        print(
            f"val   loss={val_losses['loss']:.4f} "
            f"cls={val_losses['cls_loss']:.4f} "
            f"bbox={val_losses['bbox_loss']:.4f}"
        )

        # --------------------------
        # 保存 best
        # --------------------------
        if val_losses["loss"] < best_val_loss:

            best_val_loss = val_losses["loss"]

            save_path = save_dir / "best.pth"

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "train_split": "train",
                    "val_split": "val",
                },
                save_path,
            )

            print(f"Save best model: val loss={best_val_loss:.4f}")

    print("\nTraining finished.")
    print("Best val loss:", best_val_loss)

    # =========================================================
    # 8. test：best 权重 -> 全部 proposals -> 修框 -> NMS -> 画图。
    #    test 不参与训练或选择 best，GT 只用于画图对照，不参与生成预测。
    # =========================================================
    test_best_model(
        model,
        data_root / "test",
        proposal_dir,
        save_dir,
        device,
        num_images=num_test_images,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
    )


if __name__ == "__main__":
    main()
