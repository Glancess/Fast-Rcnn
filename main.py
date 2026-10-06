import torch
from pathlib import Path
from torch.utils.data import DataLoader

from Dataset.dataset import FastRCNNVOCDataset
from Model import Fastrcnn
from engine.train_one_epoch import train_one_epoch
from engine.eval import evaluate
from Dataset.dataloader import detection_collate_fn


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


if __name__ == "__main__":
    main()
