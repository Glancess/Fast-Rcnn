from pathlib import Path
from multiprocessing import get_context
import cv2
import torch

# 两种启动方式都支持：python utils/generate.py / python -m utils.generate。
if __package__:
    from .generate_selective_search import selective_search
else:
    from generate_selective_search import selective_search


def generate_one_image(args):
    # 每个子进程只负责一张图片：读取 -> Selective Search -> 保存。
    # 必须放在文件最外层，不能写在另一个函数里面，否则多进程不好找到它。
    voc_root, image_id, max_proposals = args
    cv2.setNumThreads(1)
    torch.set_num_threads(1)  # 避免每个进程内部再开启很多线程，抢占 CPU。

    image_path = voc_root / "JPEGImages" / f"{image_id}.jpg"
    save_path = voc_root / "SelectiveSearchProposals" / f"{image_id}.pt"
    if save_path.exists():
        return image_id, None

    proposals = selective_search(str(image_path), max_proposals=max_proposals)

    # 先写临时文件，完整保存后再变成 .pt。
    # 中途停止时，下次不会把未保存完整的临时文件当作已完成的 proposal。
    temp_path = save_path.with_suffix(".tmp")
    torch.save(proposals, temp_path)
    temp_path.replace(save_path)
    return image_id, len(proposals)  # 只返回编号和数量，不在进程间传整批框。


def generate_voc_proposals(
    voc_root,
    split="trainval",
    max_proposals=2000,
    num_workers=4,
):
    if num_workers < 1:
        raise ValueError("num_workers 必须至少是 1。")
    voc_root = Path(voc_root)

    proposal_dir = voc_root / "SelectiveSearchProposals"
    split_file = voc_root / f"ImageSets/Main/{split}.txt"

    proposal_dir.mkdir(parents=True, exist_ok=True)

    with open(split_file, "r") as f:
        image_ids = [line.strip() for line in f]

    print(f"Split: {split}")
    print(f"Total images: {len(image_ids)}")

    tasks = []
    for image_id in image_ids:
        save_path = proposal_dir / f"{image_id}.pt"
        # 已经生成过就跳过
        if save_path.exists():
            print(f"{image_id} already exists, skip")
            continue
        tasks.append((voc_root, image_id, max_proposals))

    print(f"Remaining images: {len(tasks)}, num_workers: {num_workers}")
    if len(tasks) == 0:
        return

    # 开 num_workers 个进程，同时处理不同图片。spawn 让每个进程独立启动。
    # 谁先完成谁先打印，因此图片编号不一定按顺序，这是正常的。
    with get_context("spawn").Pool(processes=num_workers) as pool:
        for i, (image_id, num_proposals) in enumerate(
            pool.imap_unordered(generate_one_image, tasks), 1
        ):
            if num_proposals is None:
                print(f"[{i}/{len(tasks)}] {image_id} already exists, skip", flush=True)
            else:
                print(
                    f"[{i}/{len(tasks)}] {image_id}: [{num_proposals}, 4]", flush=True
                )


if __name__ == "__main__":
    generate_voc_proposals(
        voc_root="/root/Fast-Rcnn/data/test/VOCdevkit/VOC2007",
        split="Test",
        max_proposals=2000,
        num_workers=4,  # 你的服务器有 8 个可用 CPU，先试 4，也可以改成 8 对比速度。
    )
