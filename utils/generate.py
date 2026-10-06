from pathlib import Path
import torch
from utils.generate_selective_search import selective_search


def generate_voc_proposals(
    voc_root,
    split="trainval",
    max_proposals=2000,
):
    voc_root = Path(voc_root)

    image_dir = voc_root / "JPEGImages"
    proposal_dir = voc_root / "SelectiveSearchProposals"
    split_file = voc_root / f"ImageSets/Main/{split}.txt"

    proposal_dir.mkdir(parents=True, exist_ok=True)

    with open(split_file, "r") as f:
        image_ids = [line.strip() for line in f]

    print(f"Split: {split}")
    print(f"Total images: {len(image_ids)}")

    for i, image_id in enumerate(image_ids, 1):

        image_path = image_dir / f"{image_id}.jpg"
        save_path = proposal_dir / f"{image_id}.pt"

        # 已经生成过就跳过
        if save_path.exists():
            print(f"[{i}/{len(image_ids)}] {image_id} already exists, skip")
            continue

        proposals = selective_search(
            str(image_path),
            max_proposals=max_proposals,
        )

        torch.save(proposals, save_path)

        print(f"[{i}/{len(image_ids)}] " f"{image_id}: {proposals.shape}")


if __name__ == "__main__":
    generate_voc_proposals(
        voc_root="./data/VOCdevkit/VOC2007",
        split="trainval",
        max_proposals=2000,
    )
