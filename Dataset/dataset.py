import torch
from torchvision.datasets import VOCDetection
from pathlib import Path
import os
from torchvision.transforms.functional import normalize, to_tensor

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

CLASS_TO_IDX = {name: i + 1 for i, name in enumerate(VOC_CLASSES)}


class FastRCNNVOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, proposal_dir, image_set="train"):
        self.voc = VOCDetection(
            root=root, year="2007", image_set=image_set, download=False
        )
        self.proposal_dir = proposal_dir

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, index):
        image, target = self.voc[index]

        annotation = target["annotation"]
        boxes = []
        labels = []
        for obj in annotation["object"]:
            bbox = obj["bndbox"]
            x1 = float(bbox["xmin"]) - 1
            y1 = float(bbox["ymin"]) - 1
            x2 = float(bbox["xmax"])
            y2 = float(bbox["ymax"])
            boxes.append([x1, y1, x2, y2])
            labels.append(
                CLASS_TO_IDX[obj["name"]]
            )  # index + 1)  # VOC 类别从 1 开始，0 是背景
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(labels, dtype=torch.int64)
        # ----------------------------------------------
        image_id = Path(annotation["filename"]).stem
        proposal_path = os.path.join(self.proposal_dir, f"{image_id}.pt")
        proposals = torch.load(
            proposal_path, map_location="cpu", weights_only=True
        ).float().reshape(-1, 4)
        # VOCDetection 返回 PIL 图片；to_tensor 转成 [3,H,W]，并除以 255。
        image = to_tensor(image)
        image = normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "proposals": proposals,
        }


if __name__ == "__main__":
    testdataset = FastRCNNVOCDataset(
        root="./data",
        proposal_dir="/path/to/proposals",
        image_set="test",
    )
    data = testdataset[0]
    print("boxes.shape:", data["boxes"].shape)
    print("labels.shape:", data["labels"].shape)
    print("proposals.shape:", data["proposals"].shape)
