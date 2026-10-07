import torch
from torchvision.datasets import VOCDetection
from PIL import Image
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
    # 保留你原来的类名；Faster R-CNN 的 Dataset 不再读取离线 proposals。
    def __init__(
        self, root, image_set="train", download=False, min_size=600, max_size=1000
    ):
        self.voc = VOCDetection(
            root=root, year="2007", image_set=image_set, download=download
        )
        self.min_size = min_size
        self.max_size = max_size

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
        # 固定短边600、长边最多1000，和128/256/512的 anchors 搭配。
        # 图片和 GT 同时缩放；不能只改图片而不改框。
        w, h = image.size
        original_size = (h, w)
        scale = min(self.min_size / min(h, w), self.max_size / max(h, w))
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        boxes[:, 0::2] *= new_w / w
        boxes[:, 1::2] *= new_h / h

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
            "image_size": (image.shape[1], image.shape[2]),  # (H,W)，不是 (W,H)。
            "original_size": original_size,  # test 画图时把框缩放回原图。
        }


if __name__ == "__main__":
    testdataset = FastRCNNVOCDataset(
        root="./data",
        image_set="test",
    )
    data = testdataset[0]
    print("boxes.shape:", data["boxes"].shape)
    print("labels.shape:", data["labels"].shape)
    print("image_size:", data["image_size"])
