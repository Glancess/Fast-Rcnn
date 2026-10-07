import torch
import torch.nn.functional as F
from Dataset.dataset import FastRCNNVOCDataset
from torch.utils.data import DataLoader


def detection_collate_fn(batch):
    images = [item["image"] for item in batch]
    image_sizes = [item["image_size"] for item in batch]
    boxes = [item["boxes"] for item in batch]
    labels = [item["labels"] for item in batch]

    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    # VGG 去掉最后一个 maxpool 后 stride=16，统一补到 16 的倍数。
    max_h = (max_h + 15) // 16 * 16
    max_w = (max_w + 15) // 16 * 16

    padded_images = []

    for img in images:
        _, h, w = img.shape

        pad_right = max_w - w
        pad_bottom = max_h - h

        img = F.pad(
            img, (0, pad_right, 0, pad_bottom)
        )  # just pad the image to the right and bottom with zeros   ,the original image is in the top left corner

        padded_images.append(img)

    images = torch.stack(padded_images, dim=0)

    return {"images": images, "image_sizes": image_sizes, "boxes": boxes, "labels": labels}


def get_dataloader(
    root, batch_size=2, shuffle=True, image_set="train"
):
    dataset = FastRCNNVOCDataset(
        root=root, image_set=image_set
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=detection_collate_fn
    )
    return dataloader
