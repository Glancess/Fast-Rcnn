import torch
import torch.nn.functional as F
from Dataset.dataset import FastRCNNVOCDataset
from torch.utils.data import DataLoader


def detection_collate_fn(batch):
    images = [item["image"] for item in batch]
    proposals = [item["proposals"] for item in batch]
    boxes = [item["boxes"] for item in batch]
    labels = [item["labels"] for item in batch]

    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

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

    return {"images": images, "proposals": proposals, "boxes": boxes, "labels": labels}


def get_dataloader(
    root, proposal_dir, batch_size=2, shuffle=True, image_set="train"
):
    dataset = FastRCNNVOCDataset(
        root=root, proposal_dir=proposal_dir, image_set=image_set
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=detection_collate_fn
    )
    return dataloader
