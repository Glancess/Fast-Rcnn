import cv2
import torch


def selective_search(image_path, max_proposals=2000):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()

    rects = ss.process()  # like [[x, y, w, h], ...]

    proposals = []

    for x, y, w, h in rects:
        if w <= 1 or h <= 1:
            continue

        proposals.append([float(x), float(y), float(x + w), float(y + h)])

        if len(proposals) >= max_proposals:
            break

    return torch.tensor(proposals, dtype=torch.float32)


if __name__ == "__main__":
    image_path = "/Users/alen/Code/Pytorcc/data/VOCdevkit/VOC2007/JPEGImages/000005.jpg"
    proposals = selective_search(image_path)
    print("proposals.shape:", proposals.shape)
