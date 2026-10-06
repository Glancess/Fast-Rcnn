from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.ops import nms
from PIL import Image, ImageDraw

from Dataset.dataset import VOC_CLASSES
from utils.decode import decode_boxes


@torch.no_grad()
def predict_one_image(
    model, image, proposals, device, score_thresh=0.5, nms_thresh=0.3
):
    """image: [3,H,W]；proposals: [N,4]，xyxy。返回最终框、类别和分数。"""
    model.eval()
    _, h, w = image.shape

    if len(proposals) == 0:
        return {
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty((0,), dtype=torch.int64),
            "scores": torch.empty((0,)),
        }

    # 一次预测一张图，仍然使用全部 proposals，不做 GT 匹配或 64 个框的采样。
    images = image.unsqueeze(0).to(device)  # [1,3,H,W]
    proposals = proposals.to(device)
    cls_logits, bbox_logits = model(images, [proposals])
    scores = F.softmax(cls_logits, dim=1)  # [N,21]，包含背景。
    bbox_deltas = bbox_logits.reshape(-1, 20, 4)  # [N,20,4]

    all_boxes = []
    all_labels = []
    all_scores = []

    # 测试时不知道真实类别，所以分别处理 20 个前景类别；0 是背景，不输出。
    for class_id in range(1, 21):
        class_scores = scores[:, class_id]  # [N]
        class_deltas = bbox_deltas[:, class_id - 1, :]  # [N,4]
        boxes = decode_boxes(proposals, class_deltas)  # 修框后仍是 xyxy。

        # 不保留 NaN / inf 框；再把坐标限制到原图范围内。
        valid = torch.isfinite(boxes).all(dim=1)
        boxes[:, 0].clamp_(0, w)
        boxes[:, 2].clamp_(0, w)
        boxes[:, 1].clamp_(0, h)
        boxes[:, 3].clamp_(0, h)

        keep = (
            (class_scores >= score_thresh)
            & valid
            & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )
        boxes = boxes[keep]
        class_scores = class_scores[keep]
        if len(boxes) == 0:
            continue

        # 同类别内部去掉重复框。NMS 比较预测框与预测框，不需要 GT。
        keep = nms(boxes, class_scores, iou_threshold=nms_thresh)
        boxes = boxes[keep]
        class_scores = class_scores[keep]
        labels = torch.full(
            (len(boxes),), class_id, dtype=torch.int64, device=device
        )
        all_boxes.append(boxes)
        all_labels.append(labels)
        all_scores.append(class_scores)

    if len(all_boxes) == 0:
        return {
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty((0,), dtype=torch.int64),
            "scores": torch.empty((0,)),
        }

    # 返回 CPU Tensor，方便打印和画图：boxes [K,4]，labels/scores [K]。
    return {
        "boxes": torch.cat(all_boxes, dim=0).cpu(),
        "labels": torch.cat(all_labels, dim=0).cpu(),
        "scores": torch.cat(all_scores, dim=0).cpu(),
    }


def draw_comparison(image, gt_boxes, gt_labels, prediction, save_path):
    """image 是原始 PIL 图片；左边画绿色 GT，右边画红色预测。"""
    gt_image = image.copy()
    pred_image = image.copy()
    gt_draw = ImageDraw.Draw(gt_image)
    pred_draw = ImageDraw.Draw(pred_image)

    for box, label in zip(gt_boxes.tolist(), gt_labels.tolist()):
        class_name = VOC_CLASSES[label - 1]
        gt_draw.rectangle(box, outline="lime", width=2)
        gt_draw.text(
            (box[0], max(0, box[1] - 12)), class_name,
            fill="lime", stroke_width=1, stroke_fill="black",
        )

    for box, label, score in zip(
        prediction["boxes"].tolist(),
        prediction["labels"].tolist(),
        prediction["scores"].tolist(),
    ):
        class_name = VOC_CLASSES[label - 1]
        pred_draw.rectangle(box, outline="red", width=2)
        pred_draw.text(
            (box[0], max(0, box[1] - 12)),
            f"{class_name} {score:.2f}",
            fill="red", stroke_width=1, stroke_fill="white",
        )

    # 用原始图片画框，不直接画已经 normalize 的 Tensor。
    w, h = image.size
    comparison = Image.new("RGB", (w * 2, h + 30), color="white")
    comparison.paste(gt_image, (0, 30))
    comparison.paste(pred_image, (w, 30))
    draw = ImageDraw.Draw(comparison)
    draw.text((5, 8), "Ground Truth", fill="green")
    draw.text((w + 5, 8), "Prediction", fill="red")
    comparison.save(save_path)


def test(
    model, dataset, device, save_dir, num_images=10, score_thresh=0.5, nms_thresh=0.3
):
    """只预测并画图，不计算 loss 或 mAP；默认只看 test 前 10 张。"""
    if num_images < 1:
        raise ValueError("num_images 必须至少是 1。")
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    num_images = min(num_images, len(dataset))
    saved_images = 0

    for index in range(num_images):
        image_id = Path(dataset.voc.images[index]).stem
        try:
            sample = dataset[index]
        except FileNotFoundError as error:
            print(f"{image_id} 缺少图片、标注或 proposal，跳过：{error}")
            continue

        # GT 不传给预测函数，只在预测结束后用于左侧画框对照。
        prediction = predict_one_image(
            model, sample["image"], sample["proposals"], device,
            score_thresh=score_thresh, nms_thresh=nms_thresh,
        )
        raw_image, _ = dataset.voc[index]
        save_path = save_dir / f"{image_id}.jpg"
        draw_comparison(
            raw_image, sample["boxes"], sample["labels"], prediction, save_path
        )
        saved_images += 1
        print(f"[{index + 1}/{num_images}] {image_id}: {len(prediction['boxes'])} detections")
        for label, score, box in zip(
            prediction["labels"].tolist(),
            prediction["scores"].tolist(),
            prediction["boxes"].tolist(),
        ):
            print(VOC_CLASSES[label - 1], f"score={score:.3f}", "box=", box)
        print("Save image:", save_path)

    print(f"Test 可视化完成：保存 {saved_images}/{num_images} 张，不代表 mAP 成绩。")
    return saved_images
