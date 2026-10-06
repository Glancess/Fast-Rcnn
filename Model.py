import torch
import torch.nn as nn
from torchvision.ops import RoIPool
from torchvision.models import vgg16, VGG16_Weights


class Fastrcnn(nn.Module):
    def __init__(self, weights=VGG16_Weights.DEFAULT):
        super().__init__()
        # 正常训练用 ImageNet 权重；只加载 best 做测试时传 weights=None，避免重复下载。
        vgg = vgg16(weights=weights)
        self.backbone = vgg.features[:-1]  # type: ignore # 去掉最后一个 maxpool5
        self.classifier = vgg.classifier[:-1]  # 去掉原来的 1000 类 fc8
        self.pooled = RoIPool(
            output_size=(7, 7),
            spatial_scale=1.0 / 16.0,
        )
        self.flattened = nn.Flatten()
        self.cls_head = nn.Linear(4096, 21)  # VOC: 20类 + background
        self.bbox_head = nn.Linear(4096, 20 * 4)  # 20类，每类4个偏移

    def forward(self, images, boxes):
        features = self.backbone(images)
        # features: [B,512,H/16,W/16]，每张图片只算一次共享特征图。
        pooled = self.pooled(features, boxes)
        # pooled: [R,512,7,7]，R 是这个 batch 采样的 RoI 总数。
        flattened = self.flattened(pooled)
        classfier_output = self.classifier(flattened)
        cls_logits = self.cls_head(classfier_output)
        bbox_logits = self.bbox_head(classfier_output)
        # cls_logits: [R,21]；bbox_logits: [R,80]，不是 [B,21]。
        return cls_logits, bbox_logits


if __name__ == "__main__":
    model = Fastrcnn()
    images = torch.rand(2, 3, 224, 224)
    boxes = [
        torch.tensor([[0.0, 0.0, 100.0, 100.0], [50.0, 50.0, 150.0, 150.0]]),
        torch.tensor([[10.0, 10.0, 200.0, 200.0]]),
    ]  # 每个框是 xyxy；列表第 0/1 项分别属于第 0/1 张图片。
    cls_logits, bbox_logits = model(images, boxes)
