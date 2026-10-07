import torch
import torch.nn as nn
from torchvision.ops import RoIPool
from torchvision.models import vgg16, VGG16_Weights
import torch.nn.functional as F
from utils.RPNproposalgeneration import generate_proposals, reshape_rpn_outputs
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList


class Fastrcnn(nn.Module):
    def __init__(self, weights=VGG16_Weights.DEFAULT):
        super().__init__()
        self.anchor_generator = AnchorGenerator(
            sizes=((128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),),
        )
        # 正常训练用 ImageNet 权重；只加载 best 做测试时传 weights=None，避免重复下载。
        vgg = vgg16(weights=weights)
        self.backbone = vgg.features[:-1]
        self.rpn_conv = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.rpn_cls = nn.Conv2d(512, 2 * 9, kernel_size=1)  # 18
        self.rpn_bbox = nn.Conv2d(512, 4 * 9, kernel_size=1)  # 36
        # 9个anchor
        self.classifier = vgg.classifier[:-1]  # 去掉原来的 1000 类 fc8
        self.pooled = RoIPool(
            output_size=(7, 7),
            spatial_scale=1.0 / 16.0,
        )
        self.flattened = nn.Flatten()
        self.cls_head = nn.Linear(4096, 21)  # VOC: 20类 + background
        self.bbox_head = nn.Linear(4096, 20 * 4)  # 20类，每类4个偏移

        # 新加的层从较小的数开始；VGG 的预训练参数不动。
        for layer in [self.rpn_conv, self.rpn_cls, self.rpn_bbox, self.cls_head]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.bbox_head.weight, std=0.001)
        nn.init.zeros_(self.bbox_head.bias)

    def forward_rpn(self, images, image_sizes, proposal_training=None):
        """图片 + 每张图的 (H,W) -> 共享特征图、RPN 输出和候选框。"""
        # AnchorGenerator 从图片/特征图大小推 stride，补到 16 的倍数才能固定为 16。
        # 只在右边和下面补零；image_sizes 仍然是补零前的真实尺寸。
        pad_h = (16 - images.shape[2] % 16) % 16
        pad_w = (16 - images.shape[3] % 16) % 16
        images = F.pad(images, (0, pad_w, 0, pad_h))
        feature = self.backbone(images)

        # RPN
        x = F.relu(self.rpn_conv(feature))

        rpn_cls_logits = self.rpn_cls(x)
        rpn_bbox_deltas = self.rpn_bbox(x)

        image_list = ImageList(
            images,
            image_sizes,
        )
        # image_sizes 是 resize 后、补零前的尺寸；不是磁盘原图的尺寸。

        anchors_list = self.anchor_generator(
            image_list,
            [x],
        )
        # 目前只用 torchvision 的这一套 anchors，不混用手写 generate_anchors。
        # 两套模板顺序/中心定义不同；同一套顺序必须从输出到 loss 保持一致。

        if proposal_training is None:
            proposal_training = self.training

        # RPN output -> proposals
        proposals_list = generate_proposals(
            anchors_list,
            rpn_cls_logits,
            rpn_bbox_deltas,
            image_sizes,
            pre_nms_topk=12000 if proposal_training else 6000,
            post_nms_topk=2000 if proposal_training else 300,
        )
        cls, deltas = reshape_rpn_outputs(rpn_cls_logits, rpn_bbox_deltas)
        return feature, {
            "rpn_cls_logits": cls,  # [B,A,2]，这里没有 detach，供 RPN loss 用。
            "rpn_bbox_deltas": deltas,  # [B,A,4]
            "anchors": anchors_list,  # 长度 B，每项 [A,4]
            "proposals": proposals_list,  # 长度 B，每项 [N,4]，不带计算图。
        }

    def forward_head(self, feature, proposals_list):
        """共享特征图 + 框列表 -> 每个 RoI 的类别和四个修框偏移。"""
        pooled = self.pooled(feature, proposals_list)  # 注意是 shared feature
        flattened = self.flattened(pooled)

        classfier_output = self.classifier(flattened)
        cls_logits = self.cls_head(classfier_output)
        bbox_logits = self.bbox_head(classfier_output)
        return cls_logits, bbox_logits

    def forward(self, images, image_sizes):
        # 完整预测入口；训练要先采样 RoI，再调用 forward_head，不能全送进 FC。
        feature, outputs = self.forward_rpn(images, image_sizes)
        cls_logits, bbox_logits = self.forward_head(feature, outputs["proposals"])
        outputs["cls_logits"] = cls_logits
        outputs["bbox_logits"] = bbox_logits
        return outputs


if __name__ == "__main__":
    model = Fastrcnn(weights=None).eval()  # 接口小检查，不下载预训练权重。
    images = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        outputs = model(images, [(224, 224)])
    print("RPN cls:", outputs["rpn_cls_logits"].shape)
    print("RPN bbox:", outputs["rpn_bbox_deltas"].shape)
    print("RoI cls:", outputs["cls_logits"].shape)
    print("RoI bbox:", outputs["bbox_logits"].shape)
