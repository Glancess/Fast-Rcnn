# Fast R-CNN 学习复现：VGG16 + VOC2007

这是一个以理解代码为目的的 PyTorch 目标检测项目，保留了直观的 Dataset、DataLoader、Model、train、eval、test 分工。

目前已实现离线生成 Selective Search proposals、训练、验证 loss、保存 best 权重，以及加载 best 后修框、NMS 和 GT/预测对比图。这里的“完成”指这条学习流程已经接通，**不代表完整复现了论文的训练配置或检测成绩**。

当前默认 `only_test=True`，直接使用已有的 `checkpoints/best.pth`，不会重新训练。没有实现 mAP，没有在 README 中填入未经记录的成绩。

## 1. 已有 best 权重，怎样直接看结果

在服务器项目目录运行：

```bash
cd /root/Fast-Rcnn
python main.py
```

运行前检查 `main.py` 中这些设置：

```python
data_root = Path("/root/Fast-Rcnn/data")
proposal_dirTest = data_root / "test/VOCdevkit/VOC2007/SelectiveSearchProposals"
save_dir = Path("./checkpoints")

only_test = True
num_test_images = 10
score_thresh = 0.5
nms_thresh = 0.3
```

需要已有 best 权重、test 图片、XML 标注、`test.txt` 和对应的 proposal `.pt` 文件。

程序逐张使用全部 proposals 预测，默认只处理 test 列表中的前 10 张。结果保存到运行目录下的 `test_results/<image_id>.jpg`：左侧绿色框是真实标注，右侧红色框是预测，并显示类别和置信度。

缺少 best、test 列表或 proposal 目录时，会提示并返回；某张图片缺少图片、标注或 `.pt` 时，会提示并跳过。即使没有高分预测，也会保存左侧有 GT、右侧没有框的对比图。

**默认画 10 张图是可视化检查，不是完整 test 评测。** 当前入口需要 XML 来画 GT；真正的预测函数 `predict_one_image` 本身不需要 GT。

## 2. 项目结构

```text
Fast-Rcnn/
├── main.py                       # 选择模式、调度训练/验证、保存和加载 best
├── Model.py                      # VGG16 + RoIPool + 分类头/回归头
├── Dataset/
│   ├── dataset.py                # 图片、GT、标签、proposals 的读取与归一化
│   ├── dataloader.py             # collate_fn：补齐图片、保留变长框列表
│   └── testdataset.py            # VOC Dataset 小实验，不是完整检测测试入口
├── engine/
│   ├── train_one_epoch.py        # 一轮训练，含 backward 和 SGD 更新
│   ├── eval.py                   # 一轮验证，只计算 loss
│   └── test.py                   # 全 proposal 预测、修框、NMS、画对比图
├── utils/
│   ├── generate_selective_search.py  # 一张图片生成候选框
│   ├── generate.py                  # 多进程生成并缓存 proposals
│   ├── assignAndsampling.py         # 匹配 GT、分配标签、采样 RoI
│   ├── encode.py                    # proposal + GT → 正确偏移
│   ├── decode.py                    # proposal + 预测偏移 → 预测框
│   └── loss.py                      # 分类 loss + 前景框回归 loss
├── .gitignore
└── README.md
```

运行后可能出现 `checkpoints/` 和 `test_results/`。数据、权重、图片结果不是项目源码，不应提交进 Git。

## 3. 当前数据目录：train/val 与 test 分开放

下面是当前 `main.py` 预期的布局，不是要求所有 VOC 项目必须这样放：

```text
/root/Fast-Rcnn/data/
├── VOCdevkit/
│   └── VOC2007/                         # train/val 数据
│       ├── JPEGImages/
│       ├── Annotations/
│       ├── ImageSets/Main/
│       │   ├── train.txt
│       │   ├── val.txt
│       │   └── trainval.txt
│       └── SelectiveSearchProposals/
│           └── <image_id>.pt
└── test/
    └── VOCdevkit/
        └── VOC2007/                     # 独立 test 数据
            ├── JPEGImages/
            ├── Annotations/
            ├── ImageSets/Main/test.txt
            └── SelectiveSearchProposals/
                └── <image_id>.pt
```

`ImageSets/Main/train.txt` 等划分文件中的每一行是图片编号，例如 `000012`，不是图片内容。读取编号后，再去找对应的 JPG、XML 和 `.pt`。

当前训练只用 `train`，验证只用 `val`。`trainval` 表示 train 与 val 的合并集；给 trainval **生成候选框**是离线数据准备，不等于把 val **用于更新模型参数**。

test 在单独的 `data/test` 目录，所以调用 `test_best_model` 时传入 `data_root / "test"`。Dataset 的 root 是包含 `VOCdevkit` 的目录，而生成器的 `voc_root` 是直接包含 `JPEGImages` 的 `VOC2007` 目录，二者层级不同。

不要把 val 改名后当作独立 test，也不能拿曾经在 trainval 上训练的权重，在 val 上报告独立验证成绩。

## 4. 环境与依赖

主要依赖：Python、PyTorch、torchvision、Pillow，以及带 contrib 模块的 OpenCV。

PyTorch 和 torchvision 应使用相互兼容、适配服务器 CUDA 的版本，安装方式参考 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)。本项目使用 torchvision 的 RoIPool、box_iou 和 NMS，需要这些算子的底层实现能够正常加载。

Selective Search 使用 `cv2.ximgproc.segmentation`。在不需要 OpenCV GUI 的服务器环境中，可以选择：

```bash
python -m pip install pillow opencv-contrib-python-headless
```

如果现有环境已经能运行，不必为了收尾升级。OpenCV 的几个 pip 包共享 `cv2` 命名空间，应只选择一个适合环境的包，避免同时安装多个版本。[OpenCV 包说明](https://pypi.org/project/opencv-contrib-python/)

可以先做小检查，不会下载数据或权重：

```bash
python - <<'PY'
import torch
import torchvision
import cv2
from torchvision.ops import roi_pool, nms

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA:", torch.cuda.is_available())
print("Selective Search:", hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "segmentation"))

features = torch.randn(1, 512, 4, 4)
boxes = [torch.tensor([[0., 0., 32., 32.]])]
print("RoIPool:", roi_pool(features, boxes, (7, 7), spatial_scale=1/16).shape)
print("NMS:", nms(boxes[0], torch.tensor([0.9]), 0.3))
PY
```

这里 RoIPool 应输出 `[1,512,7,7]`。这是 CPU 算子检查，不能代替服务器 CUDA 测试。

本地轻量检查环境为 Python 3.12.13、PyTorch 2.12.1、torchvision 0.27.1、OpenCV 5.0.0、Pillow 12.3.0。这只是已检查的本地环境记录，不是服务器版本声明或强制安装要求；目前没有固定依赖文件。

## 5. 提前生成 proposals

当前 `utils/generate.py` 最下面已经配置为生成 test：

```python
generate_voc_proposals(
    voc_root="/root/Fast-Rcnn/data/test/VOCdevkit/VOC2007",
    split="test",
    max_proposals=2000,
    num_workers=4,
)
```

在项目根目录运行：

```bash
python utils/generate.py
```

如果要准备 train/val，把最下面的调用改为：

```python
generate_voc_proposals(
    voc_root="/root/Fast-Rcnn/data/VOCdevkit/VOC2007",
    split="trainval",
    max_proposals=2000,
    num_workers=4,
)
```

生成过程不使用 GT，只处理图片。OpenCV 返回 `[x,y,w,h]`，代码转换成 `[x1,y1,x2,y2]`，保存为 float32 Tensor，每张最多 2000 个框。

几个容易误解的地方：

- 这里走 CPU；GPU 不参与这个生成过程。
- `ss.process()` 先完成搜索，之后才截取最多 2000 个结果，所以减少保留数量不等于搜索提前结束。
- 生成器的 `num_workers` 是同时处理图片的进程数；训练 DataLoader 的 `num_workers` 是读数据的进程数，二者互不控制。
- 每个生成进程内部限制线程数，避免过度争抢 CPU。8 个可用 CPU 可以先用 4 个进程，再实测比较 8 个，不能假定速度线性增长。
- 已有 `.pt` 会跳过；新文件先写 `.tmp`，保存完整后再改名为 `.pt`。
- 不要同时启动两个生成程序写同一个目录。已有 `.pt` 只检查是否存在，不会自动检查损坏或生成参数是否变化。

## 6. main 的配置和两种模式

目前没有单独的 config 文件，参数直接写在 `main.py` 和对应函数里。

| 配置 | 当前值 | 作用 |
|---|---|---|
| `only_test` | `True` | 直接加载 best，只预测画图 |
| `num_test_images` | `10` | 最多查看 test 列表前 10 张 |
| `score_thresh` | `0.5` | 过滤类别置信度低的框 |
| `nms_thresh` | `0.3` | 同类别预测框之间的 NMS IoU 阈值 |
| 训练 batch_size | `2` | 每批两张原图，不是两张 RoI |
| train/val DataLoader workers | `4` | 并行读取图片、标注和缓存 proposals |
| `num_epochs` | `10` | 开启训练模式时的训练轮数 |
| SGD lr / momentum / weight_decay | `1e-3 / 0.9 / 5e-4` | 参数更新设置 |
| 每张采样 RoI 上限 | `64` | 写在 train/eval 调用中 |
| `fg_fraction` | `0.25` | 每张前景名额最多为 16 |

设备优先选择 CUDA，否则 CPU；当前不走 MPS。

### 只测试已有 best

保留 `only_test=True`。先检查文件，创建 `Fastrcnn(weights=None)`，加载 best 的全部模型权重，进行测试可视化；不创建 train/val DataLoader，不创建 SGD，不更新或覆盖 best，也不为了测试下载 VOC 或 ImageNet 权重。

### 开启一次新的训练

先准备 train/val proposals，再设置 `only_test=False`。

**这不是续训。** 程序使用 ImageNet 预训练的 VGG16 和新初始化的检测头开始训练，不会加载旧 best 或恢复旧 optimizer。若继续使用相同的 `save_dir`，本次验证 loss 创新低时会覆盖已有 `best.pth`。需要保留旧实验时，先备份权重，或为新实验设置不同的保存目录。

训练 Dataset 当前默认 `download=True`，缺少 VOC 数据时可能触发下载；`Fastrcnn()` 缺少缓存的 ImageNet 权重时也可能下载。test Dataset 则显式设置 `download=False`。

`checkpoints/` 和 `test_results/` 是相对运行目录的路径，建议始终从项目根目录运行。

## 7. Dataset → DataLoader：先把数据对应起来

一张图片的 Dataset 返回值：

| 字段 | 类型和 shape | 含义 |
|---|---|---|
| `image` | float32 Tensor `[3,H,W]` | RGB 图片 |
| `boxes` | float32 Tensor `[M,4]` | GT，xyxy |
| `labels` | int64 Tensor `[M]` | GT 类别，1～20 |
| `proposals` | float32 Tensor `[N,4]` | 原图坐标系的候选框 |

图片先用 `to_tensor` 转成 CHW 并缩放到 `[0,1]`，再按 RGB 通道归一化：

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
# 每个通道：(像素值 - mean) / std
```

这些值与 torchvision 的 ImageNet 预训练 VGG16 配套，不是本项目重新统计的 VOC 均值。目前 train/val/test 使用相同的图片处理，没有随机翻转、裁剪或 resize。

`detection_collate_fn` 把图片在右侧、下侧补零，得到 `[B,3,H_max,W_max]`；框和标签数量不同，所以保留为长度 B 的列表。没有缩放图片，因此不改变框坐标。

```python
batch["proposals"][b]  # 第 b 张图片的 proposals
batch["boxes"][b]      # 同一张图片的 GT
batch["labels"][b]     # 同一张图片的 GT 类别
```

全项目使用同一份 `VOC_CLASSES` 和 `CLASS_TO_IDX`：0 是背景，1～20 是物体类别。不能随意重排类别，否则训练标签、分类输出、回归头下标、画图名称会错位，旧权重也不能再按原来的含义解释。

## 8. 匹配 GT → 采样 → encode

对每张图片计算 `box_iou(proposals, gt_boxes)`，得到 `[N,M]`。`max(dim=1)` 为每个 proposal 找出最大 IoU 和匹配 GT 下标，并不会删除 proposal。

| 最大 IoU | proposal label | 是否采样 |
|---|---|---|
| `>= 0.5` | 匹配 GT 的类别 | 前景，参与 |
| `0.1 <= IoU < 0.5` | `0` | 背景，参与 |
| `< 0.1` | `-1` | 忽略，不参与 |

`sample_rois` 返回一维整数 Tensor `[S]`，里面是原始 proposal 的下标，不是框坐标。每张最多取 64 个，其中前景最多 16 个；前景不足时可以多取背景，背景不足时不重复补满，所以实际数量和比例不一定正好是 64、25%。

训练随机取，验证固定顺序取。使用同一组下标同时取出框、类别和匹配 GT，保证每一行对应：

```python
sampled_proposals = proposals[selected_idx]                         # [S,4]
sampled_labels = assignments["labels"][selected_idx]                # [S]
matched_gt_idx = assignments["max_indices"][selected_idx]           # [S]
sampled_matched_gt_boxes = gt_boxes[matched_gt_idx]                  # [S,4]
bbox_targets = encode_boxes(sampled_proposals, sampled_matched_gt_boxes)
```

多个 proposal 可以匹配同一个 GT。背景也保留“最像哪个 GT”的下标，并计算占位的回归目标；它的回归目标会在 loss 中被排除，背景只训练分类为 0。

encode 返回 `[dx,dy,dw,dh]`：

```text
dx = (GT中心x - proposal中心x) / proposal宽
dy = (GT中心y - proposal中心y) / proposal高
dw = log(GT宽 / proposal宽)
dh = log(GT高 / proposal高)
```

decode 则根据偏移还原中心、宽高，再转换成 xyxy。delta 是“怎样修改框”，不是 xyxy 或 xywh。

## 9. Model：两张原图与最多 128 个 RoI

定义 `B` 为图片数，`S_i` 为每张图采样数，`R=sum(S_i)` 为整批 RoI 数。

```text
images [B,3,H,W]
→ VGG16 features，去掉最后的 maxpool5
→ 共享特征图 [B,512,floor(H/16),floor(W/16)]
→ RoIPool [R,512,7,7]
→ Flatten [R,25088]
→ VGG16 FC [R,4096]
→ cls_head [R,21] + bbox_head [R,80]
```

传给模型的框仍是长度 B 的列表，列表第 i 项属于第 i 张图片。RoIPool 根据这个关系从对应特征图取区域，用 `spatial_scale=1/16` 映射原图坐标，不要在送入前再手动除一次 16。[RoIPool 接口说明](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.roi_pool.html)

卷积部分处理的是两张原图；FC 处理的是最多 128 个区域特征，不是重新把 128 张裁剪图送过 VGG 卷积。

分类头的 21 类包含背景；回归头的 80 个数是 20 个前景类别各自的四个偏移，没有背景回归分支。

## 10. Loss → backward → SGD

`FastRCNNLoss` 的输入为：

```text
class_logits [R,21]
bbox_preds   [R,80]
class_labels [R]      # 0～20，不包含已经忽略的 -1
bbox_targets [R,4]
```

分类用 CrossEntropyLoss，输入原始 logits，不要提前 softmax；前景和背景都参与。

回归输出先整理成 `[R,20,4]`，只取前景 RoI 的真实类别对应的四个偏移。例如 label=9 对应回归分支下标 8。对这些前景预测和目标计算 SmoothL1Loss，求和后除以总 RoI 数 R。没有前景时，回归 loss 为 0。

总 loss 为两项之和，函数返回三个标量 Tensor：`loss, cls_loss, bbox_loss`。

训练执行：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

`backward` 计算梯度，SGD 的 `step` 才真正更新网络参数；离线 Selective Search 不参与反向传播。

## 11. evaluate 为什么仍然使用 GT

每个 epoch 先执行 `train_one_epoch`，再把同一个、已经更新参数的模型交给 `evaluate`。

要计算验证 loss，仍然需要 GT 给 proposals 生成类别与偏移目标。但验证使用 `model.eval()`、`torch.no_grad()` 和固定采样，不调用 backward 或 optimizer.step，因此不会拿 val 更新参数。

train/eval 都返回：

```python
{"loss": float, "cls_loss": float, "bbox_loss": float}
```

这些值是整轮按实际 RoI 数加权的平均值。验证 loss 是采样区域上的误差，不是最终检测框的 mAP；训练有 Dropout 和随机采样，验证没有，因此两边 loss 也不必严格一一对应。

## 12. test：修框、过滤、NMS，然后画图

test 不匹配 GT、不抽 64 个框，每张使用其全部 proposals。模型输出 `[N,21]` 和 `[N,80]`，分别得到概率和 `[N,20,4]` 偏移。

对 1～20 每个前景类别分别：

```text
取该类别概率 [N]、该类别偏移 [N,4]
→ decode 得到 xyxy
→ 排除非有限框、限制坐标到原图
→ 排除退化框与低分框
→ 该类别内部 NMS
```

不同类别之间不会互相 NMS。NMS 用预测框之间的 IoU 删除重复候选，而不是与 GT 匹配；分类置信度也不是预测框与 GT 的 IoU。[NMS 接口说明](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.nms.html)

合并后返回 CPU Tensor：`boxes [K,4]`、`labels [K]`、`scores [K]`。GT 只在预测结束后用于左侧画图，不决定保留哪些预测。

正式检测评测还需把最终预测与 GT 匹配，统计误检、漏检并计算各类 AP/mAP。本项目暂不实现这一步，不能把置信度、验证 loss 或几张图片的观感当作正式检测成绩。

## 13. best.pth 保存了什么

当前 best 的标准是 **val 总 loss 更低**，不是 test 效果或 mAP 更高。

```python
{
    "epoch": int,                    # 从 0 开始
    "model_state_dict": ...,         # 模型权重
    "optimizer_state_dict": ...,     # SGD 状态
    "val_loss": float,
    "train_split": "train",
    "val_split": "val",
}
```

只测试时创建不下载 ImageNet 权重的模型，并用 `model.load_state_dict(checkpoint["model_state_dict"])` 加载已训练参数；不会恢复 optimizer。

目前只有 best，没有 `last.pth`，也没有恢复训练入口。保存 optimizer 状态不等于已经实现续训。以后若加 last，通常保存最新一轮用于恢复；best 保留按验证标准选中的轮次。

## 14. 函数接口速查

| 函数 | 输入 | 返回 / 主要作用 |
|---|---|---|
| `selective_search` | 图片路径 str、上限 int | 通常 `[N,4]` float32 候选框 |
| `generate_one_image` | `(Path, str, int)` 元组 | `(图片编号, 框数)`；已有文件返回框数 None |
| `generate_voc_proposals` | 路径、划分、上限、进程数 | None；把每张图的 proposals 写入磁盘 |
| `FastRCNNVOCDataset.__len__` | 无额外输入 | 图片数量 int |
| `FastRCNNVOCDataset.__getitem__` | 图片下标 int | 单张图片、GT、标签和 proposals 的 dict |
| `detection_collate_fn` | 长度 B 的样本 list | batch dict，图片 stack，框/标签保留 list |
| `get_dataloader` | 路径、batch_size、shuffle、划分 | DataLoader；main 当前直接创建，没有调用它 |
| `assign_gt_boxes` | `[N,4]`、`[M,4]`、`[M]` | labels、max_iou、max_indices，均 `[N]` |
| `sample_rois` | labels `[N]`、采样设置 | 下标 Tensor `[S]` |
| `encode_boxes` | 逐行对应的两个 `[S,4]` xyxy | 正确偏移 `[S,4]` |
| `decode_boxes` | proposal `[N,4]`、偏移 `[N,4]` | 修正后的 xyxy `[N,4]` |
| `Fastrcnn.forward` | 图片 `[B,3,H,W]`、长度 B 的框 list | logits `[R,21]`、偏移 `[R,80]` |
| `FastRCNNLoss.forward` | 两个模型输出、类别 `[R]`、目标 `[R,4]` | 总/分类/回归三个标量 Tensor |
| `train_one_epoch` | model、optimizer、loader、device | 三项平均 loss 的 float dict，并更新参数 |
| `evaluate` | model、loader、device | 三项平均 loss 的 float dict，不更新参数 |
| `predict_one_image` | model、图片 `[3,H,W]`、框 `[N,4]`、device、阈值 | boxes/labels/scores dict，全部 CPU Tensor |
| `draw_comparison` | PIL 图片、GT、预测 dict、保存路径 | None；写入对比图 |
| `test` | model、Dataset、device、输出目录、张数和阈值 | 实际保存图片数量 int |
| `test_best_model` | model、test 根目录、proposal/权重目录、device 等 | 保存图片数量；缺少必要文件时返回 None |
| `main` | 无 | None；总调度 |

## 15. 当前边界与 main 收尾检查

当前 main 的只测试分支与 `data/test` 路径已对齐；训练后测试也使用独立 test proposal 目录。没有发现会阻断这条既定流程的接口错误，但运行仍取决于服务器数据、proposal 和权重文件实际齐全。

本地轻量检查覆盖语法/import、loss 与采样接口、train/eval 参数更新区别、best 加载与只测试控制流、修框/NMS、临时图片保存，以及随机初始化 VGG16 的小输入前向。没有启动完整训练、批量生成真实数据的 proposals，或读取服务器训练权重进行正式 test 评测。真实训练曲线、具体 best loss 和检测成绩需由实际实验记录补充。

主要限制：

- 没有 mAP/AP、完整 test 报告或标准 VOC `difficult` 处理；当前 Dataset 把读取到的物体都作为 GT。
- 输入没有按固定短边 resize，没有数据增强；proposals 依赖外部 Selective Search，不是 Faster R-CNN 的 RPN。
- 采样不足时不补满，也没有把 GT 额外加入候选框保证前景；前景不足时回归学习信号可能少。
- 当前画图入口依赖 VOC XML，不是接受任意无标注图片的完整应用接口。
- 没有固定随机种子、依赖锁定、持久化 loss 曲线、自动续训或多实验管理。
- 图片过大或 proposals 多时，内存/显存使用会增加；逐张 test 不代表任意输入都不会 OOM。
- 不包含 AMP、DDP、scheduler、WandB 等扩展，也没有统一独立 config 层。

main 以后可以做的小完善，但本次不修改训练结构：

1. 用一个 `test_root = data_root / "test"` 统一测试路径，减少重复拼接。
2. 对当前要看的图片预检查 JPG/XML/pt；若一张也没保存，给出更醒目的失败提示。
3. 每次新训练使用不同保存目录，避免覆盖旧 best；如确有需求再加 last/续训。
4. 把现有简单超参数集中到 main 顶部，记录到 checkpoint，方便核对实验。

这些是易用性与实验管理建议，不是要求现在加入复杂功能。当前可以作为“训练、验证 loss、测试可视化闭环”的学习项目收尾；若要宣称标准论文成绩复现，仍需补齐评测协议和真实实验记录。

`.gitignore` 当前覆盖 `data/`、`*.pt`、`*.pth`、`__pycache__/`、`*.pyc`、`.venv/`、`.DS_Store` 和 `test_results/`。发布前检查 git status，不要提交数据、权重、环境凭据或运行缓存；其他位置的环境/缓存文件不一定已被这些规则覆盖。

## 16. 参考

- 本项目学习的检测方法：[Fast R-CNN，Ross Girshick](https://arxiv.org/abs/1504.08083)。本项目未提供与论文结果等价的评测结论。
- [VOC2007 数据与任务说明](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/index.html)。
- [torchvision RoIPool](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.roi_pool.html) 与 [NMS](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.nms.html)。
