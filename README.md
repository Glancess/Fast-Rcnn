# Faster R-CNN 学习复现：VGG16 + RPN + VOC2007

从自己的 Fast R-CNN 代码继续实现 Faster R-CNN，保留 Dataset、Model、engine、utils、main 的分工和普通循环、Tensor 索引、SGD 的写法。

当前选择原论文的四步交替训练，不是四项 loss 同时相加的近似联合训练。已经接入 RPN 目标分配、loss、proposal 生成、分阶段训练、验证 loss、checkpoint 和预测画图。这里只说明代码流程，不宣称达到论文成绩。

不再使用 Selective Search，不需要提前生成 proposal .pt。旧 Fast R-CNN best 不能直接当作这个项目的完整 Faster R-CNN 权重。

## 1. 总流程

```text
JPG + XML + train/val/test.txt
→ Dataset：RGB图片、GT框、类别，图片和GT一起resize
→ DataLoader：补齐图片，保留每张图真实尺寸和变长GT
→ VGG16：共享特征图
→ RPN：每个anchor的objectness logits + 四个偏移
  ├─ RPN训练：anchor匹配GT → 采样 → CE + SmoothL1 → backward → SGD
  └─ 候选框：anchor + 预测偏移 → decode → clip → 排序 → NMS
→ 检测器：proposal → RoIPool → FC → 类别logits + 各类偏移
  ├─ 检测器训练：proposal匹配GT → 采样 → CE + SmoothL1 → backward → SGD
  └─ 预测：按类别修框 → 分数过滤 → 同类别NMS → 画框
→ Validation：只计算当前阶段loss
→ 当前阶段best checkpoint → main调度下一阶段
```

有两次框回归：RPN学习 anchor → GT；检测器学习 proposal → GT。都用 `[dx,dy,dw,dh]`，但参照框不同。

## 2. 四步交替，交替什么

不是每个epoch先更新RPN再更新检测器，而是先完成一个阶段，再进入下一个阶段。

| 阶段 | 初始化/输入 | 更新哪些层 | loss |
|---|---|---|---|
| 1 | ImageNet VGG + 新RPN层 | backbone、RPN三层 | RPN分类 + RPN回归 |
| 2 | 另一个ImageNet VGG；固定阶段1 RPN产生框 | 检测器backbone、FC、检测头 | 检测分类 + 检测回归 |
| 3 | 阶段2 backbone + 阶段1 RPN专用层 | 只更新RPN三层，backbone冻结 | RPN分类 + RPN回归 |
| 4 | 阶段3的共享backbone/RPN + 阶段2检测上层 | 只更新FC、检测分类头和回归头 | 检测分类 + 检测回归 |

第2阶段两份模型还不共享卷积。第3阶段不是让整个VGG再次适应RPN，而是固定检测器学到的卷积，让RPN专用层适应它。第4阶段也固定共享卷积。

固定RPN在 no_grad 下在线产生框，不再引入离线缓存生成器。固定模型、输入和后处理参数下，候选框不随检测器训练而变化。

顺序对应[原论文四步交替训练](https://papers.nips.cc/paper_files/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf)。当前轮数和按val loss选best是学习版配置，不是完整论文实验协议。

## 3. 目录结构

```text
Faster-Rcnn/
├── main.py                         # 阶段交接、冻结/解冻、SGD、best、only_test
├── Model.py                        # VGG16、RPN、RoIPool、FC、检测头
├── Dataset/
│   ├── dataset.py                  # VOC图片/GT、resize和normalize
│   ├── dataloader.py               # 补零、stack、保留image_sizes
│   └── testdataset.py              # 单独运行的Dataset小实验
├── engine/
│   ├── train_one_epoch.py          # 按stage训练RPN或检测器
│   ├── eval.py                     # 当前stage的验证loss
│   └── test.py                     # 在线RPN、检测修框、NMS、画图
├── utils/
│   ├── RPNproposalgeneration.py    # 整理RPN输出、decode/clip/排序/NMS
│   ├── generate_anchors.py         # 保留的手写anchor练习，Model没有调用
│   ├── assignAndsampling.py        # 两套GT分配规则 + 采样
│   ├── encode.py                   # 参照框 + GT → 正确偏移
│   ├── decode.py                   # 参照框 + 预测偏移 → xyxy
│   └── loss.py                     # RPNLoss和FastRCNNLoss
├── .gitignore
└── README.md
```

类名 Fastrcnn 和 FastRCNNVOCDataset 暂时保留，但现在走Faster R-CNN流程，没有为了改名重排目录。

## 4. 数据和运行

data_root指向包含VOCdevkit的目录，也可以复用已有VOC数据目录，不用复制数据。

```text
data_root/
├── VOCdevkit/VOC2007/
│   ├── JPEGImages/
│   ├── Annotations/
│   └── ImageSets/Main/{train,val,trainval}.txt
└── test/VOCdevkit/VOC2007/
    ├── JPEGImages/
    ├── Annotations/
    └── ImageSets/Main/test.txt
```

txt是图片编号列表。只用train更新参数、val选择各阶段best；test不参与训练或选择权重。没有test时跳过画图，不用val冒充test。

依赖PyTorch、兼容的torchvision、Pillow。当前RPN流程不再需要OpenCV Selective Search。RoIPool/NMS需要torchvision的底层算子可正常加载。优先CUDA，否则CPU，不走MPS。

从项目目录运行 `python main.py`。先检查main中的设置：

```python
data_root = Path("/root/Faster-Rcnn/data")  # 按服务器实际位置改
save_dir = Path("./checkpoints")
only_test = False                         # 新训练；True直接测试阶段4 best
stage_epochs = [10, 10, 5, 5]             # 四阶段各自轮数
```

其余参数仍在对应文件中：batch_size=2、workers=4、SGD lr=1e-3、momentum=0.9、weight_decay=5e-4，没有另加配置框架。显存不足时可以先把两个DataLoader的batch_size改为1。

训练可能下载缺失的VOC/ImageNet权重。only_test=True使用weights=None后加载自己的best，不创建训练Dataset和optimizer，不重新训练。

重新运行训练不是续训，四阶段从头开始，同一保存目录可能覆盖已有阶段best。要保留旧实验，先换save_dir。

## 5. Dataset和DataLoader

Dataset每张返回：

```text
image         float32 [3,H,W]  # resize + normalize后的RGB图片
boxes         float32 [M,4]    # resize后GT，xyxy
labels        int64   [M]      # VOC类别1～20
image_size    tuple (H,W)     # resize后、补零前
original_size tuple (原H,原W) # 画图还原坐标
```

短边目标600，长边最多1000；长边限制生效时，短边可能不足600。图片和GT按实际新宽/高一起缩放，不能只改图片。

to_tensor把RGB像素缩放到[0,1]，再用ImageNet mean=[0.485,0.456,0.406]、std=[0.229,0.224,0.225]归一化。train/val/test使用相同的resize/normalize，目前没有随机增强。

统一CLASS_TO_IDX：0是检测背景，1～20是VOC物体类别。RPN则只有0/1，不预测chair/person等具体类别。

collate在右边、下边补零到整批最大尺寸，并补到16的倍数，得到[B,3,H_pad,W_pad]。image_sizes保留每张图的真实尺寸；GT/labels保留长度B的列表。补零不是resize，不改变GT坐标。

这样AnchorGenerator的stride固定为16，框过滤不误用补零尺寸，RoIPool也能按列表顺序找到对应图片。

## 6. Model接口和anchor顺序

特征图[B,512,Hf,Wf]，A=Hf*Wf*9：

```text
RPN 3×3卷积 + ReLU
├─ 分类1×1卷积 [B,18,Hf,Wf] → [B,A,2]
└─ 回归1×1卷积 [B,36,Hf,Wf] → [B,A,4]
```

顺序统一为特征图行 → 列 → 当前anchor编号。每个anchor的两个分类通道依次背景/物体，四个回归通道依次dx/dy/dw/dh。reshape后的行必须对应同一行anchor。

目前用torchvision AnchorGenerator。手写generate_anchors.py保留供学习，但其模板排序、中心定义和ratio含义不同，不要混用两套。

- forward_rpn(images, image_sizes)：返回(feature, outputs)，outputs有rpn_cls_logits[B,A,2]、rpn_bbox_deltas[B,A,4]、anchors列表和proposals列表。
- forward_head(feature, proposals_list)：返回cls_logits[R,21]、bbox_logits[R,80]，R是整批RoI数。
- model(images, image_sizes)：完整预测，返回上述dict并加入检测器两个输出。训练先采样再调用head，不能把每张2000框全部送入FC。

RoIPool为[R,512,7,7]，Flatten为[R,25088]，FC为[R,4096]。spatial_scale=1/16已经映射图片坐标，不要事先再把框除以16。

## 7. 两套GT分配与loss

RPN的assign_gt_anchors：跨界anchor忽略；有效anchor中IoU>=0.7为物体1，IoU<0.3为背景0，中间为-1。每个GT的最佳匹配也设为正样本，正样本规则优先。每张最多采256个，正样本最多128个，不足按实际数量取。

RPNLoss直接接收保留计算图的原始logits/deltas。分类CE；回归只比较正anchor的预测偏移和encode(anchor, matched_gt)。背景/ignore不做回归。当前分类按实际采样数平均，回归按特征图位置数归一化并乘lambda=10；不是所有论文训练细节的逐项复刻。

检测器保留原规则：IoU>=0.5为具体类别，0.1<=IoU<0.5为背景，<0.1忽略。每张最多64个RoI，前景最多16个。回归只取前景真实类别的四个预测值，目标为encode(proposal, matched_gt)，其余背景目标置零且不参与回归。

FastRCNNLoss为分类CE + 前景SmoothL1，回归求和后除以总RoI数。train/eval返回loss、cls_loss、bbox_loss三个float。阶段1/3指RPN，阶段2/4指检测器，不能跨阶段直接比较大小。RPN整轮按图片平均，检测器按实际RoI数平均。

## 8. NMS不求导，RPN怎么学习

```text
原始objectness logits → CE(预测, anchor标签) ─────────────┐
原始bbox deltas → SmoothL1(预测, anchor到GT的偏移) ───────┤→ backward
                                                       └→ RPN参数的梯度

同一输出的无梯度版本 → decode → clip → 排序 → NMS → proposals → 检测器
```

loss不是拿NMS剩下的框才算。GT匹配、标签和目标偏移只是构造监督答案，直接比较网络输出与答案，就可以反向传播。

generate_proposals在no_grad下运行，仅提供候选框位置；原始RPN输出没有detach，仍用于RPN loss。四步交替中训练检测器时RPN固定，不需要让检测器loss穿过NMS。

以后学习近似联合训练时，它同样忽略proposal坐标的梯度路径，以各自loss训练各自的头，共享backbone接收两边梯度，不是让NMS变成可导。[论文关于交替与联合训练的说明](https://arxiv.org/html/1506.01497v3#S3.SS2)

## 9. Proposal、验证、预测画图

RPN用objectness：decode修anchor → clip/去小框 → 排序 → 类别无关NMS，IoU阈值0.7。训练/验证loss最多2000个框，预测最多300个，上限不等于实际数量。

验证使用每轮更新后的模型，eval + no_grad + 固定采样，仅算当前阶段loss，没有backward/step。GT用于构造loss答案，不用于生成RPN proposals。

预测无GT分配、无64框采样，全部RPN框进入检测器，再对每个VOC类别decode、分数过滤、同类别NMS；默认score>=0.5、类别NMS阈值0.3。这是第二次修框/第二次NMS，和RPN不是同一步。

GT只在预测后用于画图。GT和预测一起从resize坐标还原到原图，保存test_results/<image_id>.jpg；左侧绿色真实框，右侧红色预测框。默认最多10张，不计算mAP。

## 10. Checkpoint和当前限制

```text
stage1_best.pth → 阶段1 RPN
stage2_best.pth → 阶段2独立检测器
stage3_best.pth → 阶段3共享backbone + 调整后的RPN
best.pth       → 阶段4最终Faster R-CNN，供only_test使用
```

每阶段按自己的val loss选best，保存stage、epoch、模型权重、SGD状态、val_loss和数据划分。下一阶段加载上一阶段best，重新创建SGD，不沿用之前的momentum。没有last/自动续训入口，保存SGD状态不等于实现续训。test要求stage=4，防止误用旧Fast R-CNN或中间阶段权重。

当前限制：

- 已接通四阶段控制流，但轮数、学习率配置、backbone冻结范围等不等同完整论文实验；尚无正式训练或mAP成绩。
- 没有mAP/AP或标准VOC difficult评测处理，val loss不是最终检测指标。
- 不足采样数时不重复补满，没有随机翻转等增强，也未拼入GT保证检测器前景候选数。
- 第2阶段保留两份模型，显存还受图片大小影响。
- 没有随机种子管理、持久化曲线、自动续训或复杂实验管理。
- 未加入AMP、DDP、scheduler、WandB等功能。

轻量检查使用真实本地VOC读取/resize/标签，真实随机初始化VGG小输入前向和单步梯度，以及小模型四阶段更新/冻结、eval、checkpoint、预测画图与main调度。没有完整训练、下载数据/权重或检查服务器成绩。随机输出不是检测效果展示。

数据、checkpoint、缓存、虚拟环境和test_results不应提交Git。本次没有自动git add、commit或push。
## 11.预测结果展示
<img width="705" height="249" alt="image" src="https://github.com/user-attachments/assets/5ecfff5a-7f1e-4eac-b695-a91ad8a34ce9" />
<img width="524" height="349" alt="image" src="https://github.com/user-attachments/assets/b0225b72-aa4f-468e-9113-ea1cfade8a5c" />
<img width="817" height="120" alt="image" src="https://github.com/user-attachments/assets/d4c96f97-f0c6-4489-aa3b-a8eae0091d78" />
<img width="404" height="101" alt="image" src="https://github.com/user-attachments/assets/4e60c6fc-ed15-49bf-bf51-62c057573fe6" />


