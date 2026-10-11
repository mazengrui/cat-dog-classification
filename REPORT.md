# 猫狗分类项目训练报告

## 1. 项目概述

本项目实现一个基于 PyTorch 的猫狗二分类图像识别系统。目标是让模型能够识别一张图片是“猫”还是“狗”。

本项目采用的是卷积神经网络（CNN）进行图像特征提取和分类。它适合初学者学习深度学习图像处理的完整流程，包括：

- 数据收集与整理
- 数据增强
- 模型构建
- 训练与验证
- 模型保存
- 单张图片预测

## 2. 项目环境

建议使用 Python 3.11 版本，原因如下：

- Python 3.14 对 PyTorch 的兼容性不稳定
- Windows 下 Python 3.11 与 PyTorch 的配合最稳妥
- 训练和预测过程在 3.11 环境中更稳定

推荐环境：

- Windows 10 / 11
- Python 3.11.x
- PyTorch
- torchvision
- numpy
- matplotlib
- pillow
- tqdm

## 3. 数据集说明

本项目使用的是猫狗分类数据集，文件目录通常如下：

```text
data/
├── raw/
│   └── PetImages/
│       ├── Cat/
│       └── Dog/
├── train/
│   ├── cat/
│   └── dog/
├── validation/
│   ├── cat/
│   └── dog/
└── ...
```

其中：

- Cat 目录保存猫图片
- Dog 目录保存狗图片
- raw/PetImages 是原始数据目录
- train 和 validation 用于模型训练和验证

### 3.1 数据预处理思路

数据处理的主要目标：

- 统一图片大小（224×224）
- 随机加入旋转、镜像、颜色抖动等增强
- 按 80/20 划分训练集和验证集
- 检测并跳过损坏图片

## 4. 项目结构

```text
cat-dog-pytorch/
├── data/
│   ├── raw/
│   │   └── PetImages/
│   │       ├── Cat/
│   │       └── Dog/
│   ├── train/
│   │   ├── cat/
│   │   └── dog/
│   └── validation/
│       ├── cat/
│       └── dog/
├── checkpoints/
│   └── cat_dog_best.pth
├── src/
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── split_petimages_safe.py
├── requirements.txt
├── README.md
├── train_history.png
├── test_images/
│   ├── cat_test.jpg
│   └── dog_test.jpg
└── ...
```

## 5. 开发环境搭建

### 5.1 创建虚拟环境

如果使用 conda：

```bash
conda create -n py311 python=3.11
conda activate py311
```

如果使用 venv：

```bash
python -m venv venv311
venv311\Scripts\activate
```

### 5.2 安装依赖

```bash
pip install torch torchvision torchaudio pillow matplotlib tqdm numpy
```

### 5.3 验证安装

```bash
python -c "import torch; print(torch.__version__)"
```

如果命令正常输出类似 2.x.x，则 PyTorch 安装成功。

## 6. 数据整理脚本

### 6.1 数据整理脚本：split_petimages_safe.py

```python
import os
import shutil
import random
from PIL import Image, UnidentifiedImageError

random.seed(42)

SRC_BASE = os.path.join("data", "raw", "PetImages")
CAT_SRC = os.path.join(SRC_BASE, "Cat")
DOG_SRC = os.path.join(SRC_BASE, "Dog")

TRAIN_CAT = os.path.join("data", "train", "cat")
TRAIN_DOG = os.path.join("data", "train", "dog")
VAL_CAT = os.path.join("data", "validation", "cat")
VAL_DOG = os.path.join("data", "validation", "dog")

for d in [TRAIN_CAT, TRAIN_DOG, VAL_CAT, VAL_DOG]:
    os.makedirs(d, exist_ok=True)


def valid_images_from_folder(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts) and os.path.isfile(os.path.join(folder, f))]
    valid = []
    for f in files:
        path = os.path.join(folder, f)
        try:
            with Image.open(path) as im:
                im.verify()
            valid.append(f)
        except (UnidentifiedImageError, OSError, IOError):
            print(f"跳过损坏或无法打开的文件: {path}")
    return valid


def split_and_copy(src_folder, train_dst, val_dst, ratio=0.8):
    files = valid_images_from_folder(src_folder)
    random.shuffle(files)
    cut = int(len(files) * ratio)
    train_files = files[:cut]
    val_files = files[cut:]

    for fname in train_files:
        shutil.copy2(os.path.join(src_folder, fname), os.path.join(train_dst, fname))
    for fname in val_files:
        shutil.copy2(os.path.join(src_folder, fname), os.path.join(val_dst, fname))

    return len(train_files), len(val_files), len(files)

if not (os.path.isdir(CAT_SRC) and os.path.isdir(DOG_SRC)):
    print("错误：找不到 Cat/Dog 文件夹")
    raise SystemExit(1)

cat_train_n, cat_val_n, cat_total = split_and_copy(CAT_SRC, TRAIN_CAT, VAL_CAT, ratio=0.8)
dog_train_n, dog_val_n, dog_total = split_and_copy(DOG_SRC, TRAIN_DOG, VAL_DOG, ratio=0.8)
print("拆分完成：")
print(f"Cat: total={cat_total}, train={cat_train_n}, val={cat_val_n}")
print(f"Dog: total={dog_total}, train={dog_train_n}, val={dog_val_n}")
```

### 6.2 运行方式

```bash
python split_petimages_safe.py
```

脚本执行后会：

- 检测损坏图片
- 将图片随机打乱
- 按 80% 训练 / 20% 验证比例分配
- 拷贝到 data/train 和 data/validation 中

## 7. CNN 模型设计

### 7.1 模型文件：src/model.py

```python
import torch
import torch.nn as nn

class CatDogCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

### 7.2 模型结构说明

- 输入：224×224×3
- 卷积层负责提取边缘、纹理、轮廓等图像特征
- MaxPool 层负责降采样和降低参数量
- Flatten 将特征图压成一维
- Dense 层负责最终分类
- Dropout 防止过拟合
- 输出层长度为 2（猫、狗）

## 8. 训练代码

### 8.1 训练脚本：src/train.py

```python
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import CatDogCNN
from utils import plot_training_history


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

train_dir = "data/train"
val_dir = "data/validation"
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

img_size = 224
batch_size = 32
epochs = 15
learning_rate = 1e-4

train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print("类别:", train_dataset.classes)
print("类别映射:", train_dataset.class_to_idx)

model = CatDogCNN(num_classes=len(train_dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_val_acc = 0.0
history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - 训练中"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = running_correct / total

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    print("-" * 60)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "class_to_idx": train_dataset.class_to_idx
        }
        torch.save(ckpt, os.path.join(checkpoint_dir, "cat_dog_best.pth"))

print(f"训练完成，最佳验证准确率: {best_val_acc:.4f}")

# 画训练曲线并保存图片
from utils import plot_training_history
plot_training_history(history)
```

### 8.2 训练过程说明

- 训练时使用 `ImageFolder` 自动从目录中读取图片
- 每一轮训练都会输出：Train Loss / Train Accuracy / Val Loss / Val Accuracy
- 只保存验证集最高正确率对应的模型参数，避免过拟合中较差的模型被保留下来
- 最终得到的模型文件为：`checkpoints/cat_dog_best.pth`

## 9. 预测脚本

### 9.1 预测脚本：src/predict.py

```python
import argparse
import torch
from torchvision import transforms
from PIL import Image
from model import CatDogCNN


def load_model(model_path):
    ckpt = torch.load(model_path, map_location="cpu")
    class_to_idx = ckpt["class_to_idx"]
    classes = list(class_to_idx.keys())

    model = CatDogCNN(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, classes


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="要预测的图片路径")
    parser.add_argument("--model", type=str, default="checkpoints/cat_dog_best.pth", help="模型路径")
    args = parser.parse_args()

    model, classes = load_model(args.model)
    image_tensor = preprocess_image(args.image)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
        label = classes[pred_idx]

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
```

### 9.2 运行方式

```bash
python src\predict.py --image test_images\cat_test.jpg
```

输出示例：

```text
Prediction: cat
Confidence: 0.9123
```

## 10. 训练命令

在项目根目录执行：

```bash
python src\train.py
```

如果训练很慢，可以先减少轮数：

```python
epochs = 5
```

如果电脑性能弱，可以把 batch size 改为 16：

```python
batch_size = 16
```

## 11. 结果解释

训练结束后，通常会得到：

- train_history.png：损失和准确率曲线图
- checkpoints/cat_dog_best.pth：最佳模型参数

如果验证集准确率接近 80% 以上，说明模型效果已经比较好；如果偏低，则说明可能需要：

- 增加训练样本
- 调整学习率
- 继续训练更多轮数
- 使用更强的数据增强
- 采用迁移学习

## 12. 常见问题与修复方案

### 12.1 No module named torch

原因：当前环境没有安装 PyTorch。

解决方案：

```bash
pip install torch torchvision torchaudio
```

### 12.2 Python 版本兼容问题

若看到 Python 3.14，说明当前环境不适合本项目。

解决方案：

- 使用 Python 3.11 环境
- conda create -n py311 python=3.11
- conda activate py311

### 12.3 Windows 下 DataLoader 报错

原因：Windows 下多进程可能出错。

解决方案：

- 在训练脚本中将 `num_workers=0`
- 这是本项目中已采取的方式

### 12.4 数据目录为空

原因：原始数据目录没有真正放入图片。

解决方案：

- 检查 `data/raw/PetImages/Cat` 和 `Dog` 中是否含有 jpg/png 文件
- 如果有一层额外目录，移动到正确位置

### 12.5 训练曲线异常

如果训练集准确率很高，但验证集很低，说明过拟合。解决方法：

- 加大 Dropout
- 增加数据增强
- 减少网络复杂度
- 继续收集图片

## 13. 总结

本项目可以帮助初学者完整理解深度学习图像处理的基本流程：

- 数据准备
- 数据增强
- CNN 模型搭建
- 训练/验证
- 模型保存
- 单张图片预测

该项目可以作为：

- 课程作业
- 个人作品集
- 毕业设计基础项目
- 深度学习入门实战案例

## 14. 后续提升方向

如果想进一步提升项目质量，可以考虑：

- 使用迁移学习（ResNet18、MobileNetV2）
- 增加更多类别
- 做一个 Streamlit 网页界面
- 做实时摄像头识别
- 增加混淆矩阵和精确率/召回率分析

## 15. 结论

本项目已经覆盖了典型的深度学习图像分类流程，拥有以下实际意义：

- 完整实现了图像分类任务
- 代码结构清晰
- 训练和预测均可以直接运行
- 易于扩展到更复杂的图像任务

如果做完本项目，后续再学习目标检测、图像分割或迁移学习会更容易。

---

“本报告已整理完成，可直接复制到 Word 中保存为 .docx 或 .pdf。”

如果需要，我可以继续为你补充：

- 论文风格版本
- 简洁汇报版（适合答辩）
- 适合提交到课程系统的版本
- 适合写进 README 的版本

如果你希望，我下一步可以把“答辩版”的报告内容再压缩成 1 页摘要版本。 
