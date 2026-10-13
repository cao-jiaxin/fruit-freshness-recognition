# -*- coding: utf-8 -*-
"""
MobileNetV2 迁移学习 - 水果新鲜度6分类训练脚本
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib
matplotlib.use('Agg')  # 非交互后端，避免无 GUI 报错
import matplotlib.pyplot as plt

# 解决中文显示乱码：指定中文字体（Windows 用黑体）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

from dataset import create_dataloaders, CLASS_NAMES_CN

#   超参数配置 统一存放全部参数的字典，方便调参
CONFIG = {
    # 数据
    'data_dir': 'D:/Claude_Workspace/archive/dataset',
    'batch_size': 32,
    'val_ratio': 0.1,
    'num_workers': 0,  # Windows 多进程兼容问题，设为 0

    # 模型
    'num_classes': 6,#  6类
    'pretrained': True, #预训练，大幅度提升准确率

    # 训练
    'epochs': 15,
    'lr': 0.001,         #学习率
    'lr_step': 7,        # 每隔多少 epoch 降低学习率
    'lr_gamma': 0.1,     # 学习率衰减因子，前期大步快速收敛，后期小幅度精细微调权重。
    'weight_decay': 1e-4,#正则化，权重衰减，打压某些过大的参数

    # 断点续训
    'resume': True,  # 是否自动从 checkpoint 恢复

    # 输出
    'save_dir': './checkpoints',#训练过程中效果最好的模型、每轮断点权重都会保存在当前项目下 checkpoints 文件夹。
}

#  模型构建 

def build_model(num_classes=6, pretrained=True):
    """构建 MobileNetV2 迁移学习模型"""
    print(f"加载 MobileNetV2 (pretrained={pretrained})...")

    from torchvision.models import MobileNet_V2_Weights
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    # 替换分类头
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )

    return model


# 训练与验证

def train_one_epoch(model, loader, optimizer, criterion, device):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()#清空上一轮的梯度
        outputs = model(images)#图片放进网络，得到预测分数，前向传播
        loss = criterion(outputs, labels)#计算预测和真实的差距
        loss.backward()#反向传播
        optimizer.step()#根据梯度更新网络权重

        running_loss += loss.item()
        _, predicted = outputs.max(1)#取分数最大的类别
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 每 20 个 batch 打印一次进度
        if (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()#验证  时候不计算梯度
def validate(model, loader, criterion, device):
    """验证"""
    model.eval()#关闭dropout
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
#没有清空梯度，反向传播，更新参数
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


#  学习曲线绘制 

def plot_curves(history, save_dir):
    """绘制 Loss 和 Accuracy 曲线"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss 曲线
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='训练损失', markersize=4)
    axes[0].plot(epochs, history['val_loss'], 'r-o', label='验证损失', markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('训练 / 验证损失曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy 曲线
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='训练准确率', markersize=4)
    axes[1].plot(epochs, history['val_acc'], 'r-o', label='验证准确率', markersize=4)
    axes[1].axhline(y=95, color='g', linestyle='--', label='95% 目标线')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('训练 / 验证准确率曲线')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"学习曲线已保存至: {plot_path}")


#  主训练流程 

def main():
    print("=" * 60)
    print("MobileNetV2 迁移学习 - 水果新鲜度6分类")
    print("=" * 60)

    # 设备检测
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"训练设备: {device}")
    print(f"PyTorch 版本: {torch.__version__}")

    # 创建保存目录
    os.makedirs(CONFIG['save_dir'], exist_ok=True)

    # ========== 数据加载 ==========
    print("\n" + "=" * 40)
    print("加载数据集...")
    train_loader, val_loader, test_loader = create_dataloaders(
        CONFIG['data_dir'],
        batch_size=CONFIG['batch_size'],
        val_ratio=CONFIG['val_ratio'],
        num_workers=CONFIG['num_workers'],
    )
    print(f"训练集: {len(train_loader.dataset)} 样本, "
          f"{len(train_loader)} 批次")
    print(f"验证集: {len(val_loader.dataset)} 样本, "
          f"{len(val_loader)} 批次")
    print(f"测试集: {len(test_loader.dataset)} 样本, "
          f"{len(test_loader)} 批次")

    # ========== 模型构建 ==========
    print("\n" + "=" * 40)
    model = build_model(
        num_classes=CONFIG['num_classes'],
        pretrained=CONFIG['pretrained'],
    )
    model = model.to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # ========== 优化器和损失函数 ==========
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'],
                           weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=CONFIG['lr_step'], gamma=CONFIG['lr_gamma']
    )

    # ========== 断点续训检查 ==========
    checkpoint_path = os.path.join(CONFIG['save_dir'], 'best_model.pth')
    start_epoch = 1
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [],  'val_acc': [],
    }
    best_val_acc = 0.0

    if CONFIG['resume'] and os.path.exists(checkpoint_path):
        print("\n" + "=" * 40)
        print("检测到已有 checkpoint，恢复训练...")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('val_acc', 0.0)
        history = ckpt.get('history', history)
        print(f"从 Epoch {start_epoch} 继续 (最佳 Val Acc: {best_val_acc:.2f}%)")
        # 更新 scheduler 到对应步数
        for _ in range(ckpt['epoch']):
            scheduler.step()

    # ========== 训练循环 ==========
    print("\n" + "=" * 40)
    print("开始训练..." if start_epoch == 1 else "继续训练...")
    print(f"Epochs: {start_epoch}->{CONFIG['epochs']} | LR: {CONFIG['lr']} | "
          f"Batch Size: {CONFIG['batch_size']}")

    for epoch in range(start_epoch, CONFIG['epochs'] + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n--- Epoch {epoch}/{CONFIG['epochs']} (LR: {current_lr:.6f}) ---")

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 记录
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # 学习率衰减
        scheduler.step()

        # 打印结果
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history,
                'config': CONFIG,
                'class_names_cn': CLASS_NAMES_CN,
            }, checkpoint_path)
            print(f"  [OK] 最佳模型已保存 (Val Acc: {val_acc:.2f}%)")

    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.2f}%")

    # ========== 画曲线 ==========
    plot_curves(history, CONFIG['save_dir'])

    print("\n" + "=" * 60)
    print("训练阶段完成！最佳模型: best_model.pth")
    print("=" * 60)


if __name__ == '__main__':
    main()