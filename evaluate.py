# -*- coding: utf-8 -*-
"""
模型评估脚本：混淆矩阵、分类报告（精确率/召回率/F1）、错误样本分析
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score)
import seaborn as sns

# 解决中文显示乱码：指定中文字体（Windows 用黑体）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

from dataset import create_dataloaders, CLASS_NAMES_CN
from train import build_model

#跑一遍测试集
@torch.no_grad()
def get_predictions(model, loader, device):
    """获取所有预测结果和真实标签"""
    model.eval()
    all_preds = []#存所有的预测结果
    all_labels = []#存所有的真实标签
    all_probs = []#存所有的概率值

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)#把分数 转换成概率
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return (np.array(all_preds), np.array(all_labels), np.array(all_probs))

#混淆矩阵
def plot_confusion_matrix(y_true, y_pred, save_path):
    """绘制混淆矩阵热力图"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES_CN,
                yticklabels=CLASS_NAMES_CN,
                annot_kws={'size': 12})
    plt.xlabel('预测类别', fontsize=13)
    plt.ylabel('真实类别', fontsize=13)
    plt.title('混淆矩阵', fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"混淆矩阵已保存至: {save_path}")

#找错误的样本
def find_error_samples(y_true, y_pred, probs, max_show=10):
    """分析识别错误的样本"""
    errors = []
    for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        if true_label != pred_label:
            errors.append({
                'true_class': CLASS_NAMES_CN[true_label],#实际是什么
                'pred_class': CLASS_NAMES_CN[pred_label],
                'confidence': probs[i][pred_label],#被错判的置信率
                'true_confidence': probs[i][true_label],
            })

    print(f"\n错误样本总数: {len(errors)}/{len(y_true)}")
    print(f"错误率: {len(errors)/len(y_true)*100:.2f}%")

    if errors:
        print(f"\n前 {min(max_show, len(errors))} 个错误样本:")
        print(f"{'真实类别':<10} {'预测类别':<10} {'预测置信度':<12} {'真实类别概率':<12}")
        print("-" * 50)
        for e in errors[:max_show]:
            print(f"{e['true_class']:<10} {e['pred_class']:<10} "
                  f"{e['confidence']*100:>7.2f}%     "
                  f"{e['true_confidence']*100:>7.2f}%")

    return errors


def plot_class_metrics(report_dict, save_path):
    """绘制每类指标柱状图"""
    classes = []
    precision_list = []
    recall_list = []
    f1_list = []

    for cls_name, metrics in report_dict.items():
        if cls_name in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        classes.append(cls_name)
        precision_list.append(metrics['precision'] * 100)
        recall_list.append(metrics['recall'] * 100)
        f1_list.append(metrics['f1-score'] * 100)

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision_list, width, label='精确率', color='#5B9BD5')
    #精确率，准
    bars2 = ax.bar(x, recall_list, width, label='召回率', color='#ED7D31')
    #召回率，全
    bars3 = ax.bar(x + width, f1_list, width, label='F1分数', color='#70AD47')
    #F1，平衡二者
    ax.set_xlabel('类别', fontsize=13)
    ax.set_ylabel('百分比 (%)', fontsize=13)
    ax.set_title('每类分类指标对比', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)
    ax.axhline(y=95, color='r', linestyle='--', alpha=0.5, label='95% 线')
    ax.grid(axis='y', alpha=0.3)

    # 在柱子上标注数值
    for bar in bars1 + bars2 + bars3:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"指标柱状图已保存至: {save_path}")


def main():
    print("=" * 60)
    print("模型评估 — 混淆矩阵 & 分类报告")
    print("=" * 60)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 1加载模型
    model_path = './checkpoints/best_model.pth'
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = build_model(num_classes=6)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"模型已加载 (Val Acc: {checkpoint.get('val_acc', 'N/A')}%)")

    # 2加载 测试集
    _, _, test_loader = create_dataloaders(
        'D:/Claude_Workspace/archive/dataset',
        batch_size=32, val_ratio=0.1, num_workers=0
    )
    print(f"测试集: {len(test_loader.dataset)} 样本")

    # 3预测
    preds, labels, probs = get_predictions(model, test_loader, device)

    # 4算整体准确率
    test_acc = accuracy_score(labels, preds)
    print(f"\n{'='*40}")
    print(f"测试集准确率: {test_acc*100:.2f}%")
    print(f"{'='*40}")

    # 5混淆矩阵
    cm_path = './checkpoints/confusion_matrix.png'
    plot_confusion_matrix(labels, preds, cm_path)

    # 6分类报告
    report = classification_report(labels, preds,
                                    target_names=CLASS_NAMES_CN,
                                    digits=4, output_dict=True)
    print("\n分类报告:")
    print(classification_report(labels, preds,
                                 target_names=CLASS_NAMES_CN,
                                 digits=4))

    # 7每类指标柱状图
    metrics_path = './checkpoints/class_metrics.png'
    plot_class_metrics(report, metrics_path)

    # 8错误样本分析
    errors = find_error_samples(labels, preds, probs)

    # 9保存评估摘要
    summary_path = './checkpoints/evaluation_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"测试集准确率: {test_acc*100:.2f}%\n\n")
        f.write("分类报告:\n")
        f.write(classification_report(labels, preds,
                                       target_names=CLASS_NAMES_CN,
                                       digits=4))
        f.write(f"\n错误样本总数: {len(errors)}\n")
        if errors:
            f.write("\n混淆模式统计:\n")
            from collections import Counter
            confusion_pairs = Counter(
                f"{CLASS_NAMES_CN[t]} -> {CLASS_NAMES_CN[p]}"
                for t, p in zip(labels, preds) if t != p
            )
            for pair, count in confusion_pairs.most_common():
                f.write(f"  {pair}: {count}次\n")

    print(f"\n评估摘要已保存至: {summary_path}")

    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()