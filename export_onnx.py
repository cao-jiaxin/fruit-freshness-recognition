# -*- coding: utf-8 -*-
"""
模型导出为 ONNX 格式，用于跨平台部署
"""

import os
import torch
from train import build_model


def export_to_onnx(model_path, onnx_path, input_size=(1, 3, 224, 224)):
    """将 PyTorch 模型导出为 ONNX 格式"""
    device = torch.device('cpu')

    # 加载模型
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = build_model(num_classes=6)
    model.load_state_dict(checkpoint['model_state_dict'])#填入模型参数
    model.eval()#进入推理模式

    # 创建虚拟输入
    dummy_input = torch.randn(*input_size)

    # 导出 ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }
    )

    print(f"ONNX 模型已导出至: {onnx_path}")

    # 验证 ONNX 模型
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX 模型验证通过 ✓")

    # 文件大小
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"模型文件大小: {size_mb:.2f} MB")


def main():
    print("=" * 50)
    print("导出 ONNX 模型")
    print("=" * 50)

    os.makedirs('./checkpoints', exist_ok=True)

    model_path = './checkpoints/best_model.pth'
    onnx_path = './checkpoints/fruit_model.onnx'

    export_to_onnx(model_path, onnx_path)

    # 也保存完整 PyTorch 脚本模型（用于直接用 torch 加载）
    device = torch.device('cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = build_model(num_classes=6)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 保存 state_dict（简洁版，给 app 用）
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names_cn': checkpoint.get('class_names_cn', [
            '新鲜苹果', '新鲜香蕉', '新鲜橘子',
            '腐烂苹果', '腐烂香蕉', '腐烂橘子'
        ]),
    }, './checkpoints/model_for_app.pth')
    print("应用版模型已保存至: ./checkpoints/model_for_app.pth")


if __name__ == '__main__':
    main()