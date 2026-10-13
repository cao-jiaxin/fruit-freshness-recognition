# -*- coding: utf-8 -*-
"""
水果种类及新鲜度检测系统 — Flask 后端 API
提供 3 个接口：
  POST /api/predict       单张图片识别
  POST /api/batch_predict  批量图片识别
  GET  /api/model_info     模型信息
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory

# ==================== 常量 ====================
CLASS_NAMES_CN = [
    '新鲜苹果', '新鲜香蕉', '新鲜橘子',
    '腐烂苹果', '腐烂香蕉', '腐烂橘子',
]

CLASS_EMOJI = ['🍎', '🍌', '🍊', '🥀', '🟤', '🟠']

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224

#  Flask 应用 
app = Flask(__name__)


# 模型加载（启动时加载一次） 
_model = None


def get_model():
    """获取全局模型实例（单例）"""
    global _model
    if _model is None:
        _model = _load_model()
    return _model


def _load_model():
    """加载 MobileNetV2 模型权重"""
    # 查找模型文件
    model_paths = [
        './checkpoints/model_for_app.pth',
        './checkpoints/best_model.pth',
    ]
    model_file = None
    for path in model_paths:
        if os.path.exists(path):
            model_file = path
            break

    if model_file is None:
        raise FileNotFoundError(
            "未找到模型文件！请先运行 train.py 训练模型，"
            "确保 checkpoints/ 目录下有 model_for_app.pth 或 best_model.pth"
        )

    print(f"[INFO] 加载模型: {model_file}")

    # 加载权重
    checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)

    # 构建 MobileNetV2
    model = models.mobilenet_v2(weights=None)

    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 6)
    )

    # 兼容不同保存格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])#用训练的权重覆盖随机值
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("[INFO] 模型加载完成 OK")
    return model


# ==================== 预处理 ====================
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """将 PIL 图片转为模型输入张量 [1, 3, 224, 224]"""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(image).unsqueeze(0)


# ==================== 推理 ====================
@torch.no_grad()
def predict_single(image: Image.Image):
    """对单张图片推理，返回 (类别索引, 置信度, 概率数组)"""
    model = get_model()
    tensor = preprocess_image(image)
    output = model(tensor)
    probs = torch.softmax(output, dim=1)[0]
    pred_idx = output.argmax(dim=1).item()
    confidence = probs[pred_idx].item()
    return pred_idx, confidence, probs.numpy()


# ==================== API 1: 单张识别 ====================
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    单张图片识别
    请求: multipart/form-data, 字段名 'file'
    返回: {
        "pred_class": "新鲜苹果",
        "pred_index": 0,
        "confidence": 95.32,
        "probabilities": [95.32, 0.15, 0.08, 4.21, 0.12, 0.12]
    }
    """
    if 'file' not in request.files:
        return jsonify({'error': '未上传文件，请使用 file 字段'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    try:
        image = Image.open(file.stream).convert('RGB')
        pred_idx, confidence, probs = predict_single(image)

        return jsonify({
            'pred_class': CLASS_NAMES_CN[pred_idx],
            'pred_index': pred_idx,
            'pred_emoji': CLASS_EMOJI[pred_idx],
            'confidence': round(confidence * 100, 2),
            'probabilities': [round(float(p) * 100, 2) for p in probs],
        })
    except Exception as e:
        return jsonify({'error': f'识别失败: {str(e)}'}), 500


# ==================== API 2: 批量识别 ====================
@app.route('/api/batch_predict', methods=['POST'])
def api_batch_predict():
    """
    批量图片识别
    请求: multipart/form-data, 字段名 'files'（多个文件）
    返回: {
        "total": 3,
        "results": [
            {"filename": "apple1.jpg", "pred_class": "新鲜苹果", "confidence": 95.32, ...},
            ...
        ]
    }
    """
    if 'files' not in request.files:
        return jsonify({'error': '未上传文件，请使用 files 字段（支持多文件）'}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': '未选择任何文件'}), 400

    results = []
    for file in files:
        if file.filename == '':
            continue
        try:
            image = Image.open(file.stream).convert('RGB')
            pred_idx, confidence, probs = predict_single(image)
            results.append({
                'filename': file.filename,
                'pred_class': CLASS_NAMES_CN[pred_idx],
                'pred_index': pred_idx,
                'pred_emoji': CLASS_EMOJI[pred_idx],
                'confidence': round(confidence * 100, 2),
            })
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e),
            })

    return jsonify({
        'total': len(results),
        'results': results,
    })


# ==================== API 3: 模型信息 ====================
@app.route('/api/model_info', methods=['GET'])
def api_model_info():
    """
    返回模型信息
    """
    return jsonify({
        'model_name': 'MobileNetV2',
        'architecture': 'MobileNetV2 (迁移学习)',
        'input_size': [3, 224, 224],
        'num_classes': 6,
        'class_names': CLASS_NAMES_CN,
        'class_emoji': CLASS_EMOJI,
        'framework': 'PyTorch',
        'pretrained': 'ImageNet V1',
        'parameters': '约 3.5M',
        'onnx_available': os.path.exists('./checkpoints/fruit_model.onnx'),
    })


# ==================== 前端页面 ====================
@app.route('/')
def index():
    """返回前端页面"""
    return send_from_directory('templates', 'frontend.html')


# ==================== 跨域支持（允许直接打开 HTML 文件调试） ====================
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


@app.route('/api/predict', methods=['OPTIONS'])
@app.route('/api/batch_predict', methods=['OPTIONS'])
def handle_options():
    """处理跨域预检请求"""
    return '', 204


# ==================== 启动 ====================
if __name__ == '__main__':
    get_model()
    app.run(host='0.0.0.0', port=5000, debug=True)