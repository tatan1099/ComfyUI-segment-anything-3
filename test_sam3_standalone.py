#!/usr/bin/env python3
"""
SAM3 本地测试脚本
在ComfyUI外独立测试SAM3推理流程
验证：模型加载 -> 图像输入 -> 文本提示 -> masks输出 -> 可视化
"""

import os
import sys
import torch
from PIL import Image
import matplotlib.pyplot as plt

# 添加sam3到路径
sam3_path = os.path.join(os.path.dirname(__file__), "sam3")
sys.path.insert(0, sam3_path)

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results

def main():
    print("=" * 60)
    print("SAM3 本地测试脚本")
    print("=" * 60)

    # 配置路径
    bpe_path = os.path.join(os.path.dirname(__file__), "sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    model_path = "/root/ComfyUI/models/sam3/sam3.pt"
    test_image_path = os.path.join(os.path.dirname(__file__), "sam3/assets/images/test_image.jpg")

    print(f"\n✓ BPE路径: {bpe_path}")
    print(f"✓ 模型路径: {model_path}")
    print(f"✓ 测试图片: {test_image_path}")

    # 检查文件是否存在
    assert os.path.exists(bpe_path), f"BPE文件不存在: {bpe_path}"
    assert os.path.exists(model_path), f"模型文件不存在: {model_path}"
    assert os.path.exists(test_image_path), f"测试图片不存在: {test_image_path}"

    # Step 1: 启用autocast（对齐官方demo）
    print("\n" + "=" * 60)
    print("Step 1: 启用autocast (bf16)")
    print("=" * 60)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    print("✓ Autocast已启用")

    # Step 2: 加载模型
    print("\n" + "=" * 60)
    print("Step 2: 加载SAM3模型")
    print("=" * 60)
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=model_path,
        load_from_HF=False,
        device="cuda",
        eval_mode=True
    )
    print("✓ 模型加载成功")

    # Step 3: 创建processor
    print("\n" + "=" * 60)
    print("Step 3: 创建Sam3Processor")
    print("=" * 60)
    processor = Sam3Processor(model, confidence_threshold=0.5)
    print("✓ Processor创建成功")

    # Step 4: 加载测试图片
    print("\n" + "=" * 60)
    print("Step 4: 加载测试图片")
    print("=" * 60)
    image = Image.open(test_image_path)
    width, height = image.size
    print(f"✓ 图片尺寸: {width} x {height}")

    # Step 5: 设置图片
    print("\n" + "=" * 60)
    print("Step 5: 设置图片并计算backbone特征")
    print("=" * 60)
    inference_state = processor.set_image(image)
    print("✓ 图片已设置")
    print(f"✓ inference_state keys: {list(inference_state.keys())}")

    # Step 6: 文本提示推理
    print("\n" + "=" * 60)
    print("Step 6: 使用文本提示 'shoe' 进行推理")
    print("=" * 60)
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(
        state=inference_state,
        prompt="head"
    )
    print("✓ 推理完成")

    # Step 7: 提取结果
    print("\n" + "=" * 60)
    print("Step 7: 提取结果")
    print("=" * 60)
    masks = inference_state["masks"]
    boxes = inference_state["boxes"]
    scores = inference_state["scores"]

    print(f"✓ 检测到 {len(masks)} 个对象")
    print(f"✓ Masks shape: {masks.shape}")
    print(f"✓ Boxes shape: {boxes.shape}")
    print(f"✓ Scores: {scores.cpu().float().numpy()}")

    # Step 8: 验证输出格式
    print("\n" + "=" * 60)
    print("Step 8: 验证输出格式")
    print("=" * 60)
    print(f"✓ masks dtype: {masks.dtype}")
    print(f"✓ masks 是否为bool: {masks.dtype == torch.bool}")
    print(f"✓ boxes dtype: {boxes.dtype}")
    print(f"✓ scores dtype: {scores.dtype}")

    # 转换为ComfyUI格式（合并所有实例）
    masks_squeezed = masks.squeeze(1)  # [N, 1, H, W] -> [N, H, W]
    merged_mask = torch.any(masks_squeezed, dim=0, keepdim=True)  # [1, H, W]
    masks_comfy = merged_mask.cpu().float()
    print(f"✓ ComfyUI格式 masks shape: {masks_comfy.shape} (所有实例已合并)")

    # Step 9: 可视化
    print("\n" + "=" * 60)
    print("Step 9: 可视化结果")
    print("=" * 60)
    output_path = "./test_sam3_output.png"
    plot_results(image, inference_state)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"✓ 可视化已保存到: {output_path}")

    # Step 10: 测试另一个文本提示
    print("\n" + "=" * 60)
    print("Step 10: 测试文本提示 'head'")
    print("=" * 60)
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(
        state=inference_state,
        prompt="head"
    )

    masks2 = inference_state["masks"]
    scores2 = inference_state["scores"]
    print(f"✓ 检测到 {len(masks2)} 个head")
    print(f"✓ Scores: {scores2.cpu().float().numpy()}")

    output_path2 = "./test_sam3_output_head.png"
    plot_results(image, inference_state)
    plt.savefig(output_path2, bbox_inches='tight', dpi=150)
    print(f"✓ 可视化已保存到: {output_path2}")

    print("\n" + "=" * 60)
    print("✅ 所有测试通过！SAM3推理流程正常")
    print("=" * 60)

if __name__ == "__main__":
    main()
