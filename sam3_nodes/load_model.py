"""
SAM3 Model Loader for ComfyUI
直接使用sam3官方代码，提供薄包装层
"""

import os
import sys
import torch

# 添加sam3到Python路径
sam3_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sam3")
if sam3_dir not in sys.path:
    sys.path.insert(0, sam3_dir)

# 导入SAM3官方组件
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def load_sam3_model(
    model_path,
    bpe_path,
    device="cuda",
    precision="bf16",
    confidence_threshold=0.5
):
    """
    加载SAM3模型并创建processor

    Args:
        model_path: 模型权重路径 (sam3.pt)
        bpe_path: BPE tokenizer路径
        device: 设备 (cuda/cpu)
        precision: 精度 (bf16/fp16/fp32)
        confidence_threshold: 置信度阈值

    Returns:
        tuple: (model, processor, autocast_ctx)
    """
    print(f"Loading SAM3 model from: {model_path}")

    # 1. 设置TF32（对齐官方）
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 2. 设置精度
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32
    }[precision]

    # 3. 启用autocast（对齐官方demo）
    autocast_ctx = None
    if device == "cuda" and precision != "fp32":
        autocast_ctx = torch.autocast("cuda", dtype=dtype)
        autocast_ctx.__enter__()

    # 4. 构建模型
    load_from_HF = not os.path.exists(model_path)

    model = build_sam3_image_model(
        bpe_path=bpe_path,
        device=device,
        eval_mode=True,
        checkpoint_path=model_path if not load_from_HF else None,
        load_from_HF=load_from_HF,
        enable_segmentation=True,
        enable_inst_interactivity=False,  # 第一版不启用
        compile=False
    )

    # 5. 创建processor
    processor = Sam3Processor(
        model,
        confidence_threshold=confidence_threshold
    )

    print(f"✓ SAM3 model loaded successfully")

    return model, processor, autocast_ctx


def unload_sam3_model(autocast_ctx):
    """卸载模型和autocast上下文"""
    if autocast_ctx is not None:
        autocast_ctx.__exit__(None, None, None)
