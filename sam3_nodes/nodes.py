"""
SAM3 ComfyUI Nodes
支持开放词汇文本分割
"""

import torch
import os
import numpy as np
from PIL import Image

import comfy.model_management as mm
from comfy.utils import ProgressBar

from .load_model import load_sam3_model, unload_sam3_model

# 脚本目录
script_directory = os.path.dirname(os.path.abspath(__file__))


class DownloadAndLoadSAM3Model:
    """SAM3 模型加载器"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
        }

    RETURN_TYPES = ("SAM3MODEL",)
    RETURN_NAMES = ("sam3_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "SAM3"

    def loadmodel(self, device, precision, confidence_threshold):
        """加载SAM3模型"""

        # 1. 检查精度设置
        if precision != "fp32" and device == "cpu":
            raise ValueError("CPU只支持fp32精度，请选择fp32或切换到cuda")

        # 2. 设置路径
        model_path = "/root/ComfyUI/models/sam3/sam3.pt"
        bpe_path = os.path.join(
            os.path.dirname(script_directory),
            "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        )

        print(f"Model path: {model_path}")
        print(f"BPE path: {bpe_path}")

        # 3. 检查文件
        if not os.path.exists(bpe_path):
            raise FileNotFoundError(f"BPE文件不存在: {bpe_path}")

        if not os.path.exists(model_path):
            print(f"⚠️  模型文件不存在: {model_path}")
            print("将从Hugging Face自动下载...")
            print("注意：需要先运行 `huggingface-cli login` 并接受facebook/sam3的使用条款")

        # 4. 加载模型
        model, processor, autocast_ctx = load_sam3_model(
            model_path=model_path,
            bpe_path=bpe_path,
            device=device,
            precision=precision,
            confidence_threshold=confidence_threshold
        )

        # 5. 返回模型字典
        sam3_model = {
            "model": model,
            "processor": processor,
            "device": device,
            "precision": precision,
            "confidence_threshold": confidence_threshold,
            "autocast_ctx": autocast_ctx
        }

        return (sam3_model,)


class Sam3Segmentation:
    """
    SAM3 分割节点
    支持文本提示进行开放词汇分割
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sam3_model": ("SAM3MODEL",),
                "image": ("IMAGE",),
                "text_prompt": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "自然语言描述，如'shoe', 'person', 'red car'"
                }),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "image")
    FUNCTION = "segment"
    CATEGORY = "SAM3"

    def segment(self, image, sam3_model, text_prompt, keep_model_loaded):
        """
        执行SAM3分割

        Args:
            image: ComfyUI图像张量 [B, H, W, C] float32 [0, 1]
            sam3_model: SAM3模型字典
            text_prompt: 文本提示
            keep_model_loaded: 是否保持模型加载

        Returns:
            mask: ComfyUI mask张量 [1, H, W] float32 (所有实例合并)
            image: ComfyUI图像张量 [1, H, W, C] float32 (mask区域的原图)
        """

        # 提取模型组件
        processor = sam3_model["processor"]
        device = sam3_model["device"]

        # ===== Step 1: 图像格式转换 =====
        # ComfyUI: [B, H, W, C] float32 [0, 1]
        # SAM3需要: PIL Image
        B, H, W, C = image.shape

        if B > 1:
            print(f"⚠️  警告：当前只支持batch_size=1，将只处理第一张图片")

        # 转换为PIL Image
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        width, height = pil_image.size

        print(f"Processing image: {width} x {height}")

        # ===== Step 2: 设置图像 =====
        inference_state = processor.set_image(pil_image)

        # ===== Step 3: 解析文本提示 =====
        if not text_prompt or not text_prompt.strip():
            raise ValueError("必须提供text_prompt！例如：'person', 'shoe', 'car'")

        # 支持逗号分隔的多个提示
        prompts = [p.strip() for p in text_prompt.split(',') if p.strip()]
        if not prompts:
            raise ValueError("必须提供有效的text_prompt！")

        print(f"Text prompts: {prompts} ({len(prompts)} prompts)")

        # ===== Step 4: 执行推理（支持多个提示） =====
        all_masks = []
        total_objects = 0

        for i, prompt in enumerate(prompts):
            processor.reset_all_prompts(inference_state)
            inference_state = processor.set_text_prompt(
                state=inference_state,
                prompt=prompt
            )

            # 提取当前提示的结果
            masks = inference_state["masks"]  # [N, 1, H, W] bool
            scores = inference_state["scores"]  # [N,] bfloat16

            num_objects = len(masks)
            total_objects += num_objects
            print(f"  [{i+1}/{len(prompts)}] '{prompt}': {num_objects} objects")

            if num_objects > 0:
                print(f"    Scores: {scores.cpu().float().numpy()}")
                all_masks.append(masks)

        print(f"✓ Total detected: {total_objects} objects")

        # ===== Step 5: 格式转换到ComfyUI =====
        # 合并所有提示的所有实例masks
        if all_masks:
            # 拼接所有masks: list of [N_i, 1, H, W] -> [sum(N_i), 1, H, W]
            combined_masks = torch.cat(all_masks, dim=0)
            # 合并为一张: [sum(N_i), 1, H, W] -> [1, H, W]
            masks_squeezed = combined_masks.squeeze(1)  # [sum(N_i), H, W]
            merged_mask = torch.any(masks_squeezed, dim=0, keepdim=True)  # [1, H, W]
            masks_comfy = merged_mask.cpu().float()
        else:
            # 无检测结果，返回空mask
            masks_comfy = torch.zeros((1, H, W), dtype=torch.float32)

        print(f"✓ Output mask shape: {masks_comfy.shape}")

        # ===== Step 6: 生成masked image =====
        # mask: [1, H, W], image: [1, H, W, C]
        # 将mask扩展到与image相同的维度进行相乘
        mask_expanded = masks_comfy.unsqueeze(-1)  # [1, H, W, 1]
        masked_image = image[0:1] * mask_expanded  # [1, H, W, C]

        print(f"✓ Output image shape: {masked_image.shape}")

        # ===== Step 7: 内存管理 =====
        if not keep_model_loaded:
            processor.reset_all_prompts(inference_state)
            inference_state.clear()
            mm.soft_empty_cache()

        return (masks_comfy, masked_image)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadSAM3Model": DownloadAndLoadSAM3Model,
    "Sam3Segmentation": Sam3Segmentation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadSAM3Model": "(Down)Load SAM3 Model",
    "Sam3Segmentation": "SAM3 Segmentation",
}
