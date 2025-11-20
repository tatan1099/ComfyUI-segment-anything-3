# SAM3 ComfyUI 节点

Meta 的 [SAM 3 (Segment Anything with Concepts)](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/) 的 ComfyUI 集成。SAM 3 支持使用自然语言提示（如 "person"、"red car"、"shoe"）进行**开放词汇文本分割**。

---

## 节点说明

### 1. DownloadAndLoadSAM3Model

加载 SAM3 模型。

**输入参数**:
- `device`: `cuda`（默认）或 `cpu`
- `precision`: `bf16`（默认）、`fp16` 或 `fp32`（CPU 必须使用）
- `confidence_threshold`: 检测阈值（0.0-1.0，默认 0.5）

**输出**:
- `sam3_model`: SAM3 模型对象

**注意事项**:
- 本地不存在时自动从 Hugging Face 下载
- 需要：`huggingface-cli login` 并在 https://huggingface.co/facebook/sam3 接受使用条款

---

### 2. Sam3Segmentation

使用文本提示分割对象。

**输入参数**:
- `sam3_model`: 来自加载器的模型
- `image`: 输入图像（ComfyUI IMAGE 格式）
- `text_prompt`: 文本描述
  - 单个类别：`"person"`、`"shoe"`、`"car"`
  - 多个类别：`"person,shoe,car"`（逗号分隔，自动合并）
  - 带属性：`"red car"`、`"person wearing blue"`
- `keep_model_loaded`: 保持模型在显存中（默认 False）

**输出**:
- `mask`: 分割掩码 `[1, H, W]` float32，所有实例已合并
- `image`: mask 区域的原图，背景黑色 `[1, H, W, C]` float32

**核心特性**:
- **多类别合并**：`"person,shoe"` 会分别推理并合并结果
- **实例合并**：所有检测到的实例（如多个人）合并为一张 mask
- **遮罩图像输出**：直接输出 mask 区域的原始图像

---

## 模型权重

**本地路径**:
```
/root/ComfyUI/models/sam3/sam3.pt
```

**自动下载**:
如果模型文件不存在，节点会自动从 Hugging Face 下载：
```bash
# 需要登录
huggingface-cli login

# 访问以下页面接受条款
https://huggingface.co/facebook/sam3
```

**模型规格**:
- 文件名：`sam3.pt`
- 大小：约 850 MB
- 参数量：848M

**BPE Tokenizer**（已包含）:
```
ComfyUI-segment-anything-3/sam3/assets/bpe_simple_vocab_16e6.txt.gz
```

---

## 使用示例

```
LoadImage → Sam3Segmentation → PreviewImage (mask)
              ↑                    ↓
    (Down)Load SAM3 Model    PreviewImage (image)
```

**步骤**:
1. 添加 `DownloadAndLoadSAM3Model`
   - device: `cuda`、precision: `bf16`、confidence_threshold: `0.5`
2. 添加 `LoadImage`
3. 添加 `Sam3Segmentation`
   - 连接 `sam3_model` 到加载器
   - 连接 `image` 到 LoadImage
   - 设置 `text_prompt`: `"person"`
4. 添加 `PreviewImage` 查看 mask 和 image 输出

---

## 文本提示示例

```python
# 单个类别
"person"                    # 分割所有人
"shoe"                      # 分割所有鞋子
"car"                       # 分割所有汽车

# 多个类别（逗号分隔）
"person,shoe"               # 人和鞋子
"person,car,dog"            # 人、车、狗
"head,hand,foot"            # 头、手、脚

# 带属性描述
"red car"                   # 红色汽车
"person wearing blue"       # 穿蓝色衣服的人
"large dog"                 # 大型犬
```

---

## 技术细节

### 推理流程（严格对齐 SAM3 官方）

```python
# 1. 启用 autocast
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

# 2. 加载模型
model = build_sam3_image_model(bpe_path, checkpoint_path, device="cuda")
processor = Sam3Processor(model, confidence_threshold=0.5)

# 3. 设置图像
inference_state = processor.set_image(pil_image)

# 4. 文本提示推理
processor.reset_all_prompts(inference_state)
inference_state = processor.set_text_prompt(state=inference_state, prompt="person")

# 5. 提取结果
masks = inference_state["masks"]    # [N, 1, H, W] bool
boxes = inference_state["boxes"]    # [N, 4] xyxy
scores = inference_state["scores"]  # [N,] confidence

# 6. 转换为 ComfyUI 格式
merged_mask = torch.any(masks.squeeze(1), dim=0, keepdim=True).float()  # [1, H, W]
```

### 图像格式转换

| 格式 | Shape | Dtype | Range |
|------|-------|-------|-------|
| ComfyUI 输入 | `[B, H, W, C]` | float32 | [0, 1] |
| PIL Image | `(W, H)` | uint8 | [0, 255] |
| SAM3 输出 | `[N, 1, H, W]` | bool | {0, 1} |
| ComfyUI mask | `[1, H, W]` | float32 | [0, 1] |
| ComfyUI image | `[1, H, W, C]` | float32 | [0, 1] |

---

## 测试

### 独立测试（ComfyUI 外）

```bash
cd /root/ComfyUI/custom_nodes/ComfyUI-segment-anything-3
python test_sam3_standalone.py
```

**输出**:
- `test_sam3_output.png` - shoe 分割结果
- `test_sam3_output_head.png` - head 分割结果

### ComfyUI 集成测试

1. 启动 ComfyUI：
```bash
cd /root/ComfyUI
python main.py
```

2. 右键 → Add Node → SAM3
3. 应该看到 2 个节点：
   - `(Down)Load SAM3 Model`
   - `SAM3 Segmentation`

---

## 常见问题

### Q: 模型下载失败

**A**: 登录 Hugging Face 并接受条款
```bash
huggingface-cli login
# 访问 https://huggingface.co/facebook/sam3 接受条款
```

### Q: CPU 运行 bf16/fp16 报错

**A**: CPU 只支持 fp32，将 `precision` 改为 `fp32`

### Q: 检测不到对象

**A**:
1. 降低 `confidence_threshold`（如 0.3）
2. 尝试不同的文本描述
3. 确认图像中确实包含该对象

### Q: CUDA 显存不足

**A**:
1. 设置 `keep_model_loaded=False`
2. 使用 `fp16` 精度
3. 使用更小的图像

---

## 文件结构

```
ComfyUI-segment-anything-3/
├── __init__.py                   # 导出 SAM2+SAM3 节点
├── sam3/                         # SAM3 官方代码
├── sam3_nodes/                   # SAM3 ComfyUI 节点
│   ├── __init__.py
│   ├── nodes.py                  # 2 个节点实现
│   └── load_model.py             # 模型加载包装
├── test_sam3_standalone.py       # 独立测试脚本
├── README_SAM3.md               # 英文版
└── README_SAM3_CN.md            # 本文件（中文版）
```

---

## 参考

- SAM3 论文：https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/
- SAM3 官方代码：https://github.com/facebookresearch/sam3
- Hugging Face 模型：https://huggingface.co/facebook/sam3
