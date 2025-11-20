# SAM3 ComfyUI Nodes

ComfyUI integration for Meta's [SAM 3 (Segment Anything with Concepts)](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/). SAM 3 enables **open-vocabulary text-based segmentation** using natural language prompts like "person", "red car", or "shoe".

---

## Nodes

### 1. DownloadAndLoadSAM3Model

Loads the SAM3 model.

**Inputs**:
- `device`: `cuda` (default) or `cpu`
- `precision`: `bf16` (default), `fp16`, or `fp32` (required for CPU)
- `confidence_threshold`: Detection threshold (0.0-1.0, default: 0.5)

**Output**:
- `sam3_model`: SAM3 model object

**Notes**:
- Model auto-downloads from Hugging Face if not found locally
- Requires: `huggingface-cli login` and accepting terms at https://huggingface.co/facebook/sam3

---

### 2. Sam3Segmentation

Segments objects using text prompts.

**Inputs**:
- `sam3_model`: Model from loader
- `image`: Input image (ComfyUI IMAGE format)
- `text_prompt`: Text description
  - Single: `"person"`, `"shoe"`, `"car"`
  - Multiple: `"person,shoe,car"` (comma-separated, auto-merged)
  - Descriptive: `"red car"`, `"person wearing blue"`
- `keep_model_loaded`: Keep model in VRAM (default: False)

**Outputs**:
- `mask`: Segmentation mask `[1, H, W]` float32, all instances merged
- `image`: Original image masked by the mask, background black `[1, H, W, C]` float32

**Key Features**:
- **Multi-class merging**: `"person,shoe"` runs inference for each class and merges results
- **Instance merging**: All detected instances (e.g., multiple people) merged into one mask
- **Masked image output**: Direct output of masked original image

---

## Model Weights

**Local Path**:
```
/root/ComfyUI/models/sam3/sam3.pt
```

**Auto-download**:
If model file doesn't exist, the node downloads from Hugging Face automatically:
```bash
# Login required
huggingface-cli login

# Accept terms at
https://huggingface.co/facebook/sam3
```

**Model Specs**:
- Filename: `sam3.pt`
- Size: ~850 MB
- Parameters: 848M

**BPE Tokenizer** (included):
```
ComfyUI-segment-anything-3/sam3/assets/bpe_simple_vocab_16e6.txt.gz
```

---

## Example Workflow

```
LoadImage → Sam3Segmentation → PreviewImage (mask)
              ↑                    ↓
    (Down)Load SAM3 Model    PreviewImage (image)
```

**Steps**:
1. Add `DownloadAndLoadSAM3Model`
   - device: `cuda`, precision: `bf16`, confidence_threshold: `0.5`
2. Add `LoadImage`
3. Add `Sam3Segmentation`
   - Connect `sam3_model` to loader
   - Connect `image` to LoadImage
   - Set `text_prompt`: `"person"`
4. Add `PreviewImage` for mask and image outputs

---

## Text Prompt Examples

```python
# Single class
"person"                    # Segment all people
"shoe"                      # Segment all shoes
"car"                       # Segment all cars

# Multiple classes (comma-separated)
"person,shoe"               # People and shoes
"person,car,dog"            # People, cars, and dogs
"head,hand,foot"            # Heads, hands, and feet

# Descriptive attributes
"red car"                   # Red cars only
"person wearing blue"       # People wearing blue
"large dog"                 # Large dogs
```

---

## Technical Details

### Inference Pipeline (Strictly Aligned with SAM3 Official)

```python
# 1. Enable autocast
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

# 2. Load model
model = build_sam3_image_model(bpe_path, checkpoint_path, device="cuda")
processor = Sam3Processor(model, confidence_threshold=0.5)

# 3. Set image
inference_state = processor.set_image(pil_image)

# 4. Text prompt inference
processor.reset_all_prompts(inference_state)
inference_state = processor.set_text_prompt(state=inference_state, prompt="person")

# 5. Extract results
masks = inference_state["masks"]    # [N, 1, H, W] bool
boxes = inference_state["boxes"]    # [N, 4] xyxy
scores = inference_state["scores"]  # [N,] confidence

# 6. Merge to ComfyUI format
merged_mask = torch.any(masks.squeeze(1), dim=0, keepdim=True).float()  # [1, H, W]
```

### Image Format Conversion

| Format | Shape | Dtype | Range |
|--------|-------|-------|-------|
| ComfyUI input | `[B, H, W, C]` | float32 | [0, 1] |
| PIL Image | `(W, H)` | uint8 | [0, 255] |
| SAM3 output | `[N, 1, H, W]` | bool | {0, 1} |
| ComfyUI mask | `[1, H, W]` | float32 | [0, 1] |
| ComfyUI image | `[1, H, W, C]` | float32 | [0, 1] |

---

## Testing

### Standalone Test (Outside ComfyUI)

```bash
cd /root/ComfyUI/custom_nodes/ComfyUI-segment-anything-3
python test_sam3_standalone.py
```

**Output**:
- `test_sam3_output.png` - shoe segmentation
- `test_sam3_output_head.png` - head segmentation

### ComfyUI Integration Test

1. Start ComfyUI:
```bash
cd /root/ComfyUI
python main.py
```

2. Right-click → Add Node → SAM3
3. You should see 2 nodes:
   - `(Down)Load SAM3 Model`
   - `SAM3 Segmentation`

---

## Troubleshooting

### Q: Model download failed

**A**: Login to Hugging Face and accept terms
```bash
huggingface-cli login
# Visit https://huggingface.co/facebook/sam3 and accept terms
```

### Q: CPU error with bf16/fp16

**A**: CPU only supports fp32. Change `precision` to `fp32`

### Q: No objects detected

**A**:
1. Lower `confidence_threshold` (e.g., 0.3)
2. Try different text descriptions
3. Verify object exists in image

### Q: CUDA out of memory

**A**:
1. Set `keep_model_loaded=False`
2. Use `fp16` precision
3. Use smaller images

---

## File Structure

```
ComfyUI-segment-anything-3/
├── __init__.py                   # Exports SAM2+SAM3 nodes
├── sam3/                         # SAM3 official code
├── sam3_nodes/                   # SAM3 ComfyUI nodes
│   ├── __init__.py
│   ├── nodes.py                  # 2 node implementations
│   └── load_model.py             # Model loading wrapper
├── test_sam3_standalone.py       # Standalone test script
├── README_SAM3.md               # This file (English)
└── README_SAM3_CN.md            # Chinese version
```

---

## References

- SAM3 Paper: https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/
- SAM3 Official Code: https://github.com/facebookresearch/sam3
- Hugging Face Model: https://huggingface.co/facebook/sam3
