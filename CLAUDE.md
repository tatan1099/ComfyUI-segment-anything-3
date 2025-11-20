# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a ComfyUI custom node package that integrates both SAM 2 (Segment Anything Model 2) and SAM 3 (Segment Anything with Concepts) from Meta AI. The repository contains two major components:

1. **ComfyUI-segment-anything-2**: SAM 2 integration for ComfyUI with point, box, and video segmentation
2. **sam3**: The full SAM 3 research codebase with text-based concept segmentation

SAM 2 provides interactive segmentation using visual prompts (points, boxes, masks), while SAM 3 adds the ability to segment objects using text prompts and open-vocabulary concepts.

## Directory Structure

```
ComfyUI-segment-anything-3/
├── ComfyUI-segment-anything-2/  # SAM 2 ComfyUI integration
│   ├── __init__.py              # Node registration entry point
│   ├── nodes.py                 # All SAM 2 ComfyUI nodes
│   ├── load_model.py            # SAM 2 model loading logic
│   ├── sam2/                    # SAM 2 model implementation
│   └── sam2_configs/            # Model architecture configs (.yaml)
└── sam3/                        # Full SAM 3 research codebase
    ├── sam3/                    # Core SAM 3 implementation
    │   ├── model/               # Model architecture
    │   ├── agent/               # SAM 3 agent for complex prompts
    │   ├── eval/                # Evaluation toolkits
    │   └── train/               # Training infrastructure
    └── examples/                # Jupyter notebook examples
```

## Architecture

### SAM 2 Integration (ComfyUI-segment-anything-2/)

**Entry Point**: `ComfyUI-segment-anything-2/__init__.py`
- Exports `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` for ComfyUI

**Core Nodes** (`nodes.py`):
- `DownloadAndLoadSAM2Model`: Downloads and loads SAM 2 models from HuggingFace
- `Sam2Segmentation`: Main segmentation node supporting points, boxes, and masks
- `Florence2toCoordinates`: Converts Florence2 detection results to coordinates
- `Sam2AutoSegmentation`: Automatic mask generation for entire images
- `Sam2VideoSegmentationAddPoints`: Interactive video segmentation with point prompts
- `Sam2VideoSegmentation`: Propagates segmentation masks across video frames

**Model Loading** (`load_model.py`):
- Manually constructs SAM 2 architecture from YAML configs
- Loads weights from safetensors format
- Creates three segmentor types: `single_image`, `video`, `automaskgenerator`
- Uses component-based assembly: ImageEncoder → SAM2Base → Predictor wrapper

**Key Model Components**:
- `SAM2Base`: Core model with image encoder, memory attention, and memory encoder
- `SAM2ImagePredictor`: Wrapper for single-image inference
- `SAM2VideoPredictor`: Wrapper for video inference with temporal memory
- `SAM2AutomaticMaskGenerator`: Generates all masks in an image automatically

### SAM 3 Integration (sam3/)

**Model Builder** (`sam3/model_builder.py`):
- `build_sam3_image_model()`: Creates SAM 3 for image segmentation with text/visual prompts
- `build_sam3_video_model()`: Creates SAM 3 for video tracking with text prompts
- `build_sam3_video_predictor()`: Multi-GPU video predictor wrapper
- `build_tracker()`: Creates the tracker module for temporal consistency

**Key Components**:
- **Vision Backbone**: ViT-based visual encoder with dual-neck architecture
- **Text Encoder**: VETextEncoder for processing text prompts and concepts
- **Transformer**: Encoder-decoder with cross-attention between vision and text
- **Segmentation Head**: Pixel decoder and universal segmentation head
- **Geometry Encoder**: Encodes visual prompts (points, boxes) as sequences
- **Tracker**: SAM 3 tracker for video temporal consistency

**Model Architecture**:
1. `Sam3Image`: Single-frame detector with text and visual prompts
2. `Sam3TrackerPredictor`: Video tracker with memory-based tracking
3. `Sam3VideoInferenceWithInstanceInteractivity`: Combined detector-tracker for video

### Data Flow

**SAM 2 Single Image**:
1. User loads model via `DownloadAndLoadSAM2Model` (downloads from HuggingFace if needed)
2. Model stored in `ComfyUI/models/sam2/`
3. `Sam2Segmentation` takes image + coordinates/boxes
4. Model encodes image → processes prompts → generates masks
5. Returns mask tensor compatible with ComfyUI mask format

**SAM 2 Video**:
1. Initialize inference state with video frames
2. Add points/boxes on frame 0 with `add_new_points_or_box()`
3. Propagate masks across all frames with `propagate_in_video()`
4. Memory attention maintains consistency across frames

**SAM 3 Image** (from research code):
1. Build model with `build_sam3_image_model()`
2. Set image with processor: `processor.set_image(image)`
3. Set text prompt: `processor.set_text_prompt(prompt="concept")`
4. Returns masks, boxes, and scores for all instances of the concept

**SAM 3 Video** (from research code):
1. Build predictor with `build_sam3_video_predictor()`
2. Start session with video path
3. Add text/visual prompts on keyframes
4. Detector finds instances, tracker maintains temporal identity
5. Returns per-frame masks with consistent instance IDs

## Model Files and Downloads

**SAM 2 Models** (auto-downloaded to `ComfyUI/models/sam2/`):
- Repository: `Kijai/sam2-safetensors` on HuggingFace
- Versions: 2.0 and 2.1
- Sizes: tiny, small, base_plus, large
- Format: `.safetensors` (safe tensor format)
- Config files in `sam2_configs/`: Map model size to architecture YAML

**SAM 3 Models** (auto-downloaded via HuggingFace Hub):
- Repository: `facebook/sam3` on HuggingFace
- Checkpoint: `sam3.pt` (848M parameters)
- Requires authentication: `huggingface-cli login` (see SAM 3 README)
- BPE tokenizer vocab: `assets/bpe_simple_vocab_16e6.txt.gz`

## Development

### Running SAM 3 Examples

```bash
# Install notebook dependencies
cd sam3
pip install -e ".[notebooks]"

# Run examples
jupyter notebook examples/sam3_image_predictor_example.ipynb
jupyter notebook examples/sam3_video_predictor_example.ipynb
```

### Testing SAM 3

```bash
cd sam3
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
ufmt format .
```

### Training SAM 3

```bash
cd sam3
pip install -e ".[train]"

# Local training
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 0 --num-gpus 1

# Cluster training
python sam3/train/train.py -c configs/odinw13/odinw_text_only_train.yaml --use-cluster 1 --num-gpus 8 --num-nodes 2
```

## Important Notes

- **Two Separate Codebases**: SAM 2 and SAM 3 are distinct. SAM 2 nodes are in `ComfyUI-segment-anything-2/`, SAM 3 is in `sam3/`
- **Model Compatibility**: SAM 2 models use different architecture than SAM 3. They are not interchangeable.
- **HuggingFace Auth**: SAM 3 requires accepting terms and authenticating with HuggingFace
- **Video Memory**: SAM 2 video segmentation maintains inference state. Use `keep_model_loaded=False` to free memory
- **Coordinate Format**: Points are JSON strings: `[{"x": 100, "y": 200}]`. Labels: 1=positive, 0=negative
- **Model Precision**: SAM 2 supports fp16/bf16/fp32. fp16/bf16 require CUDA (not CPU/MPS)
- **Video Model Versions**: SAM 2.0 video doesn't support bbox prompts; use SAM 2.1 for bbox support
- **Automatic Masking**: Use `Sam2AutoSegmentation` node with `automaskgenerator` segmentor type

## Custom Node Development

When adding new SAM 2 nodes to `nodes.py`:
1. Add class definition with `INPUT_TYPES()`, `RETURN_TYPES`, `FUNCTION`, `CATEGORY`
2. Register in `NODE_CLASS_MAPPINGS` dict
3. Register display name in `NODE_DISPLAY_NAME_MAPPINGS` dict
4. Use `comfy.model_management` for device management
5. Use `comfy.utils.ProgressBar` for progress feedback

## Troubleshooting

- **Model download fails**: Check internet connection and HuggingFace hub access
- **CUDA out of memory**: Reduce batch size, use smaller model, or enable `keep_model_loaded=False`
- **fp16 on CPU error**: SAM 2 node validates and raises error; use fp32 for CPU
- **Video segmentation state issues**: Call `model.reset_state(inference_state)` to clear
- **SAM 3 checkpoint access denied**: Accept terms on HuggingFace model page and run `huggingface-cli login`
