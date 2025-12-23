# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

STARVSRWrapper is a ComfyUI custom node that wraps the STAR (Spatio-Temporal Adaptive Restoration) video super-resolution model. It enables video upscaling with light or heavy degradation models through a ComfyUI node interface.

## Architecture

### Core Components

1. **[nodes.py](nodes.py)** - Main ComfyUI node implementation (`STARVSRNode`)
   - Implements ComfyUI's node interface pattern (INPUT_TYPES, RETURN_TYPES, FUNCTION)
   - Handles model loading from Hugging Face Hub or local cache
   - Converts between ComfyUI image format [B, H, W, C] (0-1 range) and STAR's internal format
   - Two model variants: "Light Degradation" and "Heavy Degradation" downloaded from HF repo `SherryX/STAR`

2. **[inference_utils.py](inference_utils.py)** - Preprocessing and data utilities
   - `preprocess()`: Converts input frames to normalized tensors (mean=0.5, std=0.5)
   - `tensor2vid()`: Converts output tensors back to video frames
   - `collate_fn()`: Recursive device placement for nested data structures
   - `load_video()` / `save_video()`: File I/O utilities (uses ffmpeg for encoding)

3. **[color_fix.py](color_fix.py)** - Post-processing color correction
   - `adain_color_fix()`: Adaptive Instance Normalization to match source video colors
   - `wavelet_color_fix()`: Alternative wavelet-based color correction
   - Currently uses AdaIN by default in the node pipeline

4. **video_to_video/** - STAR model implementation
   - `video_to_video_model.py`: Main `VideoToVideo_sr` class that orchestrates inference
   - `diffusion/`: Diffusion sampling and noise schedules (based on SDEdit)
   - `modules/`: UNet architecture (`ControlledV2VUNet`) and text embedder (OpenCLIP)
   - `utils/`: Configuration, logging, and seed utilities

### Data Flow

```
ComfyUI Images [B,H,W,C] (0-1)
    ↓ (convert to BGR, scale to 0-255)
Input frames (numpy arrays)
    ↓ preprocess()
Normalized tensors (mean=0.5, std=0.5)
    ↓ VAE encode
Latent space features
    ↓ Diffusion sampling (with ControlNet guidance)
Denoised latents
    ↓ VAE decode
Output tensors
    ↓ tensor2vid() + adain_color_fix()
Video frames [T,H,W,C] (0-255)
    ↓ (normalize to 0-1)
ComfyUI Images [B,H,W,C] (0-1)
```

### Model Components

- **Text Encoder**: FrozenOpenCLIPEmbedder (laion2b_s32b_b79k) for caption conditioning
- **UNet**: `ControlledV2VUNet` - temporal-aware UNet with ControlNet for LR guidance
- **VAE**: `AutoencoderKLTemporalDecoder` from Stable Video Diffusion
- **Diffusion**: GaussianDiffusion with logsnr_cosine_interp schedule, supports heun and dpmpp_2m_sde samplers
- **Chunking**: Long videos are processed in overlapping chunks (default max_chunk_len=32)

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Code Formatting
```bash
# Format code using ruff
ruff format .

# Check formatting without making changes
ruff format --check .
```

### Important Notes

- **CUDA Requirement**: Model runs on GPU (cuda:0). Hardcoded in several places.
- **Model Caching**: Models are downloaded to `./pretrained_weight/` by Hugging Face Hub and expected at `./models/STAR/` for local loading
- **Resolution Constraints**: Model pads inputs to fit minimum 720x1280 or multiples of 64 pixels
- **Memory Management**: Uses chunked processing and explicit `torch.cuda.empty_cache()` calls
- **FP16**: Model runs in half precision for efficiency
- **Color Space**: Internal processing uses BGR (cv2 format), but ComfyUI expects RGB

## Integration with ComfyUI

This is a ComfyUI custom node that should be placed in ComfyUI's `custom_nodes/` directory. The node:
- Category: "video/upscaling"
- Display name: "STAR Video Super Resolution"
- Inputs: IMAGE tensor, model type, prompt, upscale factor, sampling parameters
- Outputs: Upscaled IMAGE tensor

The `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` in [__init__.py](__init__.py) register the node with ComfyUI.
