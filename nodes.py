import os

import numpy as np
import torch
from easydict import EasyDict
from huggingface_hub import hf_hub_download

from .color_fix import adain_color_fix
from .inference_utils import collate_fn, preprocess, tensor2vid
from .video_to_video.utils.logger import get_logger
from .video_to_video.utils.seed import setup_seed
from .video_to_video.video_to_video_model import VideoToVideo_sr

logger = get_logger()


class STARVSRNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model": (["Light Degradation", "Heavy Degradation"],),
                "precision": (
                    ["fp16", "fp8"],
                    {"default": "fp16"},
                ),
                "prompt": ("STRING", {"default": "a good video", "multiline": True}),
                "resolution": (
                    "INT",
                    {"default": 720, "min": 16, "max": 16384, "step": 2},
                ),
                "max_chunk_len": (
                    "INT",
                    {"default": 32, "min": 1, "max": 128, "step": 1},
                ),
                "cfg": (
                    "FLOAT",
                    {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "sampler": (
                    ["heun", "dpmpp_2m_sde"],
                    {"default": "dpmpp_2m_sde"},
                ),
                "solver_mode": (
                    ["fast", "normal"],
                    {"default": "fast"},
                ),
                "steps": ("INT", {"default": 15, "min": 1, "max": 100, "step": 1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance_video"
    CATEGORY = "video/upscaling"

    def __init__(self):
        self.model = None
        self.current_model_type = None
        self.current_precision = None

    def load_model(self, model_type, precision="fp16"):
        """Load the appropriate model based on degradation type and precision"""
        if (
            self.model is not None
            and self.current_model_type == model_type
            and self.current_precision == precision
        ):
            return self.model

        # Map model type to Hugging Face repo files
        hf_model_config = {
            "Light Degradation": {
                "repo_id": "SherryX/STAR",
                "filename": "I2VGen-XL-based/light_deg.pt",
                "local_path": "./models/STAR/I2VGen-XL-based/light_deg.pt",
            },
            "Heavy Degradation": {
                "repo_id": "SherryX/STAR",
                "filename": "I2VGen-XL-based/heavy_deg.pt",
                "local_path": "./models/STAR/I2VGen-XL-based/heavy_deg.pt",
            },
        }

        config = hf_model_config.get(model_type)
        if config is None:
            raise ValueError(f"Unknown model type: {model_type}")

        model_path = config["local_path"]

        # Download from Hugging Face if not exists locally
        if not os.path.exists(model_path):
            logger.info(
                f"Model not found locally. Downloading from Hugging Face: {config['repo_id']}/{config['filename']}"
            )
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Download the model from Hugging Face Hub
            downloaded_path = hf_hub_download(
                repo_id=config["repo_id"],
                filename=config["filename"],
                local_dir="./models/STAR/",
                local_dir_use_symlinks=False,
            )
            logger.info(f"Model downloaded to: {downloaded_path}")
            model_path = downloaded_path

        logger.info(f"Loading model: {model_path}")

        model_cfg = EasyDict(__name__="model_cfg")
        model_cfg.model_path = model_path

        # Pass precision parameter to VideoToVideo_sr
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = VideoToVideo_sr(model_cfg, device=device, precision=precision)
        self.current_model_type = model_type
        self.current_precision = precision

        return self.model

    def enhance_video(
        self,
        images,
        model,
        precision,
        prompt,
        resolution,
        max_chunk_len,
        cfg,
        sampler,
        solver_mode,
        steps,
        seed,
    ):
        """
        Enhance video frames using STAR model

        Args:
            images: Tensor of shape [B, H, W, C] in ComfyUI format (0-1 range)
            model: Model type ("Light Degradation" or "Heavy Degradation")
            precision: Model precision ("fp16", "fp8")
            prompt: Text prompt for enhancement
            resolution: Target resolution for the shorter side (maintains aspect ratio)
            max_chunk_len: Maximum chunk length for processing
            cfg: Guidance scale
            sampler: Sampling method ("heun" or "dpmpp_2m_sde")
            solver_mode: Solver mode ("fast" or "normal")
            steps: Number of denoising steps
            seed: Random seed for reproducibility

        Returns:
            Enhanced images tensor in ComfyUI format [B, H, W, C]
        """
        # Load the appropriate model
        model_instance = self.load_model(model, precision)

        # Convert ComfyUI format [B, H, W, C] to our format
        # ComfyUI images are in range [0, 1], convert to [0, 255] for preprocessing
        input_frames = []
        for i in range(images.shape[0]):
            frame = (images[i].cpu().numpy() * 255).astype(np.uint8)
            # Convert RGB to BGR for processing (matching cv2 format)
            frame_bgr = frame[:, :, ::-1]
            input_frames.append(frame_bgr)

        # Preprocess frames
        video_data = preprocess(input_frames)
        _, _, h, w = video_data.shape
        logger.info(f"Input resolution: {(h, w)}")

        # Calculate target resolution maintaining aspect ratio
        # Use resolution parameter as the target for the shorter side
        aspect_ratio = w / h

        if h < w:  # Height is shorter
            target_h = resolution
            target_w = int(resolution * aspect_ratio)
        else:  # Width is shorter or equal
            target_w = resolution
            target_h = int(resolution / aspect_ratio)

        # Ensure dimensions are even (required by model)
        target_h = target_h + (target_h % 2)
        target_w = target_w + (target_w % 2)

        logger.info(f"Target resolution: {(target_h, target_w)}")

        # Prepare caption
        caption = prompt or model_instance.positive_prompt
        logger.info(f"Caption: {caption}")

        # Prepare data
        pre_data = {
            "video_data": video_data,
            "y": caption,
            "target_res": (target_h, target_w),
        }

        total_noise_levels = 900
        setup_seed(seed)

        # Run inference
        with torch.no_grad():
            data_tensor = collate_fn(
                pre_data, "cuda:0" if torch.cuda.is_available() else "cpu"
            )
            output = model_instance.test(
                data_tensor,
                total_noise_levels,
                steps=steps,
                solver=sampler,
                solver_mode=solver_mode,
                guide_scale=cfg,
                max_chunk_len=max_chunk_len,
            )

        # Convert output to video frames
        output = tensor2vid(output)

        # Apply color fix
        output = adain_color_fix(output, video_data)

        # Convert back to ComfyUI format [B, H, W, C] in range [0, 1]
        # output is currently [T, H, W, C] in range [0, 255]
        output_normalized = output / 255.0

        # Ensure it's a torch tensor and move to CPU
        if not isinstance(output_normalized, torch.Tensor):
            output_normalized = torch.tensor(output_normalized)
        output_normalized = output_normalized.cpu().float()

        return (output_normalized,)


NODE_CLASS_MAPPINGS = {"STARVSRNode": STARVSRNode}

NODE_DISPLAY_NAME_MAPPINGS = {"STARVSRNode": "STAR Video Super Resolution"}
