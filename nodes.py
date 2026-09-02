from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import torch
import torch.nn.functional as F
import comfy.model_management as model_management
from comfy_api.latest import io
from easydict import EasyDict
from huggingface_hub import hf_hub_download

from .color_fix import adain_color_fix, wavelet_color_fix
from .inference_utils import tensor2vid
from .video_to_video.utils.logger import get_logger
from .video_to_video.utils.seed import setup_seed
from .video_to_video.video_to_video_model import VideoToVideo_sr

try:
    import folder_paths
except ImportError:  # Allows lightweight tooling outside a ComfyUI checkout.
    folder_paths = None


logger = get_logger()

CATEGORY = "video/upscaling/STAR"
TOTAL_NOISE_LEVELS = 900

MODEL_SPECS = {
    "Light Degradation": {
        "repo_id": "SherryX/STAR",
        "filename": "I2VGen-XL-based/light_deg.pt",
    },
    "Heavy Degradation": {
        "repo_id": "SherryX/STAR",
        "filename": "I2VGen-XL-based/heavy_deg.pt",
    },
}
MODEL_OPTIONS = list(MODEL_SPECS)
PRECISION_OPTIONS = ["fp16", "fp8"]
SAMPLER_OPTIONS = ["heun", "dpmpp_2m_sde"]
SOLVER_MODE_OPTIONS = ["fast", "normal"]
COLOR_FIX_OPTIONS = ["AdaIN", "wavelet", "none"]

STAR_MODEL = io.Custom("STARVSR_MODEL")
STAR_VIDEO = io.Custom("STARVSR_VIDEO")
STAR_CONDITIONING = io.Custom("STARVSR_CONDITIONING")
STAR_LATENT = io.Custom("STARVSR_LATENT")


@dataclass(frozen=True, slots=True)
class STARModelHandle:
    model: VideoToVideo_sr
    model_type: str
    precision: str
    checkpoint_path: str


@dataclass(frozen=True, slots=True)
class STARPreparedVideo:
    video_data: torch.Tensor
    target_res: tuple[int, int]


@dataclass(frozen=True, slots=True)
class STARConditioning:
    embeddings: torch.Tensor
    prompt: str


@dataclass(frozen=True, slots=True)
class STARSampledLatent:
    samples: torch.Tensor
    padding: tuple[int, int, int, int]
    output_size: tuple[int, int]
    batch_size: int


@dataclass(slots=True)
class _CachedModel:
    identity: tuple[str, int, int]
    handle: STARModelHandle


_MODEL_CACHE: dict[tuple[str, str], _CachedModel] = {}
_MODEL_CACHE_LOCK = Lock()


def _models_root() -> Path:
    if folder_paths is not None:
        return Path(folder_paths.models_dir)
    return Path.cwd() / "models"


def _checkpoint_path(model_type: str) -> Path:
    try:
        filename = MODEL_SPECS[model_type]["filename"]
    except KeyError as exc:
        raise ValueError(f"Unknown STAR model type: {model_type}") from exc
    return _models_root() / "STAR" / filename


def _checkpoint_identity(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path.resolve()), stat.st_mtime_ns, stat.st_size


def _checkpoint_fingerprint(model_type: str) -> tuple[str, int, int] | tuple[str, str]:
    path = _checkpoint_path(model_type)
    if path.is_file():
        return _checkpoint_identity(path)
    spec = MODEL_SPECS[model_type]
    return "missing", f"{spec['repo_id']}:{spec['filename']}"


def _ensure_checkpoint(model_type: str) -> Path:
    path = _checkpoint_path(model_type)
    if path.is_file():
        return path

    spec = MODEL_SPECS[model_type]
    local_dir = _models_root() / "STAR"
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Model not found locally. Downloading from Hugging Face: "
        f"{spec['repo_id']}/{spec['filename']}"
    )
    downloaded_path = Path(
        hf_hub_download(
            repo_id=spec["repo_id"],
            filename=spec["filename"],
            local_dir=str(local_dir),
        )
    )
    logger.info(f"Model downloaded to: {downloaded_path}")
    return downloaded_path


def _load_model(model_type: str, precision: str) -> STARModelHandle:
    cache_key = model_type, precision
    with _MODEL_CACHE_LOCK:
        path = _ensure_checkpoint(model_type)
        identity = _checkpoint_identity(path)
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None and cached.identity == identity:
            return cached.handle

        logger.info(f"Loading model: {path}")
        model_cfg = EasyDict(__name__="model_cfg", model_path=str(path))
        device = model_management.get_torch_device()
        handle = STARModelHandle(
            model=VideoToVideo_sr(model_cfg, device=device, precision=precision),
            model_type=model_type,
            precision=precision,
            checkpoint_path=str(path),
        )

        # Match the old node's one-model cache. References held by ComfyUI's
        # output cache remain valid when a different model is loaded.
        _MODEL_CACHE.clear()
        _MODEL_CACHE[cache_key] = _CachedModel(identity=identity, handle=handle)
        return handle


def _require_type(value: Any, expected_type: type, input_name: str):
    if not isinstance(value, expected_type):
        raise TypeError(
            f"{input_name} must come from the matching STAR node; "
            f"received {type(value).__name__}."
        )
    return value


def _prepare_video(images: torch.Tensor, resolution: int) -> STARPreparedVideo:
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a ComfyUI IMAGE tensor.")
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(
            "STAR expects RGB IMAGE data in [frames, height, width, 3] format; "
            f"received {tuple(images.shape)}."
        )
    if images.shape[0] < 1 or images.shape[1] < 1 or images.shape[2] < 1:
        raise ValueError("STAR requires at least one non-empty video frame.")

    # Preserve the legacy wrapper's uint8 preprocessing while avoiding the
    # RGB -> BGR -> RGB and PIL round trips. Cache this tensor on the CPU so it
    # does not reserve VRAM between executions.
    video_data = (
        images.detach()
        .clamp(0.0, 1.0)
        .mul(255)
        .to(device="cpu", dtype=torch.uint8)
        .to(dtype=torch.float32)
        .div(255)
        .permute(0, 3, 1, 2)
        .contiguous()
        .mul(2)
        .sub(1)
    )

    _, _, height, width = video_data.shape
    aspect_ratio = width / height
    if height < width:
        target_h = resolution
        target_w = int(resolution * aspect_ratio)
    else:
        target_w = resolution
        target_h = int(resolution / aspect_ratio)

    target_h += target_h % 2
    target_w += target_w % 2
    logger.info(f"Input resolution: {(height, width)}")
    logger.info(f"Target resolution: {(target_h, target_w)}")
    return STARPreparedVideo(video_data=video_data, target_res=(target_h, target_w))


def _decode_video(
    model: STARModelHandle,
    latent: STARSampledLatent,
    vae_decode_chunk: int,
) -> torch.Tensor:
    video = model.model.decode_latent(
        latent.samples,
        latent.padding,
        latent.output_size,
        batch_size=latent.batch_size,
        chunk_size=vae_decode_chunk,
    )
    return tensor2vid(video).div(255).to(dtype=torch.float32, device="cpu")


def _color_fix(
    images: torch.Tensor,
    reference_images: torch.Tensor,
    method: str,
) -> torch.Tensor:
    if method == "none":
        return images
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("images must use ComfyUI's [frames, height, width, 3] format.")
    if reference_images.ndim != 4 or reference_images.shape[-1] != 3:
        raise ValueError(
            "reference_images must use ComfyUI's [frames, height, width, 3] format."
        )
    if images.shape[0] != reference_images.shape[0]:
        raise ValueError(
            "images and reference_images must contain the same number of frames."
        )

    original_device = images.device
    original_dtype = images.dtype
    target = images.detach().to(dtype=torch.float32).clamp(0.0, 1.0)
    source = (
        reference_images.detach()
        .to(device=target.device)
        .clamp(0.0, 1.0)
        .mul(255)
        .to(dtype=torch.uint8)
        .to(dtype=torch.float32)
        .div(255)
        .permute(0, 3, 1, 2)
        .contiguous()
        .mul(2)
        .sub(1)
    )

    if method == "AdaIN":
        result = adain_color_fix(target.mul(255), source).div(255)
    elif method == "wavelet":
        if source.shape[-2:] != target.shape[1:3]:
            source = F.interpolate(
                source, size=target.shape[1:3], mode="bilinear", align_corners=False
            )
        result = wavelet_color_fix(target.mul(255), source).div(255)
    else:
        raise ValueError(f"Unknown color-fix method: {method}")

    return result.to(device=original_device, dtype=original_dtype)


def _model_inputs() -> list[io.Input]:
    return [
        io.Combo.Input(
            "model",
            options=MODEL_OPTIONS,
            default="Light Degradation",
            tooltip="STAR checkpoint trained for the source degradation level.",
        ),
        io.Combo.Input(
            "precision",
            options=PRECISION_OPTIONS,
            default="fp16",
            tooltip="UNet weight precision. FP8 requires compatible PyTorch and hardware.",
        ),
    ]


def _sampling_inputs() -> list[io.Input]:
    return [
        io.Int.Input(
            "max_chunk_len",
            default=32,
            min=1,
            max=128,
            step=1,
            tooltip="Maximum temporal window processed by the diffusion model.",
        ),
        io.Float.Input(
            "cfg",
            default=7.5,
            min=0.0,
            max=20.0,
            step=0.1,
            tooltip="Classifier-free guidance strength.",
        ),
        io.Combo.Input(
            "sampler",
            options=SAMPLER_OPTIONS,
            default="dpmpp_2m_sde",
        ),
        io.Combo.Input(
            "solver_mode",
            options=SOLVER_MODE_OPTIONS,
            default="fast",
        ),
        io.Int.Input("steps", default=15, min=1, max=100, step=1),
        io.Int.Input(
            "seed",
            default=42,
            min=0,
            max=0xFFFFFFFFFFFFFFFF,
            control_after_generate=True,
        ),
    ]


def _vae_decode_chunk_input(*, optional: bool = False) -> io.Int.Input:
    return io.Int.Input(
        "vae_decode_chunk",
        default=1,
        min=1,
        max=8,
        step=1,
        optional=optional,
        tooltip=(
            "Frames decoded per VAE pass. One minimizes peak VRAM; increase only "
            "when enough VRAM is available. This value no longer invalidates "
            "diffusion sampling in the modular workflow."
        ),
    )


class STARModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="STARVSR_ModelLoader",
            display_name="STAR Model Loader",
            category=CATEGORY,
            description=(
                "Loads a STAR degradation checkpoint. Keep this separate so prompt and "
                "sampling changes reuse the loaded model through ComfyUI's cache."
            ),
            inputs=_model_inputs(),
            outputs=[STAR_MODEL.Output("star_model", display_name="STAR model")],
        )

    @classmethod
    def fingerprint_inputs(cls, model: str, precision: str, **_kwargs):
        return model, precision, _checkpoint_fingerprint(model)

    @classmethod
    def execute(cls, model: str, precision: str) -> io.NodeOutput:
        return io.NodeOutput(_load_model(model, precision))


class STARPrepareVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="STARVSR_PrepareVideo",
            display_name="STAR Prepare Video",
            category=CATEGORY,
            description=(
                "Quantizes and normalizes an IMAGE frame batch and calculates the target "
                "resolution. The cached output is CPU-resident."
            ),
            inputs=[
                io.Image.Input(
                    "images", tooltip="Video frames as a ComfyUI IMAGE batch."
                ),
                io.Int.Input(
                    "resolution",
                    default=720,
                    min=16,
                    max=16384,
                    step=2,
                    tooltip="Target size of the shorter side; aspect ratio is preserved.",
                ),
            ],
            outputs=[STAR_VIDEO.Output("video", display_name="prepared video")],
        )

    @classmethod
    def execute(cls, images: torch.Tensor, resolution: int) -> io.NodeOutput:
        return io.NodeOutput(_prepare_video(images, resolution))


class STARTextEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="STARVSR_TextEncode",
            display_name="STAR Text Encode",
            category=CATEGORY,
            description=(
                "Encodes the prompt once so sampler-setting changes can reuse the "
                "conditioning from ComfyUI's cache."
            ),
            inputs=[
                STAR_MODEL.Input("star_model"),
                io.String.Input("prompt", default="a good video", multiline=True),
            ],
            outputs=[
                STAR_CONDITIONING.Output(
                    "conditioning", display_name="STAR conditioning"
                )
            ],
        )

    @classmethod
    def execute(cls, star_model: STARModelHandle, prompt: str) -> io.NodeOutput:
        star_model = _require_type(star_model, STARModelHandle, "star_model")
        caption = prompt or star_model.model.positive_prompt
        logger.info(f"Caption: {caption}")
        embeddings = star_model.model.encode_prompt(caption)
        return io.NodeOutput(STARConditioning(embeddings=embeddings, prompt=caption))


class STARSample(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="STARVSR_Sample",
            display_name="STAR Sample",
            category=CATEGORY,
            description=(
                "VAE-encodes the prepared video and runs STAR diffusion. Returns a "
                "small CPU latent so decoding can be cached separately."
            ),
            inputs=[
                STAR_MODEL.Input("star_model"),
                STAR_VIDEO.Input("video"),
                STAR_CONDITIONING.Input("conditioning"),
                *_sampling_inputs(),
            ],
            outputs=[STAR_LATENT.Output("latent", display_name="STAR latent")],
        )

    @classmethod
    def execute(
        cls,
        star_model: STARModelHandle,
        video: STARPreparedVideo,
        conditioning: STARConditioning,
        max_chunk_len: int,
        cfg: float,
        sampler: str,
        solver_mode: str,
        steps: int,
        seed: int,
    ) -> io.NodeOutput:
        star_model = _require_type(star_model, STARModelHandle, "star_model")
        video = _require_type(video, STARPreparedVideo, "video")
        conditioning = _require_type(conditioning, STARConditioning, "conditioning")
        setup_seed(seed)
        samples, padding, output_size, batch_size = star_model.model.sample_latent(
            {"video_data": video.video_data, "target_res": video.target_res},
            conditioning.embeddings,
            total_noise_levels=TOTAL_NOISE_LEVELS,
            steps=steps,
            solver=sampler,
            solver_mode=solver_mode,
            guide_scale=cfg,
            max_chunk_len=max_chunk_len,
        )
        return io.NodeOutput(
            STARSampledLatent(
                samples=samples,
                padding=padding,
                output_size=output_size,
                batch_size=batch_size,
            )
        )


class STARDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="STARVSR_Decode",
            display_name="STAR VAE Decode",
            category=CATEGORY,
            description=(
                "Decodes a STAR latent to raw enhanced frames. Changing decode chunk "
                "size reruns only this node, not diffusion sampling."
            ),
            inputs=[
                STAR_MODEL.Input("star_model"),
                STAR_LATENT.Input("latent"),
                _vae_decode_chunk_input(),
            ],
            outputs=[io.Image.Output("images", display_name="raw images")],
        )

    @classmethod
    def execute(
        cls,
        star_model: STARModelHandle,
        latent: STARSampledLatent,
        vae_decode_chunk: int,
    ) -> io.NodeOutput:
        star_model = _require_type(star_model, STARModelHandle, "star_model")
        latent = _require_type(latent, STARSampledLatent, "latent")
        return io.NodeOutput(_decode_video(star_model, latent, vae_decode_chunk))


class STARColorFix(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="STARVSR_ColorFix",
            display_name="STAR Color Fix",
            category=CATEGORY,
            description=(
                "Matches enhanced-frame colors to the source. Keeping post-processing "
                "separate allows method changes without rerunning STAR."
            ),
            inputs=[
                io.Image.Input(
                    "images", tooltip="Raw enhanced frames from STAR decode."
                ),
                io.Image.Input(
                    "reference_images",
                    tooltip="Original source frames used as color reference.",
                ),
                io.Combo.Input("method", options=COLOR_FIX_OPTIONS, default="AdaIN"),
            ],
            outputs=[io.Image.Output("images", display_name="images")],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        reference_images: torch.Tensor,
        method: str,
    ) -> io.NodeOutput:
        return io.NodeOutput(_color_fix(images, reference_images, method))


class STARVSRNode(io.ComfyNode):
    """V3 all-in-one compatibility node retaining the released workflow ID."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="STARVSRNode",
            display_name="STAR Video Super Resolution (All-in-One)",
            category="video/upscaling",
            description=(
                "V3 compatibility node for existing workflows. New workflows should use "
                "the modular STAR nodes for finer ComfyUI caching."
            ),
            search_aliases=["STAR Video Super Resolution"],
            inputs=[
                io.Image.Input("images"),
                *_model_inputs(),
                io.String.Input("prompt", default="a good video", multiline=True),
                io.Int.Input("resolution", default=720, min=16, max=16384, step=2),
                *_sampling_inputs(),
                _vae_decode_chunk_input(optional=True),
            ],
            outputs=[io.Image.Output("images", display_name="images")],
            is_deprecated=True,
        )

    @classmethod
    def fingerprint_inputs(cls, model: str, precision: str, **_kwargs):
        return model, precision, _checkpoint_fingerprint(model)

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        model: str,
        precision: str,
        prompt: str,
        resolution: int,
        max_chunk_len: int,
        cfg: float,
        sampler: str,
        solver_mode: str,
        steps: int,
        seed: int,
        vae_decode_chunk: int = 1,
    ) -> io.NodeOutput:
        star_model = _load_model(model, precision)
        video = _prepare_video(images, resolution)
        caption = prompt or star_model.model.positive_prompt
        logger.info(f"Caption: {caption}")
        conditioning = STARConditioning(
            embeddings=star_model.model.encode_prompt(caption), prompt=caption
        )

        setup_seed(seed)
        samples, padding, output_size, batch_size = star_model.model.sample_latent(
            {"video_data": video.video_data, "target_res": video.target_res},
            conditioning.embeddings,
            total_noise_levels=TOTAL_NOISE_LEVELS,
            steps=steps,
            solver=sampler,
            solver_mode=solver_mode,
            guide_scale=cfg,
            max_chunk_len=max_chunk_len,
        )
        latent = STARSampledLatent(samples, padding, output_size, batch_size)
        decoded = _decode_video(star_model, latent, vae_decode_chunk)
        return io.NodeOutput(_color_fix(decoded, images, "AdaIN"))


STAR_NODE_CLASSES = [
    STARModelLoader,
    STARPrepareVideo,
    STARTextEncode,
    STARSample,
    STARDecode,
    STARColorFix,
    STARVSRNode,
]
