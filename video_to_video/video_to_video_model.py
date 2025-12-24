import os
import os.path as osp
import random
from typing import Any, Dict, Literal

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLTemporalDecoder

from ..video_to_video.diffusion.diffusion_sdedit import GaussianDiffusion
from ..video_to_video.diffusion.schedules_sdedit import noise_schedule
from ..video_to_video.modules import *
from ..video_to_video.utils.config import cfg
from ..video_to_video.utils.logger import get_logger

logger = get_logger()

# Try to import bitsandbytes for quantization
try:
    import bitsandbytes as bnb

    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    logger.warning(
        "bitsandbytes not available. NF4 quantization will not be supported."
    )


class VideoToVideo_sr:
    def __init__(
        self,
        opt,
        device=torch.device(f"cuda:0"),
        precision: Literal["fp16", "fp8", "nf4"] = "fp16",
    ):
        self.opt = opt
        self.device = device  # torch.device(f'cuda:0')
        self.precision = precision

        # text_encoder
        text_encoder = FrozenOpenCLIPEmbedder(
            device=self.device, pretrained="laion2b_s32b_b79k"
        )
        text_encoder.model.to(self.device)
        self.text_encoder = text_encoder
        logger.info(f"Build encoder with FrozenOpenCLIPEmbedder")

        # U-Net with ControlNet
        generator = ControlledV2VUNet()
        generator = generator.to(self.device)
        generator.eval()

        cfg.model_path = opt.model_path
        load_dict = torch.load(cfg.model_path, map_location="cpu")
        if "state_dict" in load_dict:
            load_dict = load_dict["state_dict"]
        ret = generator.load_state_dict(load_dict, strict=False)

        # Apply quantization based on precision parameter
        if self.precision == "nf4":
            if not BITSANDBYTES_AVAILABLE:
                logger.warning("bitsandbytes not available. Falling back to FP16.")
                self.generator = generator.half()
            else:
                logger.info("Quantizing model to NF4 (4-bit)")
                self.generator = self._quantize_nf4(generator)
        elif self.precision == "fp8":
            logger.info("Quantizing model to FP8")
            self.generator = self._quantize_fp8(generator)
        else:
            logger.info("Using FP16 precision")
            self.generator = generator.half()

        logger.info(
            "Load model path {}, with local status {}".format(cfg.model_path, ret)
        )

        # Noise scheduler
        sigmas = noise_schedule(
            schedule="logsnr_cosine_interp",
            n=1000,
            zero_terminal_snr=True,
            scale_min=2.0,
            scale_max=4.0,
        )
        diffusion = GaussianDiffusion(sigmas=sigmas)
        self.diffusion = diffusion
        logger.info("Build diffusion with GaussianDiffusion")

        # Temporal VAE
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid",
            subfolder="vae",
            variant="fp16",
        )
        vae.eval()
        vae.requires_grad_(False)
        vae.to(self.device)
        self.vae = vae
        logger.info("Build Temporal VAE")

        torch.cuda.empty_cache()

        self.negative_prompt = cfg.negative_prompt
        self.positive_prompt = cfg.positive_prompt

        negative_y = text_encoder(self.negative_prompt).detach()
        self.negative_y = negative_y

    def test(
        self,
        input: Dict[str, Any],
        total_noise_levels=1000,
        steps=50,
        solver="dpmpp_2m_sde",
        solver_mode="fast",
        guide_scale=7.5,
        max_chunk_len=32,
    ):
        video_data = input["video_data"]
        y = input["y"]
        (target_h, target_w) = input["target_res"]

        video_data = F.interpolate(video_data, [target_h, target_w], mode="bilinear")

        logger.info(f"video_data shape: {video_data.shape}")
        frames_num, _, h, w = video_data.shape

        padding = pad_to_fit(h, w)
        video_data = F.pad(video_data, padding, "constant", 1)

        video_data = video_data.unsqueeze(0)
        bs = 1
        video_data = video_data.to(self.device)

        video_data_feature = self.vae_encode(video_data)
        torch.cuda.empty_cache()

        y = self.text_encoder(y).detach()

        with torch.amp.autocast("cuda", enabled=True):
            t = torch.LongTensor([total_noise_levels - 1]).to(self.device)
            noised_lr = self.diffusion.diffuse(video_data_feature, t)

            model_kwargs = [{"y": y}, {"y": self.negative_y}]
            model_kwargs.append({"hint": video_data_feature})

            torch.cuda.empty_cache()
            chunk_inds = (
                make_chunks(frames_num, interp_f_num=0, max_chunk_len=max_chunk_len)
                if frames_num > max_chunk_len
                else None
            )

            gen_vid = self.diffusion.sample_sr(
                noise=noised_lr,
                model=self.generator,
                model_kwargs=model_kwargs,
                guide_scale=guide_scale,
                guide_rescale=0.2,
                solver=solver,
                solver_mode=solver_mode,
                return_intermediate=None,
                steps=steps,
                t_max=total_noise_levels - 1,
                t_min=0,
                discretization="trailing",
                chunk_inds=chunk_inds,
            )
            torch.cuda.empty_cache()

            logger.info(f"sampling, finished.")
            vid_tensor_gen = self.vae_decode_chunk(gen_vid, chunk_size=3)

            logger.info(f"temporal vae decoding, finished.")

        w1, w2, h1, h2 = padding
        vid_tensor_gen = vid_tensor_gen[:, :, h1 : h + h1, w1 : w + w1]

        gen_video = rearrange(vid_tensor_gen, "(b f) c h w -> b c f h w", b=bs)

        torch.cuda.empty_cache()

        return gen_video.type(torch.float32).cpu()

    def temporal_vae_decode(self, z, num_f):
        return self.vae.decode(
            z / self.vae.config.scaling_factor, num_frames=num_f
        ).sample

    def vae_decode_chunk(self, z, chunk_size=3):
        z = rearrange(z, "b c f h w -> (b f) c h w")
        video = []
        for ind in range(0, z.shape[0], chunk_size):
            num_f = z[ind : ind + chunk_size].shape[0]
            video.append(self.temporal_vae_decode(z[ind : ind + chunk_size], num_f))
        video = torch.cat(video)
        return video

    def vae_encode(self, t, chunk_size=1):
        num_f = t.shape[1]
        t = rearrange(t, "b f c h w -> (b f) c h w")
        z_list = []
        for ind in range(0, t.shape[0], chunk_size):
            z_list.append(
                self.vae.encode(t[ind : ind + chunk_size]).latent_dist.sample()
            )
        z = torch.cat(z_list, dim=0)
        z = rearrange(z, "(b f) c h w -> b c f h w", f=num_f)
        return z * self.vae.config.scaling_factor

    def _quantize_nf4(self, model):
        """
        Recursively quantize Linear layers to NF4 (4-bit NormalFloat)
        Requires bitsandbytes library
        """
        if not BITSANDBYTES_AVAILABLE:
            logger.error("bitsandbytes is not available. Cannot quantize to NF4.")
            return model.half()

        for name, module in model.named_children():
            if isinstance(module, torch.nn.Linear):
                # Replace with 4-bit quantized version
                quant_layer = bnb.nn.Linear4bit(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    compute_dtype=torch.bfloat16,
                    compress_statistics=True,
                    quant_type="nf4",
                )

                # Copy and quantize weights
                quant_layer.weight = bnb.nn.Params4bit(
                    module.weight.data.to("cpu"), requires_grad=False, quant_type="nf4"
                )
                if module.bias is not None:
                    quant_layer.bias = module.bias

                setattr(model, name, quant_layer)
            elif len(list(module.children())) > 0:
                # Recursively quantize child modules
                self._quantize_nf4(module)

        return model

    def _quantize_fp8(self, model):
        """
        Quantize model to FP8 (requires PyTorch 2.1+ and compatible GPU)
        Falls back to FP16 if FP8 is not supported
        """
        if hasattr(torch, "float8_e4m3fn"):
            try:
                logger.info("Converting model to FP8 (E4M3)")
                for param in model.parameters():
                    if param.dtype in [torch.float32, torch.float16]:
                        param.data = param.data.to(torch.float8_e4m3fn)
                return model
            except Exception as e:
                logger.warning(f"FP8 conversion failed: {e}. Falling back to FP16.")
                return model.half()
        else:
            logger.warning(
                "FP8 not supported on this PyTorch version. Falling back to FP16."
            )
            return model.half()


def pad_to_fit(h, w):
    BEST_H, BEST_W = 720, 1280

    if h < BEST_H:
        h1, h2 = _create_pad(h, BEST_H)
    elif h == BEST_H:
        h1 = h2 = 0
    else:
        h1 = 0
        h2 = int((h + 48) // 64 * 64) + 64 - 48 - h

    if w < BEST_W:
        w1, w2 = _create_pad(w, BEST_W)
    elif w == BEST_W:
        w1 = w2 = 0
    else:
        w1 = 0
        w2 = int(w // 64 * 64) + 64 - w
    return (w1, w2, h1, h2)


def _create_pad(h, max_len):
    h1 = int((max_len - h) // 2)
    h2 = max_len - h1 - h
    return h1, h2


def make_chunks(f_num, interp_f_num, max_chunk_len, chunk_overlap_ratio=0.5):
    MAX_CHUNK_LEN = max_chunk_len
    MAX_O_LEN = MAX_CHUNK_LEN * chunk_overlap_ratio
    chunk_len = int((MAX_CHUNK_LEN - 1) // (1 + interp_f_num) * (interp_f_num + 1) + 1)
    o_len = int((MAX_O_LEN - 1) // (1 + interp_f_num) * (interp_f_num + 1) + 1)
    chunk_inds = sliding_windows_1d(f_num, chunk_len, o_len)
    return chunk_inds


def sliding_windows_1d(length, window_size, overlap_size):
    stride = window_size - overlap_size
    ind = 0
    coords = []
    while ind < length:
        if ind + window_size * 1.25 >= length:
            coords.append((ind, length))
            break
        else:
            coords.append((ind, ind + window_size))
            ind += stride
    return coords
