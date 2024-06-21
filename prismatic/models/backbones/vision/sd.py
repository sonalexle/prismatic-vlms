import math
from typing import Callable, Tuple, List
from functools import partial

import torch
import einops
from torchvision import transforms as T
from torch.distributed.fsdp.wrap import _module_wrap_policy

from prismatic.models.backbones.vision.dhf.diffusion_extractor import DiffusionExtractor
from prismatic.models.backbones.vision.base_vision import VisionBackbone


def get_default_sd_transform(res: int = 512):
    # https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py#L779
    t = T.Compose(
        [
            T.Resize((res,res), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
    )
    return t


def get_sd_backbone_default_config():
    return {
        'num_timesteps': 1000,
        'save_timestep': 999,
        'prompt': '',
        'negative_prompt': '',
        'guidance_scale': -1,
        'use_time_emb': False,
        'idxs': '[2,3]', # SD1.5 0-11 (recommended: 2, 5, 8), SDXL 0-8 (rec: 2, 5)
        'save_mode': 'resnet_hidden',
    }


def get_sd_backbone_single_layer_config():
    return {
        'output_resolution': 16,
        'save_mode': 'resnet_hidden',
        'idxs': '[3]',
    }


def get_sd_backbone_upblock_outputs_config():
    return {
        'output_resolution': 16,
        'save_mode': 'block_output',
        'idxs': '[0,1]' # SD1.5 0-3 SDXL 0-2
    }


def get_sd_backbone_resnet_outputs_config():
    return {
        'output_resolution': 16,
        'save_mode': 'resnet_output',
        'idxs': '[2,5,8]',
    }


def get_sd_backbone_crossattn_config():
    return {
        'output_resolution': 16,
        'save_mode': 'crossattn_query',
        'idxs': '[1,4,7]' # SD1.5 0-8, SDXL 0-35
    }

def get_sd_backbone_monkey_config():
    return {
        'save_mode': 'crossattn_query',
        'idxs': '[1,4,7]',
        'use_resampler': True,
    }


class SDBackbone(VisionBackbone):
    def __init__(
            self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224,
            model_id: str = 'runwayml/stable-diffusion-v1-5', config_ver="default"
        ) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)
        config = get_sd_backbone_default_config()
        if config_ver == "default":
            pass
        elif config_ver == "single-layer":
            new_config = get_sd_backbone_single_layer_config()
        elif config_ver == "upblock-outputs":
            new_config = get_sd_backbone_upblock_outputs_config()
        elif config_ver == "resnet-outputs":
            new_config = get_sd_backbone_resnet_outputs_config()
        elif config_ver == "crossattn-query":
            new_config = get_sd_backbone_crossattn_config()
        elif config_ver == "monkey":
            new_config = get_sd_backbone_monkey_config()
        else:
            raise NotImplementedError(config_ver)
        config.update(new_config)
        config["model_id"] = model_id
        self.use_resampler: bool = config.pop('use_resampler', False)
        self.diffusion_extractor = DiffusionExtractor(**config)
        self.use_time_emb: bool = self.diffusion_extractor.use_time_emb # this also controls how the projector is initialized
        self.eval_mode: bool = True # controls fixed (True) or random timestep (False) sampling
        self.load_resolution: int = self.diffusion_extractor.load_resolution
        self.output_resolution: int = self.diffusion_extractor.output_resolution
        if self.use_resampler:
            assert self.output_resolution in [64, 128], \
                f"Resampler only supports output resolutions 64 and 128, got {self.output_resolution}"
        # note that SDv1.5 uses 512x512 images, SDXL and SD3 uses 1024x1024 natively
        self.image_transform = get_default_sd_transform(self.load_resolution)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.eval_mode = True
        feats, time_emb = self.diffusion_extractor(pixel_values, eval_mode=self.eval_mode)
        feats = feats.clone() # inference tensors cannot be saved for backward, need to create a copy
        # permute to [batch_size, w, h, channels] and reshape to [batch_size, w*h, channels] using einops
        feats = einops.rearrange(feats, 'b s l h w -> b (h w) (s l)') # s is timesteps, l is layers, collapse them
        if self.use_time_emb:
            feats = feats, time_emb.clone()
        return feats

    def get_fsdp_wrapping_policy(self) -> Callable:
        return partial(
            _module_wrap_policy,
            module_classes={DiffusionExtractor},
        )

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return 3, self.default_image_size, self.default_image_size

    @property
    def embed_dim(self) -> int:
        return sum(self.feature_dims)

    @property
    def feature_dims(self) -> List[int]:
        return self.diffusion_extractor.dims

    @property
    def num_patches(self) -> int:
        patch_size = self.default_image_size // 14
        if not self.use_resampler:
            assert patch_size == self.output_resolution, \
                f"Not using a resampler but patch size {patch_size} does not match output resolution {self.output_resolution}"
        return patch_size * patch_size

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16
