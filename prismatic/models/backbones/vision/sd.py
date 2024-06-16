import math
from typing import Callable, Tuple, List
from functools import partial

import torch
import einops
from torchvision import transforms as T
from torch.distributed.fsdp.wrap import _module_wrap_policy

from prismatic.models.backbones.vision.dhf.diffusion_extractor import DiffusionExtractor
from prismatic.models.backbones.vision.dhf.aggregation_network import AggregationNetwork
from prismatic.models.backbones.vision.base_vision import ImageTransform, LetterboxPad, VisionBackbone, unpack_tuple


def get_default_sd_transform(res: int = 512):
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return T.Compose([
        T.Resize((res,res), T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        normalize
    ])


def get_sd_backbone_default_config():
    return {
        'num_timesteps': 1000,
        'save_timestep': 999,
        'prompt': '',
        'negative_prompt': '',
        'guidance_scale': -1,
        'output_resolution': 16,
        'use_time_emb': False,
        'idxs': '[2, 3]',
        # "idxs": "[2,3,8,10]"
        # 'use_resampler': True,
    }


def get_sd_backbone_single_layer_config():
    return {
        'idxs': '[3]',
    }


def get_sd_backbone_upblock_outputs_config():
    return {
        'save_mode': 'block_output',
        'idxs': '[0,1]'
    }

def get_sd_backbone_crossattn_config():
    return {
        'save_mode': 'crossattn_query',
        'idxs': '[1,4,7]'
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
            config.update(get_sd_backbone_single_layer_config())
        elif config_ver == "upblock-outputs":
            config.update(get_sd_backbone_upblock_outputs_config())
        elif config_ver == "crossattn-query":
            config.update(get_sd_backbone_crossattn_config())
        else:
            raise NotImplementedError(config_ver)
        config["model_id"] = model_id
        self.use_resampler: bool = config.pop('use_resampler', False)
        self.use_aggregation_net: bool = False # NOTE: this is set during projector initialization
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
        feats, _, time_emb = self.diffusion_extractor(pixel_values, eval_mode=self.eval_mode)
        feats = feats.clone() # inference tensors cannot be saved for backward, need to create a copy
        feats = einops.rearrange(feats, 'b s l w h -> b (s l) w h') # s is timesteps, l is layers
        if not self.use_aggregation_net:
            # permute to [batch_size, w, h, channels] and reshape to [batch_size, w*h, channels] using einops
            feats = einops.rearrange(feats, 'b c w h -> b (w h) c')
        elif self.use_time_emb:
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
