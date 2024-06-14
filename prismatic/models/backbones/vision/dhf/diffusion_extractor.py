# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =================================================================
# This code is adapted from diffusion_extractor.py in Diffusion Hyperfeatures
# and implements extraction of diffusion features from a single random timestep 
# of the generation process, rather than all features from the inversion process.
# Original source: https://github.com/diffusion-hyperfeatures/diffusion_hyperfeatures/blob/main/archs/diffusion_extractor.py
# =================================================================

import ast
import einops
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List

from prismatic.models.backbones.vision.dhf.stable_diffusion.diffusion import generalized_steps
from prismatic.models.backbones.vision.dhf.stable_diffusion.resnet import init_resnet_func, collect_channels
from prismatic.models.backbones.vision.readout_guidance import rg_helpers

class DiffusionExtractor(nn.Module):
    def __init__(self, **config):
        super().__init__()

        ## Model setup ##
        self.model_id: str = config.pop("model_id", "runwayml/stable-diffusion-v1-5")
        self.model, self.dtype = rg_helpers.load_pipeline({"model_path": self.model_id})
        self.unet, self.vae = self.model.unet, self.model.vae
        self.text_encoder = self.model.text_encoder
        if hasattr(self.model, "text_encoder_2"):
            self.text_encoder_2 = self.model.text_encoder_2

        ## Timestep scheduling ##
        self.scheduler = self.model.scheduler
        self.num_timesteps: int = config.pop("num_timesteps", 1000)
        # self.max_train_timesteps = config.get("max_train_timesteps", self.num_timesteps // 10 * 9) # 90% of timesteps, don't train on really noisy images
        self.scheduler.set_timesteps(self.num_timesteps)
        self.emb: Optional[torch.Tensor] = None # timestep embedding, for conditioning the aggregation network
        # Note that save_timestep is in terms of number of generation steps
        # save_timestep = 0 is noise, save_timestep = T is a clean image
        # generation saves as [0...T], inversion saves as [T...0]
        self.save_timestep: Optional[int] = config.pop("save_timestep", 999)

        ## Text Embeddings ##
        self.prompt: str = config.pop("prompt", "")
        self.negative_prompt: str = config.pop("negative_prompt", "")
        self.guidance_scale: int = config.pop("guidance_scale", -1)
        self.use_time_emb: bool = config.pop('use_time_emb', False)

        ## Automatically determine default latent and image dim ##
        self.height = self.width = self.unet.config.sample_size * self.model.vae_scale_factor
        self.latent_height = self.latent_width = self.unet.config.sample_size
        self.load_resolution = self.height
        self.output_resolution = config.pop("output_resolution", self.latent_height)

        ## Hyperparameters ##
        self.diffusion_mode: str = config.pop("diffusion_mode", "generation")
        self.save_mode: str = config.pop("save_mode", "hidden") # which resnet representation to save, input, hidden, output
        # if eval mode, the timestep to save is fixed, otherwise randomly samples from [0, T-1]
        self.eval_mode: bool = config.pop("eval_mode", False)
        idxs = config.pop("idxs", None)
        self.idxs: Optional[List[int]] = None
        if idxs is not None:
            self.idxs = ast.literal_eval(idxs) # passed as a string
        self.dims: List[int] = collect_channels(self.unet, idxs=self.idxs)

    @torch.inference_mode()
    def change_cond(self, prompt, negative_prompt, batch_size=1):
        assert batch_size is not None, "Batch size must be specified."
        if "xl" in self.model_id:
            context, added_cond_kwargs = rg_helpers.get_context_sdxl(
                self.model,
                [prompt] * batch_size,
                batch_size,
                original_size=(self.height, self.width),
                crops_coords_top_left=(0, 0),
                target_size=(self.height, self.width),
                negative_prompt=negative_prompt
            )
        else:
            context = rg_helpers.get_context(self.model, [prompt] * batch_size, negative_prompt=negative_prompt)
            added_cond_kwargs = {}
        return context, added_cond_kwargs

    def init_aux_embeds(self):
        self.context, self.added_cond_kwargs = self.change_cond(self.prompt, self.negative_prompt)

    def run_generation(self, latent, guidance_scale=None, min_i=None, max_i=None):
        context, added_cond_kwargs = self.change_cond(self.prompt, self.negative_prompt, latent.shape[0])
        # context, added_cond_kwargs = self.context, self.added_cond_kwargs
        xs = generalized_steps(
            latent,
            self.unet, 
            self.scheduler, 
            run_inversion=False,
            guidance_scale=guidance_scale or self.guidance_scale,
            context=context,
            min_i=min_i,
            max_i=max_i,
            added_cond_kwargs=added_cond_kwargs
        )
        return xs

    def get_feats(self, latents, extractor_fn, preview_mode=False):
        if not preview_mode:
            init_resnet_func(self.unet, save_mode=self.save_mode, reset=True, idxs=self.idxs)
        # NOTE: this line actually runs the unet denoising, and features are saved as resblock.feats
        outputs = extractor_fn(latents)
        if not preview_mode:
            feats = rg_helpers.collect_and_resize_feats(self.model, self.idxs, self.output_resolution)
            # convert feats to [batch_size, num_timesteps, channels, w, h]
            feats = feats[..., None] # since there is only 1 time step, l=layer, s=timestep
            feats = einops.rearrange(feats, 'b w h l s -> b s l w h')
            init_resnet_func(self.unet, reset=True)
        else:
            feats = None
        return feats, outputs

    @torch.inference_mode()
    def forward(
        self, images, guidance_scale: Optional[int] = None,
        preview_mode: bool = False, eval_mode: bool = False,
        save_timestep: Optional[int] = None
    ):
        self.eval()
        assert self.diffusion_mode == "generation", "Only generation, not inversion, supported."
        latents = rg_helpers.images_to_latents(self.vae, images, (self.height, self.width))
        if eval_mode or self.eval_mode: # read the comments about save_timestep in init
            save_timestep = save_timestep or self.save_timestep or self.num_timesteps - 1
            assert not isinstance(save_timestep, list), "This implementation only saves one timestep"
        else: # we train other things, not this model (self)
            save_timestep = np.random.choice(range(1, self.num_timesteps)) # max is exclusive
        noise_timestep = self.num_timesteps - 1 - save_timestep # 0-indexed
        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, torch.tensor(noise_timestep))
        extractor_fn = lambda latents: self.run_generation(latents, guidance_scale, min_i=save_timestep, max_i=save_timestep+1)

        feats, outputs = self.get_feats(
            latents, extractor_fn=extractor_fn, preview_mode=preview_mode
        )
        if self.use_time_emb:
            emb = rg_helpers.embed_timestep(self.unet, latents, save_timestep)
        else:
            emb = None

        return feats, outputs, emb