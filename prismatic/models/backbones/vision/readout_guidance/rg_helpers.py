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

import einops
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from typing import Optional, List, Tuple

from diffusers import (
    DDIMScheduler,
    DDPMScheduler, 
    PNDMScheduler,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLPipeline,
    StableDiffusionPipeline,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    T2IAdapter, 
    StableDiffusionXLAdapterPipeline
)

from prismatic.models.backbones.vision.dhf.aggregation_network import AggregationNetwork


from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from diffusers.loaders import TextualInversionLoaderMixin, StableDiffusionXLLoraLoaderMixin
from diffusers.models.lora import (
    adjust_lora_scale_text_encoder, logger
)


# ====================
#   Load Components
# ====================
def load_pipeline(config, device=None, dtype=torch.bfloat16, scheduler_mode="ddim"):
    if "xl" in config["model_path"]:
        dtype = torch.float16 if dtype is None else dtype
        pipeline = StableDiffusionXLPipeline.from_pretrained(config["model_path"], torch_dtype=dtype)
    else:
        dtype = torch.float32 if dtype is None else dtype
        pipeline = StableDiffusionPipeline.from_pretrained(config["model_path"], torch_dtype=dtype)
    if device is not None:
        pipeline = pipeline.to(device)
    pipeline.unet.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.eval()
    pipeline.vae.eval()
    pipeline.text_encoder.eval()
    if hasattr(pipeline, "text_encoder_2"):
        pipeline.text_encoder_2.requires_grad_(False)
        pipeline.text_encoder_2.eval()
    load_scheduler(pipeline, config["model_path"], mode=scheduler_mode)
    return pipeline, dtype

def load_controlnet_pipeline(config, device, dtype=torch.float32, scheduler_mode="ddim"):
    controlnet = ControlNetModel.from_pretrained(config["controlnet_path"], torch_dtype=dtype)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(config["model_path"], controlnet=controlnet, torch_dtype=dtype).to(device)
    pipeline.unet.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    load_scheduler(pipeline, config["model_path"], mode=scheduler_mode)
    return pipeline, dtype

def load_adapter_pipeline(config, device, dtype=torch.float16, scheduler_mode="ddim"):
    t2i_adapter = T2IAdapter.from_pretrained(config["adapter_path"], torch_dtype=dtype)
    pipeline = StableDiffusionXLAdapterPipeline.from_pretrained(config["model_path"], adapter=t2i_adapter, torch_dtype=dtype).to(device)
    pipeline.unet.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    load_scheduler(pipeline, config["model_path"],  mode=scheduler_mode)
    return pipeline, dtype

def load_scheduler(pipeline, MODEL_ID, mode="ddim"):
    if mode == "ddim":
        scheduler_cls = DDIMScheduler
    elif mode == "ddpm":
        scheduler_cls = DDPMScheduler
    elif mode == "pndm":
        scheduler_cls = PNDMScheduler
    elif mode == "ead":
        scheduler_cls = EulerAncestralDiscreteScheduler
    pipeline.scheduler = scheduler_cls.from_config(pipeline.scheduler.config)

def load_aggregation_network(aggregation_config, device, dtype):
    weights_path = aggregation_config["aggregation_ckpt"]
    state_dict = torch.load(weights_path)
    config = state_dict["config"]
    aggregation_kwargs = config.get("aggregation_kwargs", {})
    custom_aggregation_kwargs = {k: v for k, v in aggregation_config.items() if "aggregation" not in k}
    aggregation_kwargs = {**aggregation_kwargs, **custom_aggregation_kwargs}
    aggregation_network = AggregationNetwork(
        projection_dim=config["projection_dim"],
        feature_dims=config["dims"],
        device=device,
        save_timestep=config["save_timestep"],
        num_timesteps=config["num_timesteps"],
        **aggregation_kwargs
    )
    aggregation_network.load_state_dict(state_dict["aggregation_network"], strict=False)
    aggregation_network = aggregation_network.to(device).to(dtype)
    return aggregation_network, config

# ====================
#   Load Latents
# ====================
def get_latents(pipeline, batch_size, device, generator, dtype, latent_dim):
    latents_shape = (pipeline.unet.in_channels, *latent_dim)
    latents = torch.randn((batch_size, *latents_shape), generator=generator)
    latents = latents.to(device).to(dtype)
    return latents

def get_prompts_latents_video(prompts, latents):
    return [prompts[0]], einops.rearrange(latents, 'f c h w -> 1 c f h w')

def get_prompts_latents(pipeline, prompt, batch_size, seed, latent_dim, device, dtype=torch.float32, same_seed=True):
    generator = torch.Generator().manual_seed(seed)
    latents_shape = (pipeline.unet.in_channels, *latent_dim)
    prompts = [prompt] * batch_size
    if same_seed:
        latents = torch.randn((1, *latents_shape), generator=generator)
        latents = latents.repeat((batch_size, 1, 1, 1))
    else:
        latents = torch.randn((batch_size, *latents_shape), generator=generator)
    latents = latents.to(device).to(dtype)
    return prompts, latents

# =========================================
#    Features, Latents, and Text Context
# =========================================
def resize_feat(feat, new_res, resize_mode="bilinear", adaptive_avg_pool=False):
    # check if the feat is actually a 4D tensor
    if len(feat.shape) == 4:
        old_res = feat.shape[-1] # feat shape (b c h w)
    elif len(feat.shape) == 3: # then we are dealing with a flattened feature
        old_res = feat.shape[1] ** 0.5 # feat shape (b hw c)
        assert old_res.is_integer(), "Feature shape is not square"
        old_res = int(old_res)
        feat = einops.rearrange(feat, 'b (h w) c -> b c h w', h=old_res, w=old_res)
    else:
        raise ValueError(f"Feature shape {feat.shape} not supported.")

    if new_res < old_res and adaptive_avg_pool:
        feat = F.adaptive_avg_pool2d(feat, new_res)
    else:
        feat = F.interpolate(feat, size=(new_res,new_res), mode=resize_mode)

    feat = einops.rearrange(feat, 'b c h w -> b h w c')

    return feat


def collect_and_resize_feats(model, idxs, collect_feats_fn, latent_dim=None, adaptive_avg_pool=False):
    if model is None:
        return None
    feature_store = {"up": collect_feats_fn(model.unet, idxs=idxs)}
    feats = []
    max_feat_res = max([feat.shape[-1] for feat in feature_store["up"]]) if latent_dim is None else latent_dim
    for key in feature_store:
        for i, feat in enumerate(feature_store[key]):
            feat = resize_feat(feat, new_res=max_feat_res, adaptive_avg_pool=adaptive_avg_pool)
            feats.append(feat)
    # Concatenate all layers along the channel
    # dimension to get shape (b s d)
    if len(feats) > 0:
        feats = torch.cat(feats, dim=-1)
    else:
        feats = None
    return feats


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    image = np.array(image).astype(np.float32)
    image = image[None, ...]
    image = einops.rearrange(image, 'b w h c -> b c w h')
    image = torch.from_numpy(image)
    image = image / 255.0
    image = 2. * image - 1.
    return image

def images_to_latents(vae, images, image_dim) -> torch.Tensor:
    with torch.no_grad():
        # Run vae in torch.float32 always to avoid black images
        vae = vae.to(torch.float32)
        is_pixel_values = torch.is_tensor(images)
        if not is_pixel_values:
            images = image_to_tensor(images)
        images = images.to(vae.device).to(vae.dtype)
        if not is_pixel_values: # assume that pixel values have been properly preprocessed
            images = torch.nn.functional.interpolate(images, size=(image_dim[0], image_dim[1]), mode="bilinear")
        latents = vae.encode(images).latent_dist.sample(generator=None) 
        latents = latents * vae.config.scaling_factor
    return latents

def decode_latents(vae, latents):
    # Ensure that vae is always in torch.float32 to prevent black images / underflow
    vae = vae.to(torch.float32)
    latents = latents.to(torch.float32)
    latents = latents / vae.config.scaling_factor
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).astype(np.uint8)
    return image

def get_context(model, prompt, negative_prompt=""):
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    device = model.text_encoder.device
    text_embeddings = model.text_encoder(text_input.input_ids.to(device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [negative_prompt] * len(prompt), padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(device))[0]
    context = [uncond_embeddings, text_embeddings]
    context = torch.cat(context)
    return context

def get_context_sdxl(
    model,
    prompt,
    batch_size,
    original_size=(1024, 1024),
    crops_coords_top_left=(0, 0),
    target_size=(1024, 1024),
    negative_prompt=""
):
    device = model.device
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = encode_prompt(
        model,
        prompt=prompt,
        num_images_per_prompt=1,
        negative_prompt=[negative_prompt] * len(prompt)
    )
    context = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    if model.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = model.text_encoder_2.config.projection_dim
    add_time_ids = model._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
    add_time_ids = add_time_ids.repeat((batch_size, 1))
    context = context.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device)
    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
    return context, added_cond_kwargs

def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    images = [np.array(image) for image in images]
    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    return pil_img

# ========================
#    Scheduler Updates
# ========================
def get_variance_noise(shape, device, generator=None):
    if generator:
        variance_noise = [torch.randn((1, *shape[1:]), device=device, generator=g) for g in generator]
        return torch.vstack(variance_noise)
    else:
        return torch.randn(shape, device=device)
    
def get_seq_iter(timesteps, run_inversion):
    seq = timesteps
    seq = torch.flip(seq, dims=(0,))
    seq_next = [-1] + list(seq[:-1])
    if run_inversion:
        seq_iter = seq_next
        seq_next_iter = seq
    else:
        seq_iter = reversed(seq)
        seq_next_iter = reversed(seq_next)
    return seq_iter, seq_next_iter

def get_at_next(scheduler, t, next_t, et):
    get_at = lambda t: scheduler.alphas_cumprod[t] if t != -1 else scheduler.final_alpha_cumprod
    get_at_next = lambda next_t: scheduler.alphas_cumprod[next_t] if next_t != -1 else scheduler.final_alpha_cumprod
    if type(t) is int or len(t.shape) == 0:
        at = get_at(t)
        at_next = get_at_next(next_t)
    else:
        device, dtype = et.device, et.dtype
        at = torch.tensor([get_at(_t) for _t in t[:et.shape[0]]])
        at_next = torch.tensor([get_at_next(_next_t) for _next_t in next_t[:et.shape[0]]])
        at = at[:, None, None, None].to(device).to(dtype)
        at_next = at_next[:, None, None, None].to(device).to(dtype)
    return at, at_next

def get_xt_next(scheduler, et, t, next_t, xt, eta=0.0, variance_noise=None):
    """
    Uses the DDIM formulation for sampling xt_next
    Denoising Diffusion Implicit Models (Song et. al., ICLR 2021).
    """
    at, at_next = get_at_next(scheduler, t, next_t, et)
    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
    if eta > 0:
        c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
    else:
        c1 = 0
    c2 = ((1 - at_next) - c1 ** 2).sqrt()
    xt_next = at_next.sqrt() * x0_t + c2 * et
    if eta > 0:
        if variance_noise is not None:
            xt_next = xt_next + c1 * variance_noise
        else:
            xt_next = xt_next + c1 * torch.randn_like(xt_next)
    return xt_next, x0_t

# ========================
#  Timestep Conditioning
# ========================
def preprocess_timestep(sample, timestep):
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        is_mps = sample.device.type == "mps"
        if isinstance(timestep, float):
            dtype = torch.float32 if is_mps else torch.float64
        else:
            dtype = torch.int32 if is_mps else torch.int64
        timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
    elif len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)
    return timesteps

@torch.inference_mode()
def embed_timestep(unet, sample, timestep):
    timesteps = preprocess_timestep(sample, timestep)
    timesteps = timesteps.expand(sample.shape[0])
    t_emb = unet.time_proj(timesteps)
    t_emb = t_emb.to(dtype=sample.dtype)
    emb = unet.time_embedding(t_emb, None)
    return emb


def encode_prompt(
    self,
    prompt: str,
    prompt_2: Optional[str] = None,
    num_images_per_prompt: int = 1,
    do_classifier_free_guidance: bool = True,
    negative_prompt: Optional[str] = None,
    negative_prompt_2: Optional[str] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    lora_scale: Optional[float] = None,
    clip_skip: Optional[int] = None,
):
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
            used in both text-encoders
        num_images_per_prompt (`int`):
            number of images that should be generated per prompt
        do_classifier_free_guidance (`bool`):
            whether to use classifier free guidance or not
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        negative_prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
            `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
            If not provided, pooled text embeddings will be generated from `prompt` input argument.
        negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
            input argument.
        lora_scale (`float`, *optional*):
            A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        clip_skip (`int`, *optional*):
            Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
            the output of the pre-final layer will be used for computing the prompt embeddings.
    """
    # set lora scale so that monkey patched LoRA
    # function of text encoder can correctly access it
    if lora_scale is not None and isinstance(self, StableDiffusionXLLoraLoaderMixin):
        self._lora_scale = lora_scale

        # dynamically adjust the LoRA scale
        if self.text_encoder is not None:
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)
            else:
                scale_lora_layers(self.text_encoder_2, lora_scale)

    prompt = [prompt] if isinstance(prompt, str) else prompt

    if prompt is not None:
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # Define tokenizers and text encoders
    tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
    text_encoders = (
        [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
    )

    if prompt_embeds is None:
        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        # textual inversion: process multi-vector tokens if necessary
        prompt_embeds_list = []
        prompts = [prompt, prompt_2]
        for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, tokenizer)

            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device), output_hidden_states=True)

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            if clip_skip is None:
                prompt_embeds = prompt_embeds.hidden_states[-2]
            else:
                # "2" because SDXL always indexes from the penultimate layer.
                prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

    # get unconditional embeddings for classifier free guidance
    zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
    if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
    elif do_classifier_free_guidance and negative_prompt_embeds is None:
        negative_prompt = negative_prompt or ""
        negative_prompt_2 = negative_prompt_2 or negative_prompt

        # normalize str to list
        negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
        negative_prompt_2 = (
            batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
        )

        uncond_tokens: List[str]
        if prompt is not None and type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = [negative_prompt, negative_prompt_2]

        negative_prompt_embeds_list = []
        for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
            if isinstance(self, TextualInversionLoaderMixin):
                negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            negative_prompt_embeds = text_encoder(
                uncond_input.input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )
            # We are only ALWAYS interested in the pooled output of the final text encoder
            negative_pooled_prompt_embeds = negative_prompt_embeds[0]
            negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

            negative_prompt_embeds_list.append(negative_prompt_embeds)

        negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

    if self.text_encoder_2 is not None:
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=self.text_encoder_2.device)
    else:
        prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=self.text_encoder_2.device)

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    if do_classifier_free_guidance:
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        if self.text_encoder_2 is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=self.device)
        else:
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.unet.dtype, device=self.device)

        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
        bs_embed * num_images_per_prompt, -1
    )
    if do_classifier_free_guidance:
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )

    if self.text_encoder is not None:
        if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder, lora_scale)

    if self.text_encoder_2 is not None:
        if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder_2, lora_scale)

    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds