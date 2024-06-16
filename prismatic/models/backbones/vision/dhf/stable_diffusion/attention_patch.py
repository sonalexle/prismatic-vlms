from typing import List

from diffusers.models.attention_processor import Attention

HOOKS = {}


def create_hook_fn(save_mode, module_save=None):
    def hook_fn(module, args, outputs):
        if save_mode in ["input", "query"]:
            feats, *_ = args
        elif save_mode in ["output"]:
            feats = outputs
        else:
            raise NotImplementedError(f"Save mode {save_mode} not supported.")
        setattr(module_save or module, "feats", feats)
    return hook_fn


def init_cross_attn_func(
    unet,
    save_mode="", # input or output
    reset=True,
    idxs=None
):
    assert save_mode in ["input", "output", "query", ""]
    # register forward hook
    layers = collect_layers(unet, idxs) # idxs=None will collect all layers
    for layer in layers:
        if reset and hasattr(layer, "feats"):
            layer.feats = None
            if layer in HOOKS:
                HOOKS.pop(layer).remove()
        if save_mode == "": continue
        if layer in HOOKS: continue
        if save_mode != "query": # intended behavior
            # if save mode is empty, we don't want to add a hook (undefined behavior)
            HOOKS[layer] = layer.register_forward_hook(create_hook_fn(save_mode))
        else:
            HOOKS[layer] = layer.to_q.register_forward_hook(create_hook_fn(save_mode, layer))


def collect_layers(unet, idxs=None) -> List[Attention]:
    layers = []
    layer_idx = 0
    for up_block in unet.up_blocks:
        if not hasattr(up_block, "attentions"): continue
        for attention in up_block.attentions:
            for transformer_block in attention.transformer_blocks:
                # these are cross attentions (image-text attn)
                if not hasattr(transformer_block, "attn2"): continue
                if idxs is None or layer_idx in idxs:
                    layers.append(transformer_block.attn2)
                layer_idx += 1
    return layers


def collect_attention_feats(unet, idxs=None):
    return [module.feats for module in collect_layers(unet, idxs)]


# from diffusers import StableDiffusionPipeline
# import torch
# pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.bfloat16)