from typing import List

from torch import nn

HOOKS = {}


def create_hook_fn(save_mode):
    def hook_fn(module, args, kwargs, outputs):
        if save_mode == "input":
            feats = kwargs["hidden_states"]
        elif save_mode == "output":
            feats = outputs
        else:
            raise NotImplementedError(f"Save mode {save_mode} not supported.")
        setattr(module, "feats", feats)
    return hook_fn


def init_unet_block_func(
    unet,
    save_mode="", # input or output
    reset=True,
    idxs=None
):
    assert save_mode in ["input", "output", ""]
    # register forward hook
    layers = collect_layers(unet, idxs) # idxs=None will collect all layers
    for layer in layers:
        if reset and hasattr(layer, "feats"):
            layer.feats = None
            if layer in HOOKS:
                HOOKS.pop(layer).remove()
        if save_mode != "" and layer not in HOOKS: # intended behavior
            # if save mode is empty, we don't want to add a hook (undefined behavior)
            HOOKS[layer] = layer.register_forward_hook(create_hook_fn(save_mode), with_kwargs=True)


def collect_layers(unet, idxs=None) -> List[nn.Module]:
    layers = []
    layer_idx = 0
    for up_block in unet.up_blocks:
        if idxs is None or layer_idx in idxs:
            layers.append(up_block)
        layer_idx += 1
    return layers


def collect_block_feats(unet, idxs=None):
    return [module.feats for module in collect_layers(unet, idxs)]