"""
materialize.py

Factory class for initializing Vision Backbones, LLM Backbones, and VLMs from a set registry; provides and exports
individual functions for clear control flow.
"""

from typing import Optional, Tuple

from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm import (
    LLaMa2LLMBackbone,
    LLMBackbone,
    MistralLLMBackbone,
    PhiLLMBackbone,
    Phi3LLMBackbone
)
from prismatic.models.backbones.vision import (
    CLIPViTBackbone,
    DinoCLIPViTBackbone,
    DinoSigLIPViTBackbone,
    DinoSigLIPSDViTBackbone,
    SigLIPSDViTBackbone,
    DinoV2ViTBackbone,
    ImageTransform,
    IN1KViTBackbone,
    SigLIPViTBackbone,
    SDBackbone,
    VisionBackbone,
)
from prismatic.models.vlms import PrismaticVLM

# === Registries =>> Maps ID --> {cls(), kwargs} :: Different Registries for Vision Backbones, LLM Backbones, VLMs ===
# fmt: off

# === Vision Backbone Registry ===
VISION_BACKBONES = {
    # === 224px Backbones ===
    "clip-vit-l": {"cls": CLIPViTBackbone, "kwargs": {"default_image_size": 224}},
    "siglip-vit-so400m": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 224}},
    "dinov2-vit-l": {"cls": DinoV2ViTBackbone, "kwargs": {"default_image_size": 224}},
    "in1k-vit-l": {"cls": IN1KViTBackbone, "kwargs": {"default_image_size": 224}},
    "dinosiglip-vit-so-224px": {"cls": DinoSigLIPViTBackbone, "kwargs": {"default_image_size": 224}},
    "dinosiglipsd-vit-so-224px": {"cls": DinoSigLIPSDViTBackbone, "kwargs": {"default_image_size": 224}},
    "siglipsd-vit-so-224px": {"cls": SigLIPSDViTBackbone, "kwargs": {"default_image_size": 224}},

    # === Assorted CLIP Backbones ===
    "clip-vit-b": {"cls": CLIPViTBackbone, "kwargs": {"default_image_size": 224}},
    "clip-vit-l-336px": {"cls": CLIPViTBackbone, "kwargs": {"default_image_size": 336}},

    # === Assorted SigLIP Backbones ===
    "siglip-vit-b16-224px": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 224}},
    "siglip-vit-b16-256px": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 256}},
    "siglip-vit-b16-384px": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 384}},
    "siglip-vit-so400m-384px": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 384}},

    # === Fused Backbones ===
    "dinoclip-vit-l-336px": {"cls": DinoCLIPViTBackbone, "kwargs": {"default_image_size": 336}},
    "dinosiglip-vit-so-384px": {"cls": DinoSigLIPViTBackbone, "kwargs": {"default_image_size": 384}},
    "dinosiglipsd-vit-so-384px": {"cls": DinoSigLIPSDViTBackbone, "kwargs": {"default_image_size": 384}},
    "siglipsd-vit-so-384px": {"cls": SigLIPSDViTBackbone, "kwargs": {"default_image_size": 384}},

    # === SD Backbones ===
    "sd1.5-224px": {"cls": SDBackbone, "kwargs": {"default_image_size": 224}},
    "sd1.5-single-layer-224px": {"cls": SDBackbone, "kwargs": {"default_image_size": 224, "config_ver": "single-layer"}},
    "sd1.5-upblock-outputs-224px": {"cls": SDBackbone, "kwargs": {"default_image_size": 224, "config_ver": "upblock-outputs"}},
    "sd1.5-resnet-outputs-224px": {"cls": SDBackbone, "kwargs": {"default_image_size": 224, "config_ver": "resnet-outputs"}},
    "sd1.5-resnet-adaptive-224px": {"cls": SDBackbone, "kwargs": {"default_image_size": 224, "config_ver": "resnet-adaptive"}},
    "sd1.5-resnet-hidden-224px": {"cls": SDBackbone, "kwargs": {"default_image_size": 224, "config_ver": "resnet-hidden"}},
    "sd1.5-crossattn-query-224px": {"cls": SDBackbone, "kwargs": {"default_image_size": 224, "config_ver": "crossattn-query"}},
    "sd1.5-monkey-224px": {"cls": SDBackbone, "kwargs": {"default_image_size": 224, "config_ver": "monkey"}},
    "sdxl-turbo-224px": {"cls": SDBackbone, "kwargs": {"default_image_size": 224, "model_id": "stabilityai/sdxl-turbo", "config_ver": "resnet-adaptive"}},
}


# === Language Model Registry ===
LLM_BACKBONES = {
    # === LLaMa-2 Pure (Non-Chat) Backbones ===
    "llama2-7b-pure": {"cls": LLaMa2LLMBackbone, "kwargs": {}},
    "llama2-13b-pure": {"cls": LLaMa2LLMBackbone, "kwargs": {}},

    # === LLaMa-2 Chat Backbones ===
    "llama2-7b-chat": {"cls": LLaMa2LLMBackbone, "kwargs": {}},
    "llama2-13b-chat": {"cls": LLaMa2LLMBackbone, "kwargs": {}},

    # === Vicuna-v1.5 Backbones ===
    "vicuna-v15-7b": {"cls": LLaMa2LLMBackbone, "kwargs": {}},
    "vicuna-v15-13b": {"cls": LLaMa2LLMBackbone, "kwargs": {}},

    # === Mistral v0.1 Backbones ===
    "mistral-v0.1-7b-pure": {"cls": MistralLLMBackbone, "kwargs": {}},
    "mistral-v0.1-7b-instruct": {"cls": MistralLLMBackbone, "kwargs": {}},

    # === Phi-2 Backbone ===
    "phi-2-3b": {"cls": PhiLLMBackbone, "kwargs": {}},

    # === Phi-3 Backbone ===
    "phi3_instruct": {"cls": Phi3LLMBackbone, "kwargs": {}},
}

# fmt: on


def get_vision_backbone_and_transform(
    vision_backbone_id: str, image_resize_strategy: str
) -> Tuple[VisionBackbone, ImageTransform]:
    """Instantiate a Vision Backbone, returning both the nn.Module wrapper class and default Image Transform."""
    if vision_backbone_id in VISION_BACKBONES:
        vision_cfg = VISION_BACKBONES[vision_backbone_id]
        vision_backbone: VisionBackbone = vision_cfg["cls"](
            vision_backbone_id, image_resize_strategy, **vision_cfg["kwargs"]
        )
        image_transform = vision_backbone.get_image_transform()
        return vision_backbone, image_transform

    else:
        raise ValueError(f"Vision Backbone `{vision_backbone_id}` is not supported!")


def get_llm_backbone_and_tokenizer(
    llm_backbone_id: str,
    llm_max_length: int = 2048,
    hf_token: Optional[str] = None,
    inference_mode: bool = False,
) -> Tuple[LLMBackbone, PreTrainedTokenizerBase]:
    if llm_backbone_id in LLM_BACKBONES:
        llm_cfg = LLM_BACKBONES[llm_backbone_id]
        llm_backbone: LLMBackbone = llm_cfg["cls"](
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            **llm_cfg["kwargs"],
        )
        tokenizer = llm_backbone.get_tokenizer()
        return llm_backbone, tokenizer

    else:
        raise ValueError(f"LLM Backbone `{llm_backbone_id}` is not supported!")


def get_vlm(
    model_id: str,
    arch_specifier: str,
    vision_backbone: VisionBackbone,
    llm_backbone: LLMBackbone,
    enable_mixed_precision_training: bool = True,
) -> PrismaticVLM:
    """Lightweight wrapper around initializing a VLM, mostly for future-proofing (if one wants to add a new VLM)."""
    return PrismaticVLM(
        model_id,
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=enable_mixed_precision_training,
        arch_specifier=arch_specifier,
    )
