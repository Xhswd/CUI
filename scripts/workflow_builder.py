#!/usr/bin/env python3
"""
Dynamic ComfyUI workflow builder.

This module keeps reusable node pointers and workflow blueprints in Python so
TUI flows can build ComfyUI JSON on demand instead of maintaining many large
static workflow files.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from model_presets import normalize_image_type
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from model_presets import normalize_image_type


PLACEHOLDER_TOKEN = "PLACEHOLDER"

TIERS = {
    "basic": "基础生成",
    "image": "图像输入",
    "control": "结构控制",
    "advanced": "高级工作流",
    "ecommerce": "电商工作流",
}

KNOWN_REQUIRED_INPUTS = {
    "CheckpointLoaderSimple": ["ckpt_name"],
    "LoraLoader": ["lora_name", "strength_model", "strength_clip", "model", "clip"],
    "CLIPTextEncode": ["text", "clip"],
    "TripleCLIPLoader": ["clip_name1", "clip_name2", "clip_name3"],
    "DualCLIPLoader": ["clip_name1", "clip_name2", "type"],
    "CLIPTextEncodeSD3": ["clip", "clip_l", "clip_g", "t5xxl", "empty_padding"],
    "CLIPTextEncodeFlux": ["clip", "clip_l", "t5xxl", "guidance"],
    "EmptyLatentImage": ["width", "height", "batch_size"],
    "EmptySD3LatentImage": ["width", "height", "batch_size"],
    "KSampler": [
        "seed", "steps", "cfg", "sampler_name", "scheduler", "denoise",
        "model", "positive", "negative", "latent_image",
    ],
    "ModelSamplingSD3": ["model", "shift"],
    "ModelSamplingFlux": ["model", "max_shift", "base_shift", "width", "height"],
    "UNETLoader": ["unet_name", "weight_dtype"],
    "VAELoader": ["vae_name"],
    "VAEDecode": ["samples", "vae"],
    "VAEEncode": ["pixels", "vae"],
    "SaveImage": ["filename_prefix", "images"],
    "LoadImage": ["image"],
    "LoadImageMask": ["image", "channel"],
    "SetLatentNoiseMask": ["samples", "mask"],
    "ControlNetLoader": ["control_net_name"],
    "ControlNetApply": ["conditioning", "control_net", "image", "strength"],
    "LatentUpscale": ["upscale_method", "width", "height", "crop", "samples"],
}


def link(node_id, output=0):
    return [str(node_id), output]


class WorkflowGraph:
    def __init__(self):
        self.nodes = {}
        self._next_id = 1

    def add(self, class_type, inputs=None, node_id=None):
        if node_id is None:
            while str(self._next_id) in self.nodes:
                self._next_id += 1
            node_id = str(self._next_id)
            self._next_id += 1
        else:
            node_id = str(node_id)
            if node_id in self.nodes:
                raise ValueError(f"Duplicate node id: {node_id}")

        self.nodes[node_id] = {
            "class_type": class_type,
            "inputs": inputs or {},
        }
        return node_id

    def export(self):
        return json.loads(json.dumps(self.nodes))


class BuildParams:
    def __init__(self, **kwargs):
        self.model = kwargs.get("model") or "MODEL_PLACEHOLDER"
        self.positive = kwargs.get("positive") or "POSITIVE_PLACEHOLDER"
        self.negative = kwargs.get("negative") or "NEGATIVE_PLACEHOLDER"
        self.seed = kwargs.get("seed", 42)
        self.steps = kwargs.get("steps", 25)
        self.cfg = kwargs.get("cfg", 7.0)
        self.width = kwargs.get("width", 1024)
        self.height = kwargs.get("height", 1024)
        self.batch_size = kwargs.get("batch_size", 1)
        self.sampler = kwargs.get("sampler") or "euler_ancestral"
        self.scheduler = kwargs.get("scheduler") or "normal"
        self.denoise = kwargs.get("denoise", 1.0)
        self.refiner_denoise = kwargs.get("refiner_denoise", 0.45)
        self.input_image = kwargs.get("input_image") or "INPUT_IMAGE_PLACEHOLDER"
        self.mask_image = kwargs.get("mask_image") or "INPUT_MASK_PLACEHOLDER"
        self.controlnet = kwargs.get("controlnet") or "CONTROLNET_MODEL_PLACEHOLDER"
        self.lora = kwargs.get("lora") or "LORA_PLACEHOLDER"
        self.lora_strength_model = kwargs.get("lora_strength_model", 0.8)
        self.lora_strength_clip = kwargs.get("lora_strength_clip", 0.8)
        self.control_strength = kwargs.get("control_strength", 1.0)
        self.filename_prefix = kwargs.get("filename_prefix") or "ComfyUI"
        self.upscale_width = kwargs.get("upscale_width")
        self.upscale_height = kwargs.get("upscale_height")
        self.sd3_clip_l = kwargs.get("sd3_clip_l") or "sdv3/clip_l.safetensors"
        self.sd3_clip_g = kwargs.get("sd3_clip_g") or "sdv3/clip_g.safetensors"
        self.sd3_t5xxl = kwargs.get("sd3_t5xxl") or "sdv3/t5xxl_fp16.safetensors"
        self.sd3_shift = kwargs.get("sd3_shift", 3.0)
        self.sd3_empty_padding = kwargs.get("sd3_empty_padding") or "empty_prompt"
        self.flux_unet = kwargs.get("flux_unet") or kwargs.get("model") or "FLUX_MODEL_PLACEHOLDER"
        self.flux_clip_l = kwargs.get("flux_clip_l") or "clip_l.safetensors"
        self.flux_t5xxl = kwargs.get("flux_t5xxl") or "t5xxl_fp16.safetensors"
        self.flux_vae = kwargs.get("flux_vae") or "ae.safetensors"
        self.flux_weight_dtype = kwargs.get("flux_weight_dtype") or "default"
        self.flux_guidance = kwargs.get("flux_guidance", 3.5)
        self.flux_max_shift = kwargs.get("flux_max_shift", 1.15)
        self.flux_base_shift = kwargs.get("flux_base_shift", 0.5)


# ---------------------------------------------------------------------------
# Node library
# ---------------------------------------------------------------------------


def checkpoint_loader(graph, ckpt_name):
    node = graph.add("CheckpointLoaderSimple", {"ckpt_name": ckpt_name})
    return link(node, 0), link(node, 1), link(node, 2)


def lora_loader(graph, model_ref, clip_ref, lora_name, strength_model=0.8, strength_clip=0.8):
    node = graph.add("LoraLoader", {
        "lora_name": lora_name,
        "strength_model": strength_model,
        "strength_clip": strength_clip,
        "model": model_ref,
        "clip": clip_ref,
    })
    return link(node, 0), link(node, 1)


def clip_text(graph, text, clip_ref):
    return graph.add("CLIPTextEncode", {"text": text, "clip": clip_ref})


def triple_clip_loader(graph, clip_l, clip_g, t5xxl):
    # ComfyUI's SD3 recipe expects clip-l, clip-g, then T5.
    return graph.add("TripleCLIPLoader", {
        "clip_name1": clip_l,
        "clip_name2": clip_g,
        "clip_name3": t5xxl,
    })


def dual_clip_loader(graph, clip_l, t5xxl, clip_type="flux"):
    return graph.add("DualCLIPLoader", {
        "clip_name1": clip_l,
        "clip_name2": t5xxl,
        "type": clip_type,
    })


def clip_text_sd3(graph, text, clip_ref, empty_padding="empty_prompt"):
    return graph.add("CLIPTextEncodeSD3", {
        "clip": clip_ref,
        "clip_l": text,
        "clip_g": text,
        "t5xxl": text,
        "empty_padding": empty_padding,
    })


def clip_text_flux(graph, text, clip_ref, guidance=3.5):
    return graph.add("CLIPTextEncodeFlux", {
        "clip": clip_ref,
        "clip_l": text,
        "t5xxl": text,
        "guidance": guidance,
    })


def empty_latent(graph, width, height, batch_size=1):
    return graph.add("EmptyLatentImage", {
        "width": width,
        "height": height,
        "batch_size": batch_size,
    })


def empty_sd3_latent(graph, width, height, batch_size=1):
    return graph.add("EmptySD3LatentImage", {
        "width": width,
        "height": height,
        "batch_size": batch_size,
    })


def model_sampling_sd3(graph, model_ref, shift=3.0):
    return graph.add("ModelSamplingSD3", {
        "model": model_ref,
        "shift": shift,
    })


def unet_loader(graph, unet_name, weight_dtype="default"):
    return graph.add("UNETLoader", {
        "unet_name": unet_name,
        "weight_dtype": weight_dtype,
    })


def vae_loader(graph, vae_name):
    return graph.add("VAELoader", {"vae_name": vae_name})


def model_sampling_flux(graph, model_ref, width, height, max_shift=1.15, base_shift=0.5):
    return graph.add("ModelSamplingFlux", {
        "model": model_ref,
        "max_shift": max_shift,
        "base_shift": base_shift,
        "width": width,
        "height": height,
    })


def load_image(graph, image_name):
    return graph.add("LoadImage", {"image": image_name})


def load_mask(graph, mask_name, channel="red"):
    return graph.add("LoadImageMask", {"image": mask_name, "channel": channel})


def vae_encode(graph, image_ref, vae_ref):
    return graph.add("VAEEncode", {"pixels": image_ref, "vae": vae_ref})


def set_latent_noise_mask(graph, latent_ref, mask_ref):
    return graph.add("SetLatentNoiseMask", {"samples": latent_ref, "mask": mask_ref})


def controlnet_loader(graph, controlnet_name):
    return graph.add("ControlNetLoader", {"control_net_name": controlnet_name})


def controlnet_apply(graph, conditioning_ref, controlnet_ref, image_ref, strength=1.0):
    return graph.add("ControlNetApply", {
        "conditioning": conditioning_ref,
        "control_net": controlnet_ref,
        "image": image_ref,
        "strength": strength,
    })


def ksampler(graph, model_ref, positive_ref, negative_ref, latent_ref, params, denoise=None):
    return graph.add("KSampler", {
        "seed": params.seed,
        "steps": params.steps,
        "cfg": params.cfg,
        "sampler_name": params.sampler,
        "scheduler": params.scheduler,
        "denoise": params.denoise if denoise is None else denoise,
        "model": model_ref,
        "positive": positive_ref,
        "negative": negative_ref,
        "latent_image": latent_ref,
    })


def latent_upscale(graph, latent_ref, width, height, method="nearest-exact"):
    return graph.add("LatentUpscale", {
        "upscale_method": method,
        "width": width,
        "height": height,
        "crop": "disabled",
        "samples": latent_ref,
    })


def vae_decode(graph, samples_ref, vae_ref):
    return graph.add("VAEDecode", {"samples": samples_ref, "vae": vae_ref})


def save_image(graph, images_ref, prefix):
    return graph.add("SaveImage", {"filename_prefix": prefix, "images": images_ref})


def model_stack(graph, params, use_lora=False):
    model_ref, clip_ref, vae_ref = checkpoint_loader(graph, params.model)
    if use_lora:
        model_ref, clip_ref = lora_loader(
            graph,
            model_ref,
            clip_ref,
            params.lora,
            params.lora_strength_model,
            params.lora_strength_clip,
        )
    return model_ref, clip_ref, vae_ref


# ---------------------------------------------------------------------------
# Workflow blueprints
# ---------------------------------------------------------------------------


def build_txt2img(params, use_lora=False, use_controlnet=False, prefix=None):
    graph = WorkflowGraph()
    model_ref, clip_ref, vae_ref = model_stack(graph, params, use_lora=use_lora)
    positive_node = clip_text(graph, params.positive, clip_ref)
    negative_node = clip_text(graph, params.negative, clip_ref)

    positive_ref = link(positive_node)
    if use_controlnet:
        control_image = load_image(graph, params.input_image)
        controlnet = controlnet_loader(graph, params.controlnet)
        positive_ref = link(controlnet_apply(
            graph,
            positive_ref,
            link(controlnet),
            link(control_image),
            params.control_strength,
        ))

    latent = empty_latent(graph, params.width, params.height, params.batch_size)
    sampled = ksampler(graph, model_ref, positive_ref, link(negative_node), link(latent), params)
    decoded = vae_decode(graph, link(sampled), vae_ref)
    save_image(graph, link(decoded), prefix or params.filename_prefix)
    return graph.export()


def build_img2img(params, use_lora=False, use_controlnet=False, prefix=None):
    graph = WorkflowGraph()
    model_ref, clip_ref, vae_ref = model_stack(graph, params, use_lora=use_lora)
    positive_node = clip_text(graph, params.positive, clip_ref)
    negative_node = clip_text(graph, params.negative, clip_ref)
    image_node = load_image(graph, params.input_image)
    latent = vae_encode(graph, link(image_node), vae_ref)

    positive_ref = link(positive_node)
    if use_controlnet:
        controlnet = controlnet_loader(graph, params.controlnet)
        positive_ref = link(controlnet_apply(
            graph,
            positive_ref,
            link(controlnet),
            link(image_node),
            params.control_strength,
        ))

    sampled = ksampler(
        graph,
        model_ref,
        positive_ref,
        link(negative_node),
        link(latent),
        params,
        denoise=params.denoise if params.denoise != 1.0 else 0.75,
    )
    decoded = vae_decode(graph, link(sampled), vae_ref)
    save_image(graph, link(decoded), prefix or params.filename_prefix)
    return graph.export()


def build_inpainting(params, use_lora=False, use_controlnet=False, prefix=None):
    graph = WorkflowGraph()
    model_ref, clip_ref, vae_ref = model_stack(graph, params, use_lora=use_lora)
    positive_node = clip_text(graph, params.positive, clip_ref)
    negative_node = clip_text(graph, params.negative, clip_ref)
    image_node = load_image(graph, params.input_image)
    mask_node = load_mask(graph, params.mask_image)
    encoded = vae_encode(graph, link(image_node), vae_ref)
    masked_latent = set_latent_noise_mask(graph, link(encoded), link(mask_node))

    positive_ref = link(positive_node)
    if use_controlnet:
        controlnet = controlnet_loader(graph, params.controlnet)
        positive_ref = link(controlnet_apply(
            graph,
            positive_ref,
            link(controlnet),
            link(image_node),
            params.control_strength,
        ))

    sampled = ksampler(graph, model_ref, positive_ref, link(negative_node), link(masked_latent), params)
    decoded = vae_decode(graph, link(sampled), vae_ref)
    save_image(graph, link(decoded), prefix or params.filename_prefix)
    return graph.export()


def build_hires_txt2img(params, use_lora=False, prefix=None):
    graph = WorkflowGraph()
    model_ref, clip_ref, vae_ref = model_stack(graph, params, use_lora=use_lora)
    positive_node = clip_text(graph, params.positive, clip_ref)
    negative_node = clip_text(graph, params.negative, clip_ref)
    latent = empty_latent(graph, params.width, params.height, params.batch_size)
    first = ksampler(graph, model_ref, link(positive_node), link(negative_node), link(latent), params)
    up_width = params.upscale_width or params.width * 2
    up_height = params.upscale_height or params.height * 2
    upscaled = latent_upscale(graph, link(first), up_width, up_height)
    second = ksampler(
        graph,
        model_ref,
        link(positive_node),
        link(negative_node),
        link(upscaled),
        params,
        denoise=params.refiner_denoise,
    )
    decoded = vae_decode(graph, link(second), vae_ref)
    save_image(graph, link(decoded), prefix or params.filename_prefix)
    return graph.export()


def build_latent_upscale(params, use_lora=False, prefix=None):
    graph = WorkflowGraph()
    model_ref, clip_ref, vae_ref = model_stack(graph, params, use_lora=use_lora)
    positive_node = clip_text(graph, params.positive, clip_ref)
    negative_node = clip_text(graph, params.negative, clip_ref)
    image_node = load_image(graph, params.input_image)
    encoded = vae_encode(graph, link(image_node), vae_ref)
    up_width = params.upscale_width or params.width * 2
    up_height = params.upscale_height or params.height * 2
    upscaled = latent_upscale(graph, link(encoded), up_width, up_height)
    sampled = ksampler(
        graph,
        model_ref,
        link(positive_node),
        link(negative_node),
        link(upscaled),
        params,
        denoise=params.denoise if params.denoise != 1.0 else 0.5,
    )
    decoded = vae_decode(graph, link(sampled), vae_ref)
    save_image(graph, link(decoded), prefix or params.filename_prefix)
    return graph.export()


def build_sd3_txt2img(params, prefix=None):
    graph = WorkflowGraph()
    model_ref, _checkpoint_clip_ref, vae_ref = checkpoint_loader(graph, params.model)
    sd3_model = model_sampling_sd3(graph, model_ref, params.sd3_shift)
    clip_node = triple_clip_loader(graph, params.sd3_clip_l, params.sd3_clip_g, params.sd3_t5xxl)
    positive_node = clip_text_sd3(graph, params.positive, link(clip_node), params.sd3_empty_padding)
    negative_node = clip_text_sd3(graph, params.negative, link(clip_node), params.sd3_empty_padding)
    latent = empty_sd3_latent(graph, params.width, params.height, params.batch_size)
    sampled = ksampler(graph, link(sd3_model), link(positive_node), link(negative_node), link(latent), params)
    decoded = vae_decode(graph, link(sampled), vae_ref)
    save_image(graph, link(decoded), prefix or params.filename_prefix)
    return graph.export()


def build_flux_txt2img(params, prefix=None):
    graph = WorkflowGraph()
    model = unet_loader(graph, params.flux_unet, params.flux_weight_dtype)
    flux_model = model_sampling_flux(
        graph,
        link(model),
        params.width,
        params.height,
        params.flux_max_shift,
        params.flux_base_shift,
    )
    clip_node = dual_clip_loader(graph, params.flux_clip_l, params.flux_t5xxl, "flux")
    positive_node = clip_text_flux(graph, params.positive, link(clip_node), params.flux_guidance)
    negative_node = clip_text_flux(graph, params.negative, link(clip_node), params.flux_guidance)
    latent = empty_latent(graph, params.width, params.height, params.batch_size)
    sampled = ksampler(graph, link(flux_model), link(positive_node), link(negative_node), link(latent), params)
    vae = vae_loader(graph, params.flux_vae)
    decoded = vae_decode(graph, link(sampled), link(vae))
    save_image(graph, link(decoded), prefix or params.filename_prefix)
    return graph.export()


def _workflow_builder(name):
    builders = {
        "txt2img": lambda p: build_txt2img(p, prefix="ComfyUI_txt2img"),
        "sd3_txt2img": lambda p: build_sd3_txt2img(
            _with_sd3_defaults(p),
            prefix="ComfyUI_sd3_txt2img",
        ),
        "flux_txt2img": lambda p: build_flux_txt2img(
            _with_flux_defaults(p),
            prefix="ComfyUI_flux_txt2img",
        ),
        "txt2img_lora": lambda p: build_txt2img(p, use_lora=True, prefix="ComfyUI_txt2img_lora"),
        "img2img": lambda p: build_img2img(p, prefix="ComfyUI_img2img"),
        "img2img_lora": lambda p: build_img2img(p, use_lora=True, prefix="ComfyUI_img2img_lora"),
        "inpainting": lambda p: build_inpainting(p, prefix="ComfyUI_inpaint"),
        "inpainting_lora": lambda p: build_inpainting(p, use_lora=True, prefix="ComfyUI_inpaint_lora"),
        "controlnet_txt2img": lambda p: build_txt2img(p, use_controlnet=True, prefix="ComfyUI_controlnet"),
        "controlnet_img2img": lambda p: build_img2img(p, use_controlnet=True, prefix="ComfyUI_controlnet_img2img"),
        "controlnet_inpaint": lambda p: build_inpainting(p, use_controlnet=True, prefix="ComfyUI_controlnet_inpaint"),
        "hires_txt2img": lambda p: build_hires_txt2img(p, prefix="ComfyUI_hires"),
        "hires_txt2img_lora": lambda p: build_hires_txt2img(p, use_lora=True, prefix="ComfyUI_hires_lora"),
        "latent_upscale": lambda p: build_latent_upscale(p, prefix="ComfyUI_upscale"),
        "ecommerce_product": lambda p: build_txt2img(p, use_lora=True, prefix="ComfyUI_ecommerce_product"),
        "ecommerce_model": lambda p: build_img2img(p, use_lora=True, prefix="ComfyUI_ecommerce_model"),
        "ecommerce_bg_replace": lambda p: build_img2img(p, use_lora=True, prefix="ComfyUI_ecommerce_bg_replace"),
        "ecommerce_tryon": lambda p: build_txt2img(
            _with_dimensions(p, width=1024, height=1536),
            use_controlnet=True,
            prefix="ComfyUI_ecommerce_tryon",
        ),
    }
    return builders[name]


def _with_dimensions(params, width, height):
    clone = BuildParams(**params.__dict__)
    clone.width = width
    clone.height = height
    return clone


def _with_sd3_defaults(params):
    clone = BuildParams(**params.__dict__)
    if clone.sampler == "euler_ancestral":
        clone.sampler = "dpmpp_2m"
    if clone.scheduler == "normal":
        clone.scheduler = "sgm_uniform"
    return clone


def _with_flux_defaults(params):
    clone = BuildParams(**params.__dict__)
    if clone.sampler == "euler_ancestral":
        clone.sampler = "euler"
    if clone.scheduler == "normal":
        clone.scheduler = "simple"
    if clone.cfg == 7.0:
        clone.cfg = 1.0
    if clone.negative == "NEGATIVE_PLACEHOLDER":
        clone.negative = ""
    return clone


WORKFLOW_BLUEPRINTS = {
    "txt2img": {
        "tier": "basic",
        "label": "基础文生图",
        "description": "Checkpoint + prompt + latent + KSampler + decode",
        "families": ["sd", "sd15", "sd2", "sdxl"],
        "image_types": ["all"],
        "requires": ["model", "positive", "negative"],
        "optional": [],
    },
    "sd3_txt2img": {
        "tier": "basic",
        "label": "SD3/SD3.5 文生图",
        "description": "Checkpoint + TripleCLIPLoader + CLIPTextEncodeSD3 + EmptySD3LatentImage",
        "families": ["sd3", "sd35"],
        "image_types": ["all"],
        "requires": ["model", "positive", "negative", "sd3_clip_l", "sd3_clip_g", "sd3_t5xxl"],
        "optional": ["sd3_shift"],
    },
    "flux_txt2img": {
        "tier": "basic",
        "label": "FLUX 文生图",
        "description": "UNETLoader + DualCLIPLoader + CLIPTextEncodeFlux + VAELoader",
        "families": ["flux"],
        "image_types": ["all"],
        "requires": ["flux_unet", "positive", "negative", "flux_clip_l", "flux_t5xxl", "flux_vae"],
        "optional": ["flux_guidance", "flux_weight_dtype", "flux_max_shift", "flux_base_shift"],
    },
    "txt2img_lora": {
        "tier": "basic",
        "label": "文生图 + LoRA",
        "description": "基础文生图并插入 LoraLoader",
        "families": ["sd", "sd15", "sd2", "sdxl"],
        "image_types": ["all"],
        "requires": ["model", "positive", "negative", "lora"],
        "optional": [],
    },
    "img2img": {
        "tier": "image",
        "label": "图生图",
        "description": "LoadImage + VAEEncode + KSampler",
        "families": ["sd", "sd15", "sd2", "sdxl"],
        "image_types": ["all"],
        "requires": ["model", "positive", "negative", "input_image"],
        "optional": ["denoise"],
    },
    "img2img_lora": {
        "tier": "image",
        "label": "图生图 + LoRA",
        "description": "图生图并插入 LoraLoader",
        "families": ["sd", "sd15", "sd2", "sdxl"],
        "image_types": ["all"],
        "requires": ["model", "positive", "negative", "input_image", "lora"],
        "optional": ["denoise"],
    },
    "inpainting": {
        "tier": "image",
        "label": "局部重绘",
        "description": "LoadImage + LoadImageMask + SetLatentNoiseMask",
        "families": ["sd", "sd15", "sd2", "sdxl"],
        "image_types": ["realistic", "portrait", "ecommerce_product", "architecture", "all"],
        "requires": ["model", "positive", "negative", "input_image", "mask_image"],
        "optional": ["denoise"],
    },
    "inpainting_lora": {
        "tier": "image",
        "label": "局部重绘 + LoRA",
        "description": "局部重绘并插入 LoraLoader",
        "families": ["sd", "sd15", "sd2", "sdxl"],
        "image_types": ["realistic", "portrait", "ecommerce_product", "architecture", "all"],
        "requires": ["model", "positive", "negative", "input_image", "mask_image", "lora"],
        "optional": ["denoise"],
    },
    "controlnet_txt2img": {
        "tier": "control",
        "label": "ControlNet 文生图",
        "description": "ControlNetApply 接入正向 conditioning",
        "families": ["sd", "sd15", "sd2", "sdxl"],
        "image_types": ["anime", "landscape", "architecture", "ecommerce_product", "portrait", "all"],
        "requires": ["model", "positive", "negative", "input_image", "controlnet"],
        "optional": ["control_strength"],
    },
    "controlnet_img2img": {
        "tier": "control",
        "label": "ControlNet 图生图",
        "description": "图生图并使用输入图作为 ControlNet 参考",
        "families": ["sd", "sd15", "sd2", "sdxl"],
        "image_types": ["realistic", "portrait", "architecture", "ecommerce_product", "all"],
        "requires": ["model", "positive", "negative", "input_image", "controlnet"],
        "optional": ["denoise", "control_strength"],
    },
    "controlnet_inpaint": {
        "tier": "control",
        "label": "ControlNet 局部重绘",
        "description": "局部重绘叠加 ControlNet 结构控制",
        "families": ["sd", "sd15", "sd2", "sdxl"],
        "image_types": ["realistic", "portrait", "architecture", "ecommerce_product", "all"],
        "requires": ["model", "positive", "negative", "input_image", "mask_image", "controlnet"],
        "optional": ["denoise", "control_strength"],
    },
    "hires_txt2img": {
        "tier": "advanced",
        "label": "两阶段高清修复",
        "description": "第一轮采样后 latent upscale，再低 denoise 二次采样",
        "families": ["sd", "sd15", "sd2", "sdxl"],
        "image_types": ["all"],
        "requires": ["model", "positive", "negative"],
        "optional": ["upscale_width", "upscale_height", "refiner_denoise"],
    },
    "hires_txt2img_lora": {
        "tier": "advanced",
        "label": "两阶段高清修复 + LoRA",
        "description": "高清修复并插入 LoraLoader",
        "families": ["sd", "sd15", "sd2", "sdxl"],
        "image_types": ["all"],
        "requires": ["model", "positive", "negative", "lora"],
        "optional": ["upscale_width", "upscale_height", "refiner_denoise"],
    },
    "latent_upscale": {
        "tier": "advanced",
        "label": "潜空间放大重采样",
        "description": "输入图编码到 latent 后放大并重采样",
        "families": ["sd", "sd15", "sd2", "sdxl"],
        "image_types": ["all"],
        "requires": ["model", "positive", "negative", "input_image"],
        "optional": ["upscale_width", "upscale_height", "denoise"],
    },
    "ecommerce_product": {
        "tier": "ecommerce",
        "label": "电商商品图",
        "description": "商品图文生图，默认插入 LoRA 节点指针",
        "families": ["sd", "sd15", "sd2", "sdxl"],
        "image_types": ["ecommerce_product"],
        "requires": ["model", "positive", "negative", "lora"],
        "optional": [],
    },
    "ecommerce_model": {
        "tier": "ecommerce",
        "label": "电商模特图",
        "description": "输入模特图进行图生图编辑",
        "families": ["sd", "sd15", "sd2", "sdxl"],
        "image_types": ["portrait", "ecommerce_model"],
        "requires": ["model", "positive", "negative", "input_image", "lora"],
        "optional": ["denoise"],
    },
    "ecommerce_bg_replace": {
        "tier": "ecommerce",
        "label": "电商换背景",
        "description": "基于商品图进行背景替换/重绘",
        "families": ["sd", "sd15", "sd2", "sdxl"],
        "image_types": ["ecommerce_product"],
        "requires": ["model", "positive", "negative", "input_image", "lora"],
        "optional": ["denoise"],
    },
    "ecommerce_tryon": {
        "tier": "ecommerce",
        "label": "虚拟试穿 ControlNet",
        "description": "竖图尺寸 + ControlNet 姿态/结构控制",
        "families": ["sd", "sd15", "sd2", "sdxl"],
        "image_types": ["portrait", "ecommerce_model"],
        "requires": ["model", "positive", "negative", "input_image", "controlnet"],
        "optional": ["control_strength"],
    },
}


for _workflow_id in WORKFLOW_BLUEPRINTS:
    WORKFLOW_BLUEPRINTS[_workflow_id]["id"] = _workflow_id


# ---------------------------------------------------------------------------
# Menu and validation
# ---------------------------------------------------------------------------


def infer_model_family(model_name=None, explicit_family=None):
    if explicit_family and explicit_family != "auto":
        return explicit_family
    if not model_name:
        return "sdxl"

    name = model_name.lower()
    if "flux" in name:
        return "flux"
    if any(term in name for term in ("sd3.5", "sd35", "stable-diffusion-3.5")):
        return "sd35"
    if any(term in name for term in ("sd3", "sd_3", "stable-diffusion-3")):
        return "sd3"
    if any(term in name for term in ("sd2", "sd2.1", "sd_2", "2.1", "v2-1", "768-v")):
        return "sd2"
    sd15_terms = [
        "sd15", "sd1.5", "1_5", "v1-5", "anything", "counterfeit",
        "meina", "revanimated", "rev_animated", "dreamshaper_8",
    ]
    if any(term in name for term in sd15_terms):
        return "sd15"
    sdxl_terms = ["xl", "sdxl", "pony", "juggernaut", "realvisxl", "animagine"]
    if any(term in name for term in sdxl_terms):
        return "sdxl"
    if any(term in name for term in ("stable-diffusion", "sd_", "sd-")):
        return "sd"
    return "sdxl"


def workflow_matches_image_type(blueprint, image_type):
    normalized = normalize_image_type(image_type) if image_type else "all"
    image_types = blueprint.get("image_types", ["all"])
    return "all" in image_types or normalized in image_types


def available_workflows(model_family="sdxl", image_type=None):
    results = []
    for workflow_id, blueprint in WORKFLOW_BLUEPRINTS.items():
        if model_family not in blueprint.get("families", []):
            continue
        if not workflow_matches_image_type(blueprint, image_type):
            continue
        results.append({k: v for k, v in blueprint.items() if k != "builder"})
    return results


def build_menu(model_family="sdxl", image_type=None, model_name=None):
    family = infer_model_family(model_name, model_family)
    workflows = available_workflows(family, image_type)
    tiers = []
    for tier_id, tier_label in TIERS.items():
        items = [workflow for workflow in workflows if workflow["tier"] == tier_id]
        if items:
            tiers.append({"id": tier_id, "label": tier_label, "workflows": items})

    unsupported = []
    if family == "flux":
        unsupported.append(
            "FLUX 工作流需要独立 diffusion model、clip_l、t5xxl 和 VAE 文件。缺少这些文件时请先放入 ComfyUI/models/diffusion_models、models/text_encoders 和 models/vae。"
        )
    elif family in ("sd3", "sd35"):
        unsupported.append(
            "SD3/SD3.5 工作流需要独立 text encoders：clip_l、clip_g、t5xxl。缺少这些文件时请先放入 ComfyUI/models/text_encoders。"
        )

    return {
        "model_family": family,
        "model_name": model_name,
        "image_type": normalize_image_type(image_type) if image_type else None,
        "tiers": tiers,
        "unsupported_notes": unsupported,
    }


def find_placeholders(workflow):
    placeholders = []

    def walk(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
        elif isinstance(value, str) and PLACEHOLDER_TOKEN in value:
            placeholders.append({"path": path, "value": value})

    walk(workflow, "")
    return placeholders


def iter_links(workflow):
    for node_id, node in workflow.items():
        for input_name, value in node.get("inputs", {}).items():
            if (
                isinstance(value, list) and len(value) == 2
                and isinstance(value[1], int)
                and str(value[0]) in workflow
            ):
                yield node_id, input_name, str(value[0]), value[1]


def upstream_reachable_from_saves(workflow):
    reverse = {}
    for node_id, _input_name, src_id, _output in iter_links(workflow):
        reverse.setdefault(node_id, set()).add(src_id)

    roots = [node_id for node_id, node in workflow.items() if node.get("class_type") == "SaveImage"]
    seen = set()
    stack = roots[:]
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        stack.extend(reverse.get(node_id, set()) - seen)
    return seen


def validate_workflow(workflow, allow_placeholders=False):
    errors = []
    warnings = []

    if not isinstance(workflow, dict):
        return {"valid": False, "errors": ["Workflow root must be a JSON object."], "warnings": []}

    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            errors.append(f"Node {node_id} must be an object.")
            continue
        class_type = node.get("class_type")
        inputs = node.get("inputs")
        if not class_type:
            errors.append(f"Node {node_id} is missing class_type.")
        if not isinstance(inputs, dict):
            errors.append(f"Node {node_id} inputs must be an object.")
            continue

        for required in KNOWN_REQUIRED_INPUTS.get(class_type, []):
            if required not in inputs:
                errors.append(f"Node {node_id} ({class_type}) missing input: {required}")

        for input_name, value in inputs.items():
            if (
                isinstance(value, list) and len(value) == 2
                and isinstance(value[1], int)
                and not isinstance(value[0], (dict, list))
            ):
                source_id = str(value[0])
                if source_id not in workflow:
                    errors.append(
                        f"Node {node_id}.{input_name} links to missing node {source_id}."
                    )
                if value[1] < 0:
                    errors.append(f"Node {node_id}.{input_name} has negative output index.")

    placeholders = find_placeholders(workflow)
    if placeholders and not allow_placeholders:
        for item in placeholders:
            errors.append(f"Unresolved placeholder at {item['path']}: {item['value']}")

    save_nodes = [node_id for node_id, node in workflow.items() if node.get("class_type") == "SaveImage"]
    if not save_nodes:
        errors.append("Workflow has no SaveImage node.")

    reachable = upstream_reachable_from_saves(workflow)
    if reachable:
        unused = sorted(set(workflow) - reachable, key=lambda x: int(x) if x.isdigit() else x)
        if unused:
            warnings.append(f"Unused nodes not connected to SaveImage: {', '.join(unused)}")

    return {"valid": not errors, "errors": errors, "warnings": warnings}


def build_workflow(workflow_id, params):
    if workflow_id not in WORKFLOW_BLUEPRINTS:
        raise ValueError(f"Unknown workflow id: {workflow_id}")
    return _workflow_builder(workflow_id)(params)


def print_menu(menu):
    print(f"Model family: {menu['model_family']}")
    if menu.get("image_type"):
        print(f"Image type: {menu['image_type']}")
    for note in menu.get("unsupported_notes", []):
        print(f"NOTE: {note}")
    for tier in menu["tiers"]:
        print(f"\n[{tier['label']}]")
        for workflow in tier["workflows"]:
            requires = ", ".join(workflow.get("requires", []))
            print(f"  {workflow['id']}: {workflow['label']} | requires: {requires}")


def load_workflow(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, output_path=None):
    text = json.dumps(data, indent=2, ensure_ascii=False)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    else:
        print(text)


def params_from_args(args):
    return BuildParams(
        model=args.model,
        positive=args.positive,
        negative=args.negative,
        seed=args.seed,
        steps=args.steps,
        cfg=args.cfg,
        width=args.width,
        height=args.height,
        batch_size=args.batch_size,
        sampler=args.sampler,
        scheduler=args.scheduler,
        denoise=args.denoise,
        refiner_denoise=args.refiner_denoise,
        input_image=args.input_image,
        mask_image=args.mask_image,
        controlnet=args.controlnet,
        lora=args.lora,
        lora_strength_model=args.lora_strength_model,
        lora_strength_clip=args.lora_strength_clip,
        control_strength=args.control_strength,
        filename_prefix=args.filename_prefix,
        upscale_width=args.upscale_width,
        upscale_height=args.upscale_height,
        sd3_clip_l=args.sd3_clip_l,
        sd3_clip_g=args.sd3_clip_g,
        sd3_t5xxl=args.sd3_t5xxl,
        sd3_shift=args.sd3_shift,
        sd3_empty_padding=args.sd3_empty_padding,
        flux_unet=args.flux_unet,
        flux_clip_l=args.flux_clip_l,
        flux_t5xxl=args.flux_t5xxl,
        flux_vae=args.flux_vae,
        flux_weight_dtype=args.flux_weight_dtype,
        flux_guidance=args.flux_guidance,
        flux_max_shift=args.flux_max_shift,
        flux_base_shift=args.flux_base_shift,
    )


def add_build_args(parser):
    parser.add_argument("--model", help="Checkpoint filename")
    parser.add_argument("--positive", help="Positive prompt")
    parser.add_argument("--negative", help="Negative prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=7.0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sampler", default="euler_ancestral")
    parser.add_argument("--scheduler", default="normal")
    parser.add_argument("--denoise", type=float, default=1.0)
    parser.add_argument("--refiner-denoise", type=float, default=0.45)
    parser.add_argument("--input-image", help="ComfyUI input filename")
    parser.add_argument("--mask-image", help="ComfyUI mask filename")
    parser.add_argument("--controlnet", help="ControlNet model filename")
    parser.add_argument("--control-strength", type=float, default=1.0)
    parser.add_argument("--lora", help="LoRA filename")
    parser.add_argument("--lora-strength-model", type=float, default=0.8)
    parser.add_argument("--lora-strength-clip", type=float, default=0.8)
    parser.add_argument("--filename-prefix", default="ComfyUI")
    parser.add_argument("--upscale-width", type=int)
    parser.add_argument("--upscale-height", type=int)
    parser.add_argument("--sd3-clip-l", default="sdv3/clip_l.safetensors",
                        help="SD3/SD3.5 CLIP-L filename under ComfyUI models/text_encoders")
    parser.add_argument("--sd3-clip-g", default="sdv3/clip_g.safetensors",
                        help="SD3/SD3.5 CLIP-G filename under ComfyUI models/text_encoders")
    parser.add_argument("--sd3-t5xxl", default="sdv3/t5xxl_fp16.safetensors",
                        help="SD3/SD3.5 T5 XXL filename under ComfyUI models/text_encoders")
    parser.add_argument("--sd3-shift", type=float, default=3.0,
                        help="Shift value for ModelSamplingSD3")
    parser.add_argument("--sd3-empty-padding", default="empty_prompt", choices=["none", "empty_prompt"],
                        help="Empty padding mode for CLIPTextEncodeSD3")
    parser.add_argument("--flux-unet", help="FLUX diffusion model filename under ComfyUI models/diffusion_models")
    parser.add_argument("--flux-clip-l", default="clip_l.safetensors",
                        help="FLUX CLIP-L filename under ComfyUI models/text_encoders")
    parser.add_argument("--flux-t5xxl", default="t5xxl_fp16.safetensors",
                        help="FLUX T5 XXL filename under ComfyUI models/text_encoders")
    parser.add_argument("--flux-vae", default="ae.safetensors",
                        help="FLUX VAE filename under ComfyUI models/vae")
    parser.add_argument("--flux-weight-dtype", default="default",
                        choices=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                        help="Weight dtype for UNETLoader")
    parser.add_argument("--flux-guidance", type=float, default=3.5,
                        help="Guidance value for CLIPTextEncodeFlux")
    parser.add_argument("--flux-max-shift", type=float, default=1.15,
                        help="Max shift for ModelSamplingFlux")
    parser.add_argument("--flux-base-shift", type=float, default=0.5,
                        help="Base shift for ModelSamplingFlux")


def main():
    parser = argparse.ArgumentParser(description="Dynamic ComfyUI workflow builder")
    subparsers = parser.add_subparsers(dest="command")

    menu_parser = subparsers.add_parser("menu", help="Show model-aware TUI workflow menu")
    menu_parser.add_argument("--model-family", default="auto", choices=["auto", "sd", "sd15", "sd2", "sdxl", "sd3", "sd35", "flux"])
    menu_parser.add_argument("--model", help="Selected checkpoint name for family inference")
    menu_parser.add_argument("--image-type", help="Selected image type")
    menu_parser.add_argument("--json", action="store_true", dest="json_output")

    ids_parser = subparsers.add_parser("ids", help="List dynamic workflow ids")
    ids_parser.add_argument("--json", action="store_true", dest="json_output")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect one workflow blueprint")
    inspect_parser.add_argument("workflow_id", choices=sorted(WORKFLOW_BLUEPRINTS.keys()))
    inspect_parser.add_argument("--json", action="store_true", dest="json_output")

    build_parser = subparsers.add_parser("build", help="Build workflow JSON from node pointers")
    build_parser.add_argument("--workflow-id", required=True, choices=sorted(WORKFLOW_BLUEPRINTS.keys()))
    build_parser.add_argument("--output", help="Output JSON path")
    build_parser.add_argument("--allow-placeholders", action="store_true")
    add_build_args(build_parser)

    validate_parser = subparsers.add_parser("validate", help="Validate workflow JSON")
    validate_parser.add_argument("--workflow", required=True, help="Workflow JSON path")
    validate_parser.add_argument("--allow-placeholders", action="store_true")
    validate_parser.add_argument("--json", action="store_true", dest="json_output")

    args = parser.parse_args()

    if args.command == "menu":
        menu = build_menu(args.model_family, args.image_type, args.model)
        if args.json_output:
            write_json(menu)
        else:
            print_menu(menu)

    elif args.command == "ids":
        workflows = list(WORKFLOW_BLUEPRINTS.values())
        if args.json_output:
            write_json(workflows)
        else:
            for workflow in workflows:
                print(f"{workflow['id']}: {workflow['label']} ({TIERS[workflow['tier']]})")

    elif args.command == "inspect":
        blueprint = WORKFLOW_BLUEPRINTS[args.workflow_id]
        if args.json_output:
            write_json(blueprint)
        else:
            print(f"{blueprint['id']}: {blueprint['label']}")
            print(f"Tier: {TIERS[blueprint['tier']]}")
            print(f"Description: {blueprint['description']}")
            print(f"Families: {', '.join(blueprint['families'])}")
            print(f"Requires: {', '.join(blueprint['requires'])}")
            print(f"Optional: {', '.join(blueprint['optional']) or 'none'}")

    elif args.command == "build":
        workflow = build_workflow(args.workflow_id, params_from_args(args))
        validation = validate_workflow(workflow, allow_placeholders=args.allow_placeholders)
        if not validation["valid"]:
            for error in validation["errors"]:
                print(f"ERROR: {error}", file=sys.stderr)
            sys.exit(1)
        write_json(workflow, args.output)
        if args.output:
            print(f"Built workflow: {args.output}")

    elif args.command == "validate":
        workflow = load_workflow(args.workflow)
        result = validate_workflow(workflow, allow_placeholders=args.allow_placeholders)
        if args.json_output:
            write_json(result)
        else:
            print(f"Valid: {result['valid']}")
            for error in result["errors"]:
                print(f"ERROR: {error}")
            for warning in result["warnings"]:
                print(f"WARNING: {warning}")
        if not result["valid"]:
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
