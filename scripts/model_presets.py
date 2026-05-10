#!/usr/bin/env python3
"""
Model Presets - Curated model recommendations for different use cases
Includes checkpoints, LoRAs, VAEs, and other fine-tuned models
Organized by task type and model family (SDXL, SD1.5, FLUX, etc.)
"""

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
import sys
from pathlib import Path

# ============================================================
# Checkpoint Model Presets
# ============================================================

CHECKPOINT_PRESETS = {
    # --- SDXL Checkpoints ---
    "sdxl-base": {
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "file": "sd_xl_base_1.0.safetensors",
        "family": "sdxl",
        "desc": "SDXL Base 1.0 - 通用基础模型，适合大多数场景",
        "best_for": ["通用", "电商", "风景", "人物"],
        "resolution": "1024x1024",
        "steps": 25,
        "cfg": 7,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
    },
    "sdxl-refiner": {
        "repo": "stabilityai/stable-diffusion-xl-refiner-1.0",
        "file": "sd_xl_refiner_1.0.safetensors",
        "family": "sdxl",
        "desc": "SDXL Refiner - 精细化模型，配合Base使用提升细节",
        "best_for": ["细节增强", "高质量输出"],
        "resolution": "1024x1024",
        "steps": 25,
        "cfg": 7,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
    },
    "dreamshaper-xl": {
        "repo": "Lykon/dreamshaper-xl-v2-turbo",
        "file": "dreamshaperXL_v21TurboDPMSDE.safetensors",
        "family": "sdxl",
        "desc": "DreamShaper XL Turbo - 快速出图，质量优秀",
        "best_for": ["快速生成", "通用", "创意"],
        "resolution": "1024x1024",
        "steps": 8,
        "cfg": 2,
        "sampler": "dpmpp_sde",
        "scheduler": "karras",
    },
    "juggernaut-xl": {
        "repo": "RunDiffusion/Juggernaut-XL-v9",
        "file": "juggernautXL_v9Rdphoto2Lightning.safetensors",
        "family": "sdxl",
        "desc": "Juggernaut XL - 照片级写实模型，电商首选",
        "best_for": ["电商", "写实照片", "产品摄影", "人物"],
        "resolution": "1024x1024",
        "steps": 20,
        "cfg": 4.5,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
    },
    "realvisxl": {
        "repo": "SG161222/RealVisXL_V4.0",
        "file": "RealVisXL_V4.0.safetensors",
        "family": "sdxl",
        "desc": "RealVis XL - 超写实模型，皮肤纹理优秀",
        "best_for": ["写实人物", "电商模特", "肖像"],
        "resolution": "1024x1024",
        "steps": 25,
        "cfg": 7,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
    },
    # --- SD 1.5 Checkpoints ---
    "sd15-base": {
        "repo": "runwayml/stable-diffusion-v1-5",
        "file": "v1-5-pruned-emaonly.safetensors",
        "family": "sd15",
        "desc": "SD 1.5 Base - 经典基础模型，兼容性最好",
        "best_for": ["通用", "LoRA兼容"],
        "resolution": "512x512",
        "steps": 25,
        "cfg": 7,
        "sampler": "euler_ancestral",
        "scheduler": "normal",
    },
    "dreamshaper-8": {
        "repo": "Lykon/DreamShaper",
        "file": "DreamShaper_8_pruned.safetensors",
        "family": "sd15",
        "desc": "DreamShaper 8 - SD1.5 最佳通用模型之一",
        "best_for": ["通用", "创意", "动漫", "写实"],
        "resolution": "512x512",
        "steps": 25,
        "cfg": 7,
        "sampler": "euler_ancestral",
        "scheduler": "normal",
    },
    "rev-animated": {
        "repo": "stablediffusionapi/rev-animated",
        "file": "rev_animated_v122.safetensors",
        "family": "sd15",
        "desc": "ReV Animated - 动漫风格优秀",
        "best_for": ["动漫", "插画", "二次元"],
        "resolution": "512x512",
        "steps": 25,
        "cfg": 7,
        "sampler": "euler_ancestral",
        "scheduler": "normal",
    },
    # --- FLUX Checkpoints ---
    "flux-dev": {
        "repo": "black-forest-labs/FLUX.1-dev",
        "file": "flux1-dev.safetensors",
        "family": "flux",
        "desc": "FLUX.1 Dev - 最新架构，文字理解能力最强",
        "best_for": ["通用", "文字渲染", "复杂构图", "创意"],
        "resolution": "1024x1024",
        "steps": 20,
        "cfg": 3.5,
        "sampler": "euler",
        "scheduler": "simple",
    },
    "flux-schnell": {
        "repo": "black-forest-labs/FLUX.1-schnell",
        "file": "flux1-schnell.safetensors",
        "family": "flux",
        "desc": "FLUX.1 Schnell - 极速出图，1-4步完成",
        "best_for": ["快速生成", "预览", "批量"],
        "resolution": "1024x1024",
        "steps": 4,
        "cfg": 1,
        "sampler": "euler",
        "scheduler": "simple",
    },
}

# ============================================================
# LoRA Model Presets
# ============================================================

LORA_PRESETS = {
    # --- Detail Enhancement ---
    "detail-tweaker-xl": {
        "repo": "Owen15911/sdxl-detail-tweaker",
        "file": "sdxl_detail_tweaker.safetensors",
        "family": "sdxl",
        "desc": "细节增强 - 提升画面精细度和纹理",
        "strength": 0.8,
        "best_for": ["细节", "纹理", "质量提升"],
        "tags": ["detail", "quality"],
    },
    "detail-tweaker-15": {
        "repo": "Owen15911/detail-tweaker",
        "file": "detail_tweaker.safetensors",
        "family": "sd15",
        "desc": "SD1.5细节增强",
        "strength": 0.8,
        "best_for": ["细节", "质量提升"],
        "tags": ["detail", "quality"],
    },
    "add-details-xl": {
        "repo": "artificialguybr/sdxl-add-details",
        "file": "add_details.sdxl.safetensors",
        "family": "sdxl",
        "desc": "添加细节 - 让画面更精细",
        "strength": 0.6,
        "best_for": ["细节", "锐度"],
        "tags": ["detail", "sharp"],
    },

    # --- Style Transfer ---
    "oil-painting-xl": {
        "repo": "artificialguybr/oil-painting-sdxl",
        "file": "oil_painting.sdxl.safetensors",
        "family": "sdxl",
        "desc": "油画风格 - 古典油画效果",
        "strength": 0.7,
        "best_for": ["油画", "古典", "艺术"],
        "tags": ["style", "oil", "art"],
    },
    "watercolor-xl": {
        "repo": "artificialguybr/watercolor-sdxl",
        "file": "watercolor.sdxl.safetensors",
        "family": "sdxl",
        "desc": "水彩风格 - 柔和水彩效果",
        "strength": 0.7,
        "best_for": ["水彩", "柔和", "艺术"],
        "tags": ["style", "watercolor", "art"],
    },
    "anime-style-xl": {
        "repo": "artificialguybr/anime-style-sdxl",
        "file": "anime_style.sdxl.safetensors",
        "family": "sdxl",
        "desc": "动漫风格 - 日系动漫效果",
        "strength": 0.7,
        "best_for": ["动漫", "二次元", "插画"],
        "tags": ["style", "anime"],
    },

    # --- E-commerce ---
    "product-photo-xl": {
        "repo": "artificialguybr/product-photography-sdxl",
        "file": "product_photography.sdxl.safetensors",
        "family": "sdxl",
        "desc": "产品摄影增强 - 电商商品图专用",
        "strength": 0.7,
        "best_for": ["电商", "产品摄影", "白底图"],
        "tags": ["ecommerce", "product"],
    },

    # --- FLUX LoRAs ---
    "flux-frosting": {
        "repo": "alvdansen/frosting-lane-flux",
        "file": "frosting_lane_flux.safetensors",
        "family": "flux",
        "desc": "FLUX柔和梦幻风格",
        "strength": 0.8,
        "best_for": ["梦幻", "柔和", "艺术"],
        "tags": ["style", "dreamy", "flux"],
    },
    "flux-realism": {
        "repo": "alvdansen/realism-for-flux",
        "file": "realism_flux.safetensors",
        "family": "flux",
        "desc": "FLUX写实增强",
        "strength": 0.7,
        "best_for": ["写实", "照片"],
        "tags": ["realism", "photo", "flux"],
    },
}

# ============================================================
# VAE Model Presets
# ============================================================

VAE_PRESETS = {
    "sdxl-vae": {
        "repo": "stabilityai/sdxl-vae",
        "file": "sdxl_vae.safetensors",
        "family": "sdxl",
        "desc": "SDXL官方VAE - 改善色彩和细节",
        "best_for": ["SDXL通用", "色彩改善"],
    },
    "sdxl-vae-fp16": {
        "repo": "stabilityai/sdxl-vae-fp16-fix",
        "file": "sdxl_vae_fp16.safetensors",
        "family": "sdxl",
        "desc": "SDXL VAE FP16 - 半精度版本，节省显存",
        "best_for": ["SDXL", "低显存"],
    },
    "sd15-vae": {
        "repo": "stabilityai/sd-vae-ft-mse",
        "file": "vae-ft-mse-840000-ema-pruned.safetensors",
        "family": "sd15",
        "desc": "SD 1.5官方改进VAE - 更好的色彩还原",
        "best_for": ["SD1.5通用", "色彩改善"],
    },
    "sd15-ema-vae": {
        "repo": "stabilityai/sd-vae-ft-ema",
        "file": "vae-ft-ema-560000-ema-pruned.safetensors",
        "family": "sd15",
        "desc": "SD 1.5 EMA VAE - 更平滑的输出",
        "best_for": ["SD1.5", "平滑效果"],
    },
}

# ============================================================
# ControlNet Presets
# ============================================================

CONTROLNET_PRESETS = {
    "sdxl-canny": {
        "repo": "diffusers/controlnet-canny-sdxl-1.0",
        "file": "diffusion_pytorch_model.safetensors",
        "family": "sdxl",
        "desc": "SDXL Canny ControlNet - 适合边缘、线稿、产品轮廓控制",
        "best_for": ["构图控制", "线稿", "产品", "建筑"],
    },
    "sdxl-depth": {
        "repo": "diffusers/controlnet-depth-sdxl-1.0",
        "file": "diffusion_pytorch_model.safetensors",
        "family": "sdxl",
        "desc": "SDXL Depth ControlNet - 适合空间、姿态、景深结构控制",
        "best_for": ["景色", "建筑", "人物姿态", "空间结构"],
    },
    "sd15-canny": {
        "repo": "lllyasviel/control_v11p_sd15_canny",
        "file": "diffusion_pytorch_model.safetensors",
        "family": "sd15",
        "desc": "SD1.5 Canny ControlNet - 经典边缘控制模型",
        "best_for": ["线稿", "边缘", "产品", "二次元"],
    },
    "sd15-depth": {
        "repo": "lllyasviel/control_v11f1p_sd15_depth",
        "file": "diffusion_pytorch_model.safetensors",
        "family": "sd15",
        "desc": "SD1.5 Depth ControlNet - 经典深度结构控制模型",
        "best_for": ["景色", "建筑", "人物姿态", "空间结构"],
    },
}

# ============================================================
# Upscale Model Presets
# ============================================================

UPSCALE_PRESETS = {
    "4x-ultrasharp": {
        "repo": "Kim2091/UltraSharp",
        "file": "4x-UltraSharp.pth",
        "desc": "4x UltraSharp - 最佳通用放大模型",
        "scale": 4,
        "best_for": ["通用放大", "细节保留"],
    },
    "4x-nmkd-superscale": {
        "repo": "gemasai/4x_NMKD-Superscale-SP_178000_G",
        "file": "4x_NMKD-Superscale-SP_178000_G.pth",
        "desc": "4x NMKD Superscale - 高质量放大",
        "scale": 4,
        "best_for": ["通用放大", "照片"],
    },
    "4x-realesrgan": {
        "repo": "xinntao/Real-ESRGAN",
        "file": "RealESRGAN_x4plus.pth",
        "desc": "Real-ESRGAN 4x - 腾讯出品，真实世界放大",
        "scale": 4,
        "best_for": ["真实照片", "老照片修复"],
    },
    "2x-digital-art": {
        "repo": "Kim2091/2x-Digital-Art",
        "file": "2x-Digital-Art.pth",
        "desc": "2x Digital Art - 数字艺术放大",
        "scale": 2,
        "best_for": ["插画", "动漫", "数字艺术"],
    },
}


MODEL_EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin")

MODEL_DIRS = {
    "checkpoint": "checkpoints",
    "lora": "loras",
    "vae": "vae",
    "controlnet": "controlnet",
    "upscale": "upscale_models",
}

MIRRORS = {
    "hf-mirror": "https://hf-mirror.com",
    "huggingface": "https://huggingface.co",
}

IMAGE_TYPE_ALIASES = {
    "anime": ["anime", "二次元", "动漫", "插画", "manga", "cartoon", "toon"],
    "realistic": ["realistic", "写实", "真人", "照片", "photorealistic", "photo", "realism"],
    "animal": ["animal", "动物", "宠物", "pet", "wildlife"],
    "landscape": ["landscape", "景色", "风景", "自然", "scenery", "nature"],
    "ecommerce_product": ["ecommerce", "电商", "商品", "产品", "product"],
    "portrait": ["portrait", "人像", "人物", "model", "fashion", "模特"],
    "architecture": ["architecture", "建筑", "室内", "interior", "building"],
    "fast": ["fast", "快速", "preview", "预览"],
}

IMAGE_TYPE_LABELS = {
    "anime": "二次元/动漫",
    "realistic": "写实/照片",
    "animal": "动物/宠物",
    "landscape": "景色/自然",
    "ecommerce_product": "电商商品",
    "portrait": "人像/模特",
    "architecture": "建筑/室内",
    "fast": "快速预览",
}

TYPE_SEARCH_QUERIES = {
    "anime": "anime",
    "realistic": "realistic",
    "animal": "animal wildlife",
    "landscape": "landscape",
    "ecommerce_product": "product photography",
    "portrait": "portrait realistic",
    "architecture": "architecture interior",
    "fast": "turbo lightning",
}

TYPE_INCLUDE_KEYWORDS = {
    "anime": [
        "anime", "anything", "anylora", "counterfeit", "meina", "revanimated",
        "rev_animated", "revanimated", "toon", "cartoon", "manga", "pastel",
        "mistoon", "animagine", "waifu", "pony", "illustration", "illustrious",
    ],
    "realistic": [
        "realvis", "juggernaut", "realistic", "realism",
        "photoreal", "epicrealism", "absolute", "cyberrealistic", "majicmix",
        "realisticvision", "realistic_vision", "epicrealism", "deliberate",
    ],
    "animal": [
        "animal", "wildlife", "pet", "dog", "cat", "creature", "zoo",
        "fur", "furry", "horse", "bird", "lion", "tiger", "wolf",
    ],
    "landscape": [
        "landscape", "scenery", "nature", "outdoor", "environment", "forest",
        "mountain", "river", "sky", "cloud", "vista", "terrain",
    ],
    "ecommerce_product": [
        "product", "ecommerce", "commercial", "catalog", "packshot",
        "whitebg", "white_bg", "stilllife", "still_life",
    ],
    "portrait": [
        "portrait", "face", "headshot", "fashion", "model", "person",
        "people", "skin", "beauty",
    ],
    "architecture": [
        "architecture", "building", "interior", "exterior", "room", "house",
        "home", "architectural", "realestate", "real_estate",
    ],
    "fast": ["turbo", "lightning", "schnell", "fast", "lcm"],
}

TYPE_EXCLUDE_KEYWORDS = {
    "anime": ["realvis", "juggernaut", "epicrealism", "photoreal"],
    "realistic": [
        "anything", "anime", "counterfeit", "meina", "toon", "cartoon",
        "manga", "revanimated", "rev_animated", "animagine", "waifu",
    ],
    "ecommerce_product": ["anything", "anime", "toon", "cartoon", "manga"],
    "portrait": ["anything", "toon", "cartoon", "manga"],
    "architecture": ["anything", "anime", "toon", "cartoon", "manga"],
}

CATEGORY_PRESET_MAPS = {
    "checkpoint": CHECKPOINT_PRESETS,
    "lora": LORA_PRESETS,
    "vae": VAE_PRESETS,
    "controlnet": CONTROLNET_PRESETS,
    "upscale": UPSCALE_PRESETS,
}

COMBO_CATEGORY_KEYS = {
    "checkpoint": "checkpoint",
    "lora": "lora",
    "vae": "vae",
    "controlnet": "controlnet",
}

IMAGE_TYPE_COMBOS = {
    "ecommerce_product": {
        "sdxl": {
            "checkpoint": "juggernaut-xl",
            "lora": "product-photo-xl",
            "vae": "sdxl-vae",
            "controlnet": "sdxl-canny",
            "desc": "电商商品图: Juggernaut XL + 产品摄影LoRA，可选 Canny 控制产品轮廓",
        },
        "flux": {
            "checkpoint": "flux-dev",
            "lora": None,
            "vae": None,
            "controlnet": None,
            "desc": "电商商品图 FLUX 方案: FLUX Dev，适合复杂文字理解和高质量构图",
        },
    },
    "ecommerce_model": {
        "sdxl": {
            "checkpoint": "realvisxl",
            "lora": "detail-tweaker-xl",
            "vae": "sdxl-vae",
            "controlnet": "sdxl-depth",
            "desc": "电商模特图: RealVis XL + 细节增强LoRA，可选 Depth 控制姿态/空间",
        },
    },
    "anime": {
        "sd15": {
            "checkpoint": "rev-animated",
            "lora": None,
            "vae": "sd15-vae",
            "controlnet": "sd15-canny",
            "desc": "二次元 SD1.5 方案: ReV Animated + SD1.5 VAE，可选 Canny 控线稿",
        },
        "sdxl": {
            "checkpoint": "dreamshaper-xl",
            "lora": "anime-style-xl",
            "vae": "sdxl-vae",
            "controlnet": None,
            "desc": "二次元 SDXL 方案: DreamShaper XL + 动漫风格LoRA",
        },
    },
    "realistic": {
        "sdxl": {
            "checkpoint": "juggernaut-xl",
            "lora": "detail-tweaker-xl",
            "vae": "sdxl-vae",
            "controlnet": None,
            "desc": "写实照片: Juggernaut XL + 细节增强LoRA",
        },
        "flux": {
            "checkpoint": "flux-dev",
            "lora": "flux-realism",
            "vae": None,
            "controlnet": None,
            "desc": "写实 FLUX 方案: FLUX Dev + 写实增强LoRA",
        },
    },
    "animal": {
        "sdxl": {
            "checkpoint": "juggernaut-xl",
            "lora": "detail-tweaker-xl",
            "vae": "sdxl-vae",
            "controlnet": None,
            "desc": "动物/宠物写实: Juggernaut XL + 细节增强LoRA",
        },
        "flux": {
            "checkpoint": "flux-dev",
            "lora": None,
            "vae": None,
            "controlnet": None,
            "desc": "动物/宠物通用: FLUX Dev，适合复杂自然语言描述",
        },
    },
    "landscape": {
        "sdxl": {
            "checkpoint": "sdxl-base",
            "lora": "add-details-xl",
            "vae": "sdxl-vae",
            "controlnet": "sdxl-depth",
            "desc": "景色/自然: SDXL Base + 细节LoRA，可选 Depth 控制空间层次",
        },
        "flux": {
            "checkpoint": "flux-dev",
            "lora": "flux-frosting",
            "vae": None,
            "controlnet": None,
            "desc": "景色/自然 FLUX 方案: FLUX Dev + 柔和风格LoRA",
        },
    },
    "portrait": {
        "sdxl": {
            "checkpoint": "realvisxl",
            "lora": "detail-tweaker-xl",
            "vae": "sdxl-vae",
            "controlnet": "sdxl-depth",
            "desc": "人像/模特: RealVis XL + 细节增强LoRA，可选 Depth 控制姿态",
        },
    },
    "architecture": {
        "sdxl": {
            "checkpoint": "sdxl-base",
            "lora": "add-details-xl",
            "vae": "sdxl-vae",
            "controlnet": "sdxl-canny",
            "desc": "建筑/室内: SDXL Base + 细节LoRA，可选 Canny 保持线条结构",
        },
        "flux": {
            "checkpoint": "flux-dev",
            "lora": None,
            "vae": None,
            "controlnet": None,
            "desc": "建筑/室内 FLUX 方案: FLUX Dev，适合复杂空间描述",
        },
    },
    "fast": {
        "sdxl": {
            "checkpoint": "dreamshaper-xl",
            "lora": None,
            "vae": "sdxl-vae-fp16",
            "controlnet": None,
            "desc": "快速生成: DreamShaper XL Turbo (8步即可)",
        },
        "flux": {
            "checkpoint": "flux-schnell",
            "lora": None,
            "vae": None,
            "controlnet": None,
            "desc": "极速方案: FLUX Schnell (1-4步完成)",
        },
    },
}


def normalize_image_type(image_type):
    """Map Chinese/English user text to a supported image type when possible."""
    if not image_type:
        return None

    value = image_type.strip().lower().replace("-", "_").replace(" ", "_")
    for canonical, aliases in IMAGE_TYPE_ALIASES.items():
        if value == canonical or value in aliases:
            return canonical
        if any(alias in value for alias in aliases):
            return canonical
    return value


def get_image_type_label(image_type):
    return IMAGE_TYPE_LABELS.get(image_type, image_type)


def normalize_model_text(value):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def preset_match_terms(category, preset_id):
    """Terms that can identify an installed file as a curated preset."""
    preset = CATEGORY_PRESET_MAPS.get(category, {}).get(preset_id, {})
    raw_terms = [preset_id]
    for key in ("file", "repo"):
        value = preset.get(key)
        if not value:
            continue
        raw_terms.append(Path(value).stem)
        raw_terms.append(Path(value).name)
    terms = []
    for term in raw_terms:
        normalized = normalize_model_text(term)
        if normalized and normalized not in terms:
            terms.append(normalized)
    return terms


def curated_preset_ids_for_type(image_type, category):
    """Return preset IDs from curated combos that are relevant to an image type."""
    combo_key = COMBO_CATEGORY_KEYS.get(category)
    if not combo_key:
        return []

    preset_ids = []
    for combo in get_type_combos(image_type, family="any").values():
        preset_id = combo.get(combo_key)
        if preset_id and preset_id not in preset_ids:
            preset_ids.append(preset_id)
    return preset_ids


def local_model_match_reason(image_type, model):
    """Return a strict match reason, or None if the local file is not type-relevant."""
    normalized_type = normalize_image_type(image_type)
    name = normalize_model_text(model["name"])
    stem = normalize_model_text(Path(model["name"]).stem)
    category = model.get("category", "checkpoint")

    excludes = TYPE_EXCLUDE_KEYWORDS.get(normalized_type, [])
    if any(normalize_model_text(pattern) in name for pattern in excludes):
        return None

    for preset_id in curated_preset_ids_for_type(normalized_type, category):
        for term in preset_match_terms(category, preset_id):
            if len(term) >= 4 and (term in name or stem in term):
                return f"curated:{preset_id}"

    for keyword in TYPE_INCLUDE_KEYWORDS.get(normalized_type, []):
        term = normalize_model_text(keyword)
        if len(term) >= 3 and term in name:
            return f"filename:{keyword}"

    if normalized_type not in IMAGE_TYPE_COMBOS:
        for term in normalize_model_text(normalized_type).split("_"):
            if len(term) >= 3 and term in name:
                return f"custom:{term}"

    return None


def scan_local_models(comfyui_path, category=None):
    """Return installed model-like files under a ComfyUI models directory."""
    if not comfyui_path:
        return []

    base = Path(comfyui_path) / "models"
    categories = [category] if category else list(MODEL_DIRS.keys())
    results = []
    for cat in categories:
        model_dir = base / MODEL_DIRS[cat]
        if not model_dir.exists():
            continue
        for path in sorted(model_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS:
                results.append({
                    "category": cat,
                    "name": path.name,
                    "path": str(path),
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
                })
    return results


def filter_local_models(comfyui_path, image_type, category=None):
    matches = []
    for model in scan_local_models(comfyui_path, category=category):
        reason = local_model_match_reason(image_type, model)
        if reason:
            model = {**model, "match_reason": reason}
            matches.append(model)
    return matches


def get_type_combos(image_type, family="any"):
    normalized = normalize_image_type(image_type)
    combos = IMAGE_TYPE_COMBOS.get(normalized, {})
    if family and family != "any":
        return {family: combos[family]} if family in combos else {}
    return combos


def get_recommended_combo(task_type, family="sdxl"):
    """Get a recommended model combo (checkpoint + LoRA + VAE) for a task type."""
    task_combos = get_type_combos(task_type, family="any")
    return task_combos.get(family, task_combos.get("sdxl", {}))


def list_presets(category=None, family=None):
    """List all presets with optional filtering."""
    all_categories = {
        "checkpoint": CHECKPOINT_PRESETS,
        "lora": LORA_PRESETS,
        "vae": VAE_PRESETS,
        "controlnet": CONTROLNET_PRESETS,
        "upscale": UPSCALE_PRESETS,
    }

    results = {}
    for cat_name, cat_data in all_categories.items():
        if category and cat_name != category:
            continue
        filtered = {}
        for key, preset in cat_data.items():
            if family and preset.get("family") != family:
                continue
            filtered[key] = preset
        if filtered:
            results[cat_name] = filtered

    return results


def print_presets(presets):
    """Pretty print presets."""
    for category, items in presets.items():
        print(f"\n{'='*60}")
        print(f"  {category.upper()} Models")
        print(f"{'='*60}")
        for key, info in items.items():
            print(f"\n  [{key}]")
            print(f"    {info['desc']}")
            print(f"    Repo: {info['repo']}")
            print(f"    File: {info.get('file', 'N/A')}")
            if info.get("family"):
                print(f"    Family: {info['family']}")
            if info.get("best_for"):
                print(f"    Best for: {', '.join(info['best_for'])}")
            if info.get("resolution"):
                print(f"    Resolution: {info['resolution']}")
            if info.get("steps"):
                print(f"    Steps: {info['steps']} | CFG: {info.get('cfg')} | Sampler: {info.get('sampler')}")
            if info.get("strength"):
                print(f"    Recommended strength: {info['strength']}")
            if info.get("scale"):
                print(f"    Scale: {info['scale']}x")


def print_combos(task_type=None):
    """Print recommended model combinations."""
    tasks = list(IMAGE_TYPE_COMBOS.keys())
    if task_type:
        normalized = normalize_image_type(task_type)
        tasks = [normalized] if normalized in tasks else []

    print(f"\n{'='*60}")
    print(f"  Recommended Model Combinations")
    print(f"{'='*60}")

    for task in tasks:
        print(f"\n  Type: {get_image_type_label(task)} ({task})")
        print(f"  {'-'*40}")
        for family in ["sdxl", "flux", "sd15"]:
            combo = get_recommended_combo(task, family)
            if combo:
                print(f"    [{family}] {combo['desc']}")
                if combo.get("checkpoint"):
                    ckpt = CHECKPOINT_PRESETS.get(combo["checkpoint"], {})
                    print(f"      Checkpoint: {combo['checkpoint']} ({ckpt.get('file', 'N/A')})")
                if combo.get("lora"):
                    lora = LORA_PRESETS.get(combo["lora"], {})
                    print(f"      LoRA: {combo['lora']} (strength: {lora.get('strength', 'N/A')})")
                if combo.get("vae"):
                    vae = VAE_PRESETS.get(combo["vae"], {})
                    print(f"      VAE: {combo['vae']} ({vae.get('file', 'N/A')})")
                if combo.get("controlnet"):
                    controlnet = CONTROLNET_PRESETS.get(combo["controlnet"], {})
                    print(f"      ControlNet: {combo['controlnet']} ({controlnet.get('file', 'N/A')})")


def search_huggingface_models(image_type, mirror="hf-mirror", limit=5):
    """Search HuggingFace for candidate text-to-image models for a type."""
    normalized = normalize_image_type(image_type)
    query = TYPE_SEARCH_QUERIES.get(normalized, f"{image_type} stable diffusion checkpoint safetensors")
    mirror_url = MIRRORS.get(mirror, MIRRORS["hf-mirror"])
    params = urllib.parse.urlencode({
        "search": query,
        "limit": limit,
        "sort": "downloads",
        "pipeline_tag": "text-to-image",
    })
    url = f"{mirror_url}/api/models?{params}"

    req = urllib.request.Request(url, headers={"User-Agent": "CUI-Model-Presets/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"Search failed: {e}", file=sys.stderr)
        return []

    results = []
    for model in data:
        model_id = model.get("modelId", "")
        if not model_id:
            continue
        tags = model.get("tags", [])
        results.append({
            "repo": model_id,
            "downloads": model.get("downloads", 0),
            "likes": model.get("likes", 0),
            "tags": tags[:8],
            "last_modified": model.get("lastModified", ""),
        })
    return results[:limit]


def build_type_recommendation(image_type, family="any", comfyui_path=None,
                              search_if_missing=False, mirror="hf-mirror", limit=5):
    """Build a complete recommendation payload for a selected image type."""
    normalized = normalize_image_type(image_type)
    local_matches = filter_local_models(comfyui_path, normalized, category="checkpoint") if comfyui_path else []
    combos = get_type_combos(normalized, family=family)
    search_results = []

    if search_if_missing and not local_matches and not combos:
        search_results = search_huggingface_models(image_type, mirror=mirror, limit=limit)
    elif search_if_missing and not local_matches and normalized not in IMAGE_TYPE_COMBOS:
        search_results = search_huggingface_models(image_type, mirror=mirror, limit=limit)

    return {
        "input_type": image_type,
        "image_type": normalized,
        "label": get_image_type_label(normalized),
        "local_checkpoints": local_matches,
        "recommended_combos": combos,
        "search_results": search_results,
    }


def print_type_recommendation(payload):
    """Pretty print model recommendations for a selected image type."""
    print(f"\nImage type: {payload['label']} ({payload['image_type']})")
    print("=" * 70)

    local = payload["local_checkpoints"]
    if local:
        print(f"\nLocal matching checkpoints ({len(local)}):")
        for item in local:
            print(f"  - {item['name']} ({item['size_mb']} MB, {item['match_reason']})")
        print("\nLocal match found. Use one of these before suggesting downloads.")
    else:
        print("\nLocal matching checkpoints: none")

    combos = payload["recommended_combos"]
    if combos:
        print("\nRecommended combos:")
        for family, combo in combos.items():
            print(f"  [{family}] {combo['desc']}")
            ckpt = CHECKPOINT_PRESETS.get(combo.get("checkpoint"), {})
            if ckpt:
                print(f"    Checkpoint: {combo['checkpoint']} | {ckpt.get('repo')} | {ckpt.get('file')}")
            lora_id = combo.get("lora")
            if lora_id:
                lora = LORA_PRESETS.get(lora_id, {})
                print(f"    LoRA: {lora_id} | {lora.get('repo')} | {lora.get('file')} | strength {lora.get('strength')}")
            vae_id = combo.get("vae")
            if vae_id:
                vae = VAE_PRESETS.get(vae_id, {})
                print(f"    VAE: {vae_id} | {vae.get('repo')} | {vae.get('file')}")
            controlnet_id = combo.get("controlnet")
            if controlnet_id:
                controlnet = CONTROLNET_PRESETS.get(controlnet_id, {})
                print(f"    ControlNet: {controlnet_id} | {controlnet.get('repo')} | {controlnet.get('file')}")
    else:
        print("\nNo curated combo for this custom type.")

    search_results = payload["search_results"]
    if search_results:
        print("\nOnline search candidates:")
        for result in search_results:
            print(f"  - {result['repo']} | downloads {result['downloads']} | likes {result['likes']}")


def main():
    parser = argparse.ArgumentParser(description="Model Presets for ComfyUI")
    subparsers = parser.add_subparsers(dest="command")

    # types - List supported image types
    subparsers.add_parser("types", help="List supported image types")

    # list - List presets
    list_parser = subparsers.add_parser("list", help="List model presets")
    list_parser.add_argument("--category", choices=["checkpoint", "lora", "vae", "controlnet", "upscale"],
                             help="Filter by category")
    list_parser.add_argument("--family", choices=["sdxl", "sd15", "flux"],
                             help="Filter by model family")

    # combos - Show recommended combos
    combos_parser = subparsers.add_parser("combos", help="Show recommended model combinations")
    combos_parser.add_argument("--task", choices=list(IMAGE_TYPE_COMBOS.keys()),
                               help="Filter by task type")

    # recommend - Get recommendation for a specific use case
    rec_parser = subparsers.add_parser("recommend", help="Get model recommendation")
    rec_parser.add_argument("--task", required=True,
                            choices=list(IMAGE_TYPE_COMBOS.keys()),
                            help="Use case/task type")
    rec_parser.add_argument("--family", default="sdxl", choices=["sdxl", "sd15", "flux"],
                            help="Model family preference")

    # recommend-type - Recommend by selected image type and local availability
    type_parser = subparsers.add_parser("recommend-type", help="Recommend models by image type")
    type_parser.add_argument("--image-type", required=True,
                             help="Image type, e.g. realistic, 二次元, 动物, 景色, or custom text")
    type_parser.add_argument("--family", default="any", choices=["any", "sdxl", "sd15", "flux"],
                             help="Model family preference")
    type_parser.add_argument("--comfyui-path", help="ComfyUI installation path for local model checks")
    type_parser.add_argument("--search-if-missing", action="store_true",
                             help="Search HuggingFace when no local/curated match exists")
    type_parser.add_argument("--mirror", default="hf-mirror", choices=list(MIRRORS.keys()))
    type_parser.add_argument("--limit", type=int, default=5, help="Max online search results")
    type_parser.add_argument("--json", action="store_true", dest="json_output",
                             help="Output machine-readable JSON")

    # local - List local model files matching an image type
    local_parser = subparsers.add_parser("local", help="List local models matching an image type")
    local_parser.add_argument("--comfyui-path", required=True, help="ComfyUI installation path")
    local_parser.add_argument("--image-type", required=True,
                              help="Image type, e.g. realistic, 二次元, 动物, 景色, or custom text")
    local_parser.add_argument("--category", choices=["checkpoint", "lora", "vae", "controlnet", "upscale"],
                              help="Filter local model category")
    local_parser.add_argument("--json", action="store_true", dest="json_output",
                              help="Output machine-readable JSON")

    # search-type - Search HuggingFace for custom/unknown image types
    search_parser = subparsers.add_parser("search-type", help="Search online model candidates by image type")
    search_parser.add_argument("--image-type", required=True,
                               help="Image type or custom text")
    search_parser.add_argument("--mirror", default="hf-mirror", choices=list(MIRRORS.keys()))
    search_parser.add_argument("--limit", type=int, default=5, help="Max results")
    search_parser.add_argument("--json", action="store_true", dest="json_output",
                               help="Output machine-readable JSON")

    # export - Export presets as JSON
    export_parser = subparsers.add_parser("export", help="Export presets to JSON")
    export_parser.add_argument("--output", required=True, help="Output JSON file path")
    export_parser.add_argument("--category", choices=["checkpoint", "lora", "vae", "controlnet", "upscale"],
                               help="Export specific category only")

    args = parser.parse_args()

    if args.command == "types":
        print("Supported image types:")
        print("-" * 50)
        for key, label in IMAGE_TYPE_LABELS.items():
            aliases = ", ".join(IMAGE_TYPE_ALIASES.get(key, []))
            print(f"  {key}: {label} ({aliases})")

    elif args.command == "list":
        presets = list_presets(args.category, args.family)
        if not presets:
            print("No presets match your filter.")
        else:
            print_presets(presets)

    elif args.command == "combos":
        print_combos(args.task)

    elif args.command == "recommend":
        combo = get_recommended_combo(args.task, args.family)
        if combo:
            print(f"\nRecommended for: {args.task} ({args.family})")
            print(f"{'='*50}")
            print(f"  {combo['desc']}")
            if combo.get("checkpoint"):
                ckpt = CHECKPOINT_PRESETS.get(combo["checkpoint"], {})
                print(f"\n  Checkpoint: {combo['checkpoint']}")
                print(f"    File: {ckpt.get('file', 'N/A')}")
                print(f"    Resolution: {ckpt.get('resolution', 'N/A')}")
                print(f"    Steps: {ckpt.get('steps', 'N/A')} | CFG: {ckpt.get('cfg', 'N/A')}")
                print(f"    Sampler: {ckpt.get('sampler', 'N/A')} | Scheduler: {ckpt.get('scheduler', 'N/A')}")
            if combo.get("lora"):
                lora = LORA_PRESETS.get(combo["lora"], {})
                print(f"\n  LoRA: {combo['lora']}")
                print(f"    File: {lora.get('file', 'N/A')}")
                print(f"    Strength: {lora.get('strength', 'N/A')}")
            if combo.get("vae"):
                vae = VAE_PRESETS.get(combo["vae"], {})
                print(f"\n  VAE: {combo['vae']}")
                print(f"    File: {vae.get('file', 'N/A')}")
            if combo.get("controlnet"):
                controlnet = CONTROLNET_PRESETS.get(combo["controlnet"], {})
                print(f"\n  ControlNet: {combo['controlnet']}")
                print(f"    File: {controlnet.get('file', 'N/A')}")
        else:
            print(f"No recommendation found for {args.task}/{args.family}")

    elif args.command == "recommend-type":
        payload = build_type_recommendation(
            image_type=args.image_type,
            family=args.family,
            comfyui_path=args.comfyui_path,
            search_if_missing=args.search_if_missing,
            mirror=args.mirror,
            limit=args.limit,
        )
        if args.json_output:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print_type_recommendation(payload)

    elif args.command == "local":
        matches = filter_local_models(args.comfyui_path, args.image_type, category=args.category)
        if args.json_output:
            print(json.dumps(matches, indent=2, ensure_ascii=False))
        elif matches:
            print(f"Local matches for {args.image_type}:")
            for item in matches:
                print(f"  - [{item['category']}] {item['name']} ({item['size_mb']} MB, {item['match_reason']})")
        else:
            print(f"No local matches for {args.image_type}.")

    elif args.command == "search-type":
        results = search_huggingface_models(args.image_type, mirror=args.mirror, limit=args.limit)
        if args.json_output:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        elif results:
            print(f"Online candidates for {args.image_type}:")
            for result in results:
                print(f"  - {result['repo']} | downloads {result['downloads']} | likes {result['likes']}")
        else:
            print(f"No online candidates found for {args.image_type}.")

    elif args.command == "export":
        all_data = {}
        categories = [args.category] if args.category else ["checkpoint", "lora", "vae", "controlnet", "upscale"]
        for cat in categories:
            all_data[cat] = list_presets(cat).get(cat, {})

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        print(f"Exported presets to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
