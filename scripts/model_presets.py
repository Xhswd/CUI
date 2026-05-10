#!/usr/bin/env python3
"""
Model Presets - Curated model recommendations for different use cases
Includes checkpoints, LoRAs, VAEs, and other fine-tuned models
Organized by task type and model family (SDXL, SD1.5, FLUX, etc.)
"""

import argparse
import json
import sys

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


def get_recommended_combo(task_type, family="sdxl"):
    """Get a recommended model combo (checkpoint + LoRA + VAE) for a task type."""
    combos = {
        "ecommerce_product": {
            "sdxl": {
                "checkpoint": "juggernaut-xl",
                "lora": "product-photo-xl",
                "vae": "sdxl-vae",
                "desc": "电商商品图最佳组合: Juggernaut XL + 产品摄影LoRA",
            },
            "flux": {
                "checkpoint": "flux-dev",
                "lora": None,
                "vae": None,
                "desc": "FLUX电商方案: FLUX Dev (无需额外LoRA)",
            },
        },
        "ecommerce_model": {
            "sdxl": {
                "checkpoint": "realvisxl",
                "lora": "detail-tweaker-xl",
                "vae": "sdxl-vae",
                "desc": "电商模特图最佳组合: RealVis XL + 细节增强LoRA",
            },
        },
        "anime": {
            "sdxl": {
                "checkpoint": "dreamshaper-xl",
                "lora": "anime-style-xl",
                "vae": "sdxl-vae",
                "desc": "动漫风格最佳组合: DreamShaper XL + 动漫风格LoRA",
            },
            "sd15": {
                "checkpoint": "rev-animated",
                "lora": None,
                "vae": "sd15-vae",
                "desc": "SD1.5动漫方案: ReV Animated + 改进VAE",
            },
        },
        "realistic": {
            "sdxl": {
                "checkpoint": "juggernaut-xl",
                "lora": "detail-tweaker-xl",
                "vae": "sdxl-vae",
                "desc": "写实照片最佳组合: Juggernaut XL + 细节增强LoRA",
            },
        },
        "fast": {
            "sdxl": {
                "checkpoint": "dreamshaper-xl",
                "lora": None,
                "vae": "sdxl-vae-fp16",
                "desc": "快速生成: DreamShaper XL Turbo (8步即可)",
            },
            "flux": {
                "checkpoint": "flux-schnell",
                "lora": None,
                "vae": None,
                "desc": "极速方案: FLUX Schnell (1-4步完成)",
            },
        },
    }

    task_combos = combos.get(task_type, {})
    return task_combos.get(family, task_combos.get("sdxl", {}))


def list_presets(category=None, family=None):
    """List all presets with optional filtering."""
    all_categories = {
        "checkpoint": CHECKPOINT_PRESETS,
        "lora": LORA_PRESETS,
        "vae": VAE_PRESETS,
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
    tasks = ["ecommerce_product", "ecommerce_model", "anime", "realistic", "fast"]
    if task_type:
        tasks = [task_type] if task_type in tasks else []

    print(f"\n{'='*60}")
    print(f"  Recommended Model Combinations")
    print(f"{'='*60}")

    for task in tasks:
        print(f"\n  Task: {task}")
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


def main():
    parser = argparse.ArgumentParser(description="Model Presets for ComfyUI")
    subparsers = parser.add_subparsers(dest="command")

    # list - List presets
    list_parser = subparsers.add_parser("list", help="List model presets")
    list_parser.add_argument("--category", choices=["checkpoint", "lora", "vae", "upscale"],
                             help="Filter by category")
    list_parser.add_argument("--family", choices=["sdxl", "sd15", "flux"],
                             help="Filter by model family")

    # combos - Show recommended combos
    combos_parser = subparsers.add_parser("combos", help="Show recommended model combinations")
    combos_parser.add_argument("--task", choices=["ecommerce_product", "ecommerce_model",
                                                   "anime", "realistic", "fast"],
                               help="Filter by task type")

    # recommend - Get recommendation for a specific use case
    rec_parser = subparsers.add_parser("recommend", help="Get model recommendation")
    rec_parser.add_argument("--task", required=True,
                            choices=["ecommerce_product", "ecommerce_model",
                                     "anime", "realistic", "fast"],
                            help="Use case/task type")
    rec_parser.add_argument("--family", default="sdxl", choices=["sdxl", "sd15", "flux"],
                            help="Model family preference")

    # export - Export presets as JSON
    export_parser = subparsers.add_parser("export", help="Export presets to JSON")
    export_parser.add_argument("--output", required=True, help="Output JSON file path")
    export_parser.add_argument("--category", choices=["checkpoint", "lora", "vae", "upscale"],
                               help="Export specific category only")

    args = parser.parse_args()

    if args.command == "list":
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
        else:
            print(f"No recommendation found for {args.task}/{args.family}")

    elif args.command == "export":
        all_data = {}
        categories = [args.category] if args.category else ["checkpoint", "lora", "vae", "upscale"]
        for cat in categories:
            all_data[cat] = list_presets(cat).get(cat, {})

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        print(f"Exported presets to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
