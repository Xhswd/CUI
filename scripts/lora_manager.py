#!/usr/bin/env python3
"""
LoRA Manager - Discover, download, list, and recommend LoRA models for ComfyUI
Supports hf-mirror for fast downloads in China
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path

MIRRORS = {
    "hf-mirror": "https://hf-mirror.com",
    "huggingface": "https://huggingface.co",
}

# Curated popular LoRA models organized by use case
LORA_PRESETS = {
    # --- SDXL LoRAs ---
    "sdxl_detail": {
        "repo": "Owen15911/sdxl-detail-tweaker",
        "file": "sdxl_detail_tweaker.safetensors",
        "type": "sdxl",
        "desc": "SDXL细节增强 - 提升画面细节和纹理",
        "strength": 0.8,
        "tags": ["detail", "quality", "sdxl"],
    },
    "sdxl_offset": {
        "repo": "CiroN2022/sdxl-offset",
        "file": "sdxl_offset.safetensors",
        "type": "sdxl",
        "desc": "SDXL偏移LoRA - 改善色彩和对比度",
        "strength": 1.0,
        "tags": ["color", "contrast", "sdxl"],
    },
    "sdxl_add_detail": {
        "repo": "artificialguybr/sdxl-add-details",
        "file": "add_details.sdxl.safetensors",
        "type": "sdxl",
        "desc": "SDXL添加细节 - 增加画面精细度",
        "strength": 0.6,
        "tags": ["detail", "sharp", "sdxl"],
    },
    # --- SD 1.5 LoRAs ---
    "sd15_detail": {
        "repo": "Owen15911/detail-tweaker",
        "file": "detail_tweaker.safetensors",
        "type": "sd15",
        "desc": "SD1.5细节增强",
        "strength": 0.8,
        "tags": ["detail", "quality", "sd15"],
    },
    # --- E-commerce LoRAs ---
    "ecommerce_product_enhance": {
        "repo": "artificialguybr/product-photography-sdxl",
        "file": "product_photography.sdxl.safetensors",
        "type": "sdxl",
        "desc": "电商产品摄影增强 - 提升商品图质量",
        "strength": 0.7,
        "tags": ["ecommerce", "product", "sdxl"],
    },
    # --- Anime LoRAs ---
    "anime_style": {
        "repo": "Owen15911/anime-style-sdxl",
        "file": "anime_style.sdxl.safetensors",
        "type": "sdxl",
        "desc": "动漫风格LoRA",
        "strength": 0.7,
        "tags": ["anime", "style", "sdxl"],
    },
    # --- FLUX LoRAs ---
    "flux_detail": {
        "repo": "alvdansen/frosting-lane-flux",
        "file": "frosting_lane_flux.safetensors",
        "type": "flux",
        "desc": "FLUX风格增强 - 柔和梦幻风格",
        "strength": 0.8,
        "tags": ["style", "dreamy", "flux"],
    },
    "flux_realism": {
        "repo": "alvdansen/realism-for-flux",
        "file": "realism_flux.safetensors",
        "type": "flux",
        "desc": "FLUX写实增强 - 更逼真的照片效果",
        "strength": 0.7,
        "tags": ["realism", "photo", "flux"],
    },
}

# Popular LoRA repos on HuggingFace for search
LORA_SEARCH_KEYWORDS = {
    "detail": ["detail-tweaker", "add-details", "enhance-detail"],
    "anime": ["anime-style", "anime-lora", "ghibli"],
    "realism": ["realism", "photo-realistic", "photorealistic"],
    "ecommerce": ["product-photography", "commercial", "catalog"],
    "style": ["art-style", "oil-painting", "watercolor"],
    "character": ["character-lora", "portrait", "face"],
    "landscape": ["landscape", "scenery", "nature"],
    "architecture": ["architecture", "building", "interior"],
}


def list_local_loras(comfyui_path):
    """List locally installed LoRA models."""
    lora_dir = Path(comfyui_path) / "models" / "loras"
    if not lora_dir.exists():
        return []

    loras = []
    for f in sorted(lora_dir.iterdir()):
        if f.suffix in (".safetensors", ".ckpt", ".pt", ".pth", ".bin"):
            size_mb = f.stat().st_size / (1024 * 1024)
            loras.append({
                "name": f.name,
                "path": str(f),
                "size_mb": round(size_mb, 1),
            })
    return loras


def list_presets(filter_type=None, filter_tag=None):
    """List available LoRA presets with optional filtering."""
    results = []
    for key, preset in LORA_PRESETS.items():
        if filter_type and preset["type"] != filter_type:
            continue
        if filter_tag and filter_tag not in preset.get("tags", []):
            continue
        results.append({"id": key, **preset})
    return results


def search_loras(keyword, mirror="hf-mirror", limit=5):
    """Search HuggingFace for LoRA models matching a keyword."""
    mirror_url = MIRRORS.get(mirror, MIRRORS["hf-mirror"])

    # Try searching via HF API
    search_url = f"{mirror_url}/api/models?search={keyword}+lora&limit={limit}&sort=downloads"
    try:
        req = urllib.request.Request(search_url, headers={"User-Agent": "CUI-LoRA-Manager/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results = []
        for model in data:
            model_id = model.get("modelId", "")
            if not model_id:
                continue
            # Filter for likely LoRA models (small file size, safetensors)
            results.append({
                "repo": model_id,
                "downloads": model.get("downloads", 0),
                "likes": model.get("likes", 0),
                "tags": model.get("tags", [])[:5],
                "last_modified": model.get("lastModified", ""),
            })
        return results
    except Exception as e:
        print(f"Search failed: {e}", file=sys.stderr)
        return []


def download_lora(repo, filename=None, output_dir=None, mirror="hf-mirror"):
    """Download a LoRA model from HuggingFace."""
    mirror_url = MIRRORS.get(mirror, MIRRORS["hf-mirror"])

    if not output_dir:
        print("Error: --output directory is required", file=sys.stderr)
        return False

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to find the safetensors file
    api_url = f"{mirror_url}/api/models/{repo}"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "CUI-LoRA-Manager/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        siblings = data.get("siblings", [])
        all_files = [s["rfilename"] for s in siblings if s.get("rfilename")]

        if not filename:
            # Auto-detect LoRA file (prefer single safetensors, exclude checkpoints)
            safetensors = [f for f in all_files if f.endswith(".safetensors")]
            # Filter out large checkpoint files (likely > 2GB based on name patterns)
            lora_candidates = [f for f in safetensors if
                               any(kw in f.lower() for kw in ["lora", "adapter", "rank"]) or
                               not any(kw in f.lower() for kw in ["checkpoint", "full", "base", "unet"])]
            if lora_candidates:
                filename = lora_candidates[0]
            elif safetensors:
                filename = safetensors[0]
            else:
                print(f"No suitable LoRA file found in {repo}", file=sys.stderr)
                return False

        print(f"Downloading LoRA: {repo}/{filename}")

        # Try huggingface-cli first
        env = os.environ.copy()
        if "hf-mirror" in mirror_url:
            env["HF_ENDPOINT"] = "https://hf-mirror.com"

        cmd = ["huggingface-cli", "download", repo, filename, "--local-dir", str(output_dir)]
        try:
            result = subprocess.run(cmd, env=env, check=False)
            if result.returncode == 0:
                print(f"Download complete: {output_dir / filename}")
                return True
        except FileNotFoundError:
            pass

        # Fallback to wget/curl
        download_url = f"{mirror_url}/{repo}/resolve/main/{filename}"
        out_path = output_dir / filename

        for tool, cmd_base in [
            ("wget", ["wget", "-c", "-q", "--show-progress", download_url, "-O", str(out_path)]),
            ("curl", ["curl", "-L", "-C", "-", "-o", str(out_path), download_url]),
        ]:
            try:
                result = subprocess.run(cmd_base, check=False)
                if result.returncode == 0:
                    print(f"Download complete: {out_path}")
                    return True
            except FileNotFoundError:
                continue

        print("Download failed: no download tool available", file=sys.stderr)
        return False

    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return False


def get_lora_info(comfyui_path, lora_name):
    """Get info about an installed LoRA."""
    lora_path = Path(comfyui_path) / "models" / "loras" / lora_name
    if not lora_path.exists():
        return None

    size_mb = lora_path.stat().st_size / (1024 * 1024)
    return {
        "name": lora_name,
        "path": str(lora_path),
        "size_mb": round(size_mb, 1),
        "suffix": lora_path.suffix,
    }


def main():
    parser = argparse.ArgumentParser(description="LoRA Model Manager for ComfyUI")
    parser.add_argument("--api-url", default="http://127.0.0.1:8188", help="ComfyUI API URL")

    subparsers = parser.add_subparsers(dest="command")

    # list - List local LoRA models
    list_parser = subparsers.add_parser("list", help="List installed LoRA models")
    list_parser.add_argument("--comfyui-path", required=True, help="ComfyUI installation path")

    # presets - Show recommended LoRA presets
    presets_parser = subparsers.add_parser("presets", help="Show recommended LoRA presets")
    presets_parser.add_argument("--type", choices=["sdxl", "sd15", "flux"], help="Filter by model type")
    presets_parser.add_argument("--tag", help="Filter by tag")

    # search - Search HuggingFace for LoRAs
    search_parser = subparsers.add_parser("search", help="Search HuggingFace for LoRA models")
    search_parser.add_argument("keyword", help="Search keyword")
    search_parser.add_argument("--mirror", default="hf-mirror", choices=list(MIRRORS.keys()))
    search_parser.add_argument("--limit", type=int, default=5, help="Max results")

    # download - Download a LoRA model
    dl_parser = subparsers.add_parser("download", help="Download a LoRA model")
    dl_parser.add_argument("--repo", required=True, help="HuggingFace repo ID (e.g., user/lora-name)")
    dl_parser.add_argument("--filename", help="Specific filename in repo")
    dl_parser.add_argument("--output", required=True, help="Output directory (ComfyUI/models/loras/)")
    dl_parser.add_argument("--mirror", default="hf-mirror", choices=list(MIRRORS.keys()))

    # info - Get info about a specific LoRA
    info_parser = subparsers.add_parser("info", help="Get info about an installed LoRA")
    info_parser.add_argument("--comfyui-path", required=True, help="ComfyUI installation path")
    info_parser.add_argument("--name", required=True, help="LoRA filename")

    # install-preset - Install a preset LoRA
    install_parser = subparsers.add_parser("install-preset", help="Install a preset LoRA")
    install_parser.add_argument("--preset", required=True, help="Preset ID (from presets command)")
    install_parser.add_argument("--output", required=True, help="Output directory")
    install_parser.add_argument("--mirror", default="hf-mirror", choices=list(MIRRORS.keys()))

    args = parser.parse_args()

    if args.command == "list":
        loras = list_local_loras(args.comfyui_path)
        if not loras:
            print("No LoRA models installed.")
            print(f"Put .safetensors files in: {args.comfyui_path}/models/loras/")
        else:
            print(f"Installed LoRA models ({len(loras)}):")
            print("-" * 60)
            for lora in loras:
                print(f"  {lora['name']}  ({lora['size_mb']} MB)")

    elif args.command == "presets":
        presets = list_presets(filter_type=args.type, filter_tag=args.tag)
        if not presets:
            print("No presets match your filter.")
        else:
            print(f"Recommended LoRA Presets ({len(presets)}):")
            print("-" * 70)
            for p in presets:
                print(f"  [{p['id']}]")
                print(f"    {p['desc']}")
                print(f"    Repo: {p['repo']}")
                print(f"    Type: {p['type']} | Strength: {p['strength']} | Tags: {', '.join(p['tags'])}")
                print()

    elif args.command == "search":
        print(f"Searching HuggingFace for '{args.keyword}' LoRA models...")
        results = search_loras(args.keyword, mirror=args.mirror, limit=args.limit)
        if not results:
            print("No results found.")
        else:
            print(f"Found {len(results)} models:")
            print("-" * 70)
            for r in results:
                print(f"  {r['repo']}")
                print(f"    Downloads: {r['downloads']} | Likes: {r['likes']}")
                print(f"    Tags: {', '.join(r['tags'])}")
                print()

    elif args.command == "download":
        success = download_lora(
            repo=args.repo,
            filename=args.filename,
            output_dir=args.output,
            mirror=args.mirror,
        )
        if not success:
            sys.exit(1)

    elif args.command == "info":
        info = get_lora_info(args.comfyui_path, args.name)
        if info:
            print(f"LoRA: {info['name']}")
            print(f"  Path: {info['path']}")
            print(f"  Size: {info['size_mb']} MB")
            print(f"  Format: {info['suffix']}")
        else:
            print(f"LoRA not found: {args.name}")
            sys.exit(1)

    elif args.command == "install-preset":
        if args.preset not in LORA_PRESETS:
            print(f"Unknown preset: {args.preset}", file=sys.stderr)
            print(f"Available presets: {', '.join(LORA_PRESETS.keys())}")
            sys.exit(1)

        preset = LORA_PRESETS[args.preset]
        print(f"Installing preset: {args.preset}")
        print(f"  {preset['desc']}")
        success = download_lora(
            repo=preset["repo"],
            filename=preset.get("file"),
            output_dir=args.output,
            mirror=args.mirror,
        )
        if success:
            print(f"\nRecommended strength: {preset['strength']}")
        else:
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
