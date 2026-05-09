#!/usr/bin/env python3
"""
Model download script - supports hf-mirror and direct HuggingFace downloads
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path

MIRRORS = {
    "hf-mirror": "https://hf-mirror.com",
    "huggingface": "https://huggingface.co",
}

MODEL_TYPE_DIRS = {
    "checkpoint": "checkpoints",
    "lora": "loras",
    "controlnet": "controlnet",
    "vae": "vae",
    "upscale": "upscale_models",
    "embedding": "embeddings",
    "clip": "clip",
}

KNOWN_REPOS = {
    "sdxl-base": {
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "file": "sd_xl_base_1.0.safetensors",
        "type": "checkpoint",
    },
    "sdxl-refiner": {
        "repo": "stabilityai/stable-diffusion-xl-refiner-1.0",
        "file": "sd_xl_refiner_1.0.safetensors",
        "type": "checkpoint",
    },
    "sd15": {
        "repo": "runwayml/stable-diffusion-v1-5",
        "file": "v1-5-pruned-emaonly.safetensors",
        "type": "checkpoint",
    },
    "sd21": {
        "repo": "stabilityai/stable-diffusion-2-1",
        "file": "v2-1_768-ema-pruned.safetensors",
        "type": "checkpoint",
    },
    "sd3-medium": {
        "repo": "stabilityai/stable-diffusion-3-medium",
        "file": "sd3_medium.safetensors",
        "type": "checkpoint",
    },
    "flux-schnell": {
        "repo": "black-forest-labs/FLUX.1-schnell",
        "file": "flux1-schnell.safetensors",
        "type": "checkpoint",
    },
    "flux-dev": {
        "repo": "black-forest-labs/FLUX.1-dev",
        "file": "flux1-dev.safetensors",
        "type": "checkpoint",
    },
}


def resolve_model_input(model_input):
    if model_input in KNOWN_REPOS:
        info = KNOWN_REPOS[model_input]
        return info["repo"], info["file"], info["type"]

    if "/" in model_input:
        parts = model_input.split("/")
        if len(parts) >= 2:
            repo = "/".join(parts[:2])
            filename = "/".join(parts[2:]) if len(parts) > 2 else None
            return repo, filename, "checkpoint"

    if model_input.endswith((".safetensors", ".ckpt", ".pt", ".pth", ".bin")):
        return None, model_input, "checkpoint"

    return None, model_input, "checkpoint"


def search_hf_file(repo, filename, mirror_url):
    api_url = f"{mirror_url}/api/models/{repo}"
    try:
        req = urllib.request.Request(api_url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        siblings = data.get("siblings", [])
        all_files = [s["rfilename"] for s in siblings if s.get("rfilename")]

        if filename:
            # Check if exact file exists
            if filename in all_files:
                return filename
            # Check if it's a directory prefix (multi-file model)
            matching = [f for f in all_files if f.startswith(filename + "/") or f.startswith(filename)]
            if matching:
                return matching  # Return list for multi-file

        # Single file preference: non-sharded safetensors first
        safetensors_files = [f for f in all_files if f.endswith(".safetensors")]
        # Filter out sharded files (e.g., diffusion_pytorch_model-00001-of-00003.safetensors)
        single_files = [f for f in safetensors_files if "-00001-of-" not in f and "shard" not in f.lower()]
        if single_files:
            return single_files[0]
        if safetensors_files:
            return safetensors_files[0]

        ckpt_files = [f for f in all_files if f.endswith((".ckpt", ".pt", ".pth", ".bin"))]
        if ckpt_files:
            return ckpt_files[0]

        return None
    except Exception as e:
        print(f"API查询失败: {e}", file=sys.stderr)
        return None


def download_with_hf_cli(repo, filename, output_dir, mirror_url):
    env = os.environ.copy()
    if "hf-mirror" in mirror_url:
        env["HF_ENDPOINT"] = "https://hf-mirror.com"

    cmd = ["huggingface-cli", "download", repo]
    if filename:
        cmd.append(filename)
    cmd.extend(["--local-dir", output_dir])

    print(f"使用 huggingface-cli 下载...")
    print(f"命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, env=env, check=False)
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        print("huggingface-cli 未安装，尝试其他方式...", file=sys.stderr)
    return False


def download_with_wget(url, output_path):
    print(f"使用 wget 下载...")
    print(f"URL: {url}")
    print(f"输出: {output_path}")

    cmd = ["wget", "-c", "-q", "--show-progress", url, "-O", output_path]
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def download_with_curl(url, output_path):
    print(f"使用 curl 下载...")
    print(f"URL: {url}")
    print(f"输出: {output_path}")

    cmd = ["curl", "-L", "-C", "-", "-o", output_path, url]
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def download_with_python(url, output_path):
    print(f"使用 Python 下载...")
    print(f"URL: {url}")
    print(f"输出: {output_path}")

    try:
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"下载失败: {e}", file=sys.stderr)
        return False


def download_file(url, output_path):
    if download_with_wget(url, output_path):
        return True
    if download_with_curl(url, output_path):
        return True
    if download_with_python(url, output_path):
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Download models from HuggingFace mirrors")
    parser.add_argument("--model", required=True, help="Model name, repo ID, or shortcut (e.g., sdxl-base)")
    parser.add_argument("--mirror", default="hf-mirror", choices=list(MIRRORS.keys()), help="Download mirror")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--type", default=None, choices=list(MODEL_TYPE_DIRS.keys()), help="Model type")
    parser.add_argument("--filename", default=None, help="Specific filename in repo")

    args = parser.parse_args()
    mirror_url = MIRRORS[args.mirror]

    repo, filename, model_type = resolve_model_input(args.model)

    if args.type:
        model_type = args.type
    if args.filename:
        filename = args.filename

    output_dir = Path(args.output)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    if repo:
        print(f"仓库: {repo}")
        if not filename:
            print("正在搜索模型文件...")
            filename = search_hf_file(repo, filename, mirror_url)
            if not filename:
                print("未找到模型文件，请指定 --filename", file=sys.stderr)
                sys.exit(1)

        # Handle multi-file models (list of filenames)
        if isinstance(filename, list):
            print(f"多文件模型，共 {len(filename)} 个文件")
            if download_with_hf_cli(repo, None, str(output_dir), mirror_url):
                print(f"\n✓ 下载完成: {output_dir}")
                sys.exit(0)
            # Fallback: download each file
            for fn in filename:
                download_url = f"{mirror_url}/{repo}/resolve/main/{fn}"
                out_path = output_dir / fn
                out_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"下载: {fn}")
                if not download_file(download_url, str(out_path)):
                    print(f"\n✗ 下载失败: {fn}", file=sys.stderr)
                    sys.exit(1)
            print(f"\n✓ 下载完成: {output_dir}")
            sys.exit(0)

        print(f"文件: {filename}")
        print(f"类型: {model_type}")

        if download_with_hf_cli(repo, filename, str(output_dir), mirror_url):
            print(f"\n✓ 下载完成: {output_dir}")
            sys.exit(0)

        download_url = f"{mirror_url}/{repo}/resolve/main/{filename}"
        # Preserve subdirectory structure from repo
        out_path = output_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if download_file(download_url, str(out_path)):
            print(f"\n✓ 下载完成: {out_path}")
        else:
            print(f"\n✗ 下载失败", file=sys.stderr)
            sys.exit(1)
    else:
        if not filename:
            print("请指定模型文件名或仓库ID", file=sys.stderr)
            sys.exit(1)

        output_path = str(output_dir / Path(filename).name)
        print(f"文件: {filename}")
        print(f"输出: {output_path}")
        print("提示: 未指定仓库，无法自动下载。请提供完整的仓库ID (如 stabilityai/stable-diffusion-xl-base-1.0)")
        sys.exit(1)


if __name__ == "__main__":
    main()
