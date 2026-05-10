#!/usr/bin/env python3
"""
Image Post-Processing Script for ComfyUI outputs
Features: background removal, resize, format conversion, image enhancement
Uses PIL/Pillow for image operations, with optional rembg for background removal
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from PIL import Image, ImageEnhance, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from rembg import remove as rembg_remove
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False


def check_dependencies():
    """Check available dependencies."""
    deps = {
        "Pillow": HAS_PIL,
        "rembg": HAS_REMBG,
    }
    missing = [name for name, available in deps.items() if not available]
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
    return deps


def remove_background(input_path, output_path, alpha_matting=False):
    """Remove background from image using rembg."""
    if not HAS_REMBG:
        print("Error: rembg is required for background removal", file=sys.stderr)
        print("Install with: pip install rembg", file=sys.stderr)
        return False

    if not HAS_PIL:
        print("Error: Pillow is required", file=sys.stderr)
        return False

    print(f"Removing background: {input_path}")
    img = Image.open(input_path)

    # Process in RGB mode for rembg, preserve alpha if exists
    if img.mode == "RGBA":
        # Already has alpha, use it
        result = rembg_remove(img, alpha_matting=alpha_matting)
    else:
        result = rembg_remove(img.convert("RGB"), alpha_matting=alpha_matting)

    # Save as PNG to preserve transparency
    out = Path(output_path)
    if out.suffix.lower() != ".png":
        out = out.with_suffix(".png")

    out.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(out), "PNG")
    print(f"Saved: {out} (transparent PNG)")
    return True


def resize_image(input_path, output_path, width=None, height=None, scale=None,
                 maintain_aspect=True, resample="lanczos"):
    """Resize image with options for exact size or scale factor."""
    if not HAS_PIL:
        print("Error: Pillow is required", file=sys.stderr)
        return False

    img = Image.open(input_path)
    orig_w, orig_h = img.size

    if scale:
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
    elif width and height:
        new_w, new_h = width, height
    elif width:
        ratio = width / orig_w
        new_w = width
        new_h = int(orig_h * ratio) if maintain_aspect else orig_h
    elif height:
        ratio = height / orig_h
        new_h = height
        new_w = int(orig_w * ratio) if maintain_aspect else orig_w
    else:
        print("Error: specify --width, --height, or --scale", file=sys.stderr)
        return False

    resample_methods = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS,
    }
    method = resample_methods.get(resample, Image.LANCZOS)

    print(f"Resizing: {orig_w}x{orig_h} -> {new_w}x{new_h}")
    resized = img.resize((new_w, new_h), method)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    resized.save(str(out))
    print(f"Saved: {out}")
    return True


def convert_format(input_path, output_path, quality=95):
    """Convert image format (e.g., PNG to JPG, WEBP, etc.)."""
    if not HAS_PIL:
        print("Error: Pillow is required", file=sys.stderr)
        return False

    img = Image.open(input_path)
    out = Path(output_path)

    # Handle transparency for formats that don't support it
    if out.suffix.lower() in (".jpg", ".jpeg") and img.mode == "RGBA":
        # Create white background
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
        print("Converted RGBA to RGB (white background for JPEG)")

    out.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = {}
    if out.suffix.lower() in (".jpg", ".jpeg", ".webp"):
        save_kwargs["quality"] = quality
    if out.suffix.lower() == ".png":
        save_kwargs["optimize"] = True

    img.save(str(out), **save_kwargs)
    print(f"Converted: {input_path} -> {out}")
    return True


def enhance_image(input_path, output_path, brightness=1.0, contrast=1.0,
                  saturation=1.0, sharpness=1.0, denoise=False):
    """Enhance image quality with adjustable parameters."""
    if not HAS_PIL:
        print("Error: Pillow is required", file=sys.stderr)
        return False

    img = Image.open(input_path).convert("RGB")

    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
        print(f"  Brightness: {brightness}")

    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
        print(f"  Contrast: {contrast}")

    if saturation != 1.0:
        img = ImageEnhance.Color(img).enhance(saturation)
        print(f"  Saturation: {saturation}")

    if sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
        print(f"  Sharpness: {sharpness}")

    if denoise:
        img = img.filter(ImageFilter.MedianFilter(size=3))
        print(f"  Denoise: applied")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out))
    print(f"Saved: {out}")
    return True


def crop_to_square(input_path, output_path, position="center"):
    """Crop image to square aspect ratio."""
    if not HAS_PIL:
        print("Error: Pillow is required", file=sys.stderr)
        return False

    img = Image.open(input_path)
    w, h = img.size
    size = min(w, h)

    if position == "center":
        left = (w - size) // 2
        top = (h - size) // 2
    elif position == "top":
        left = (w - size) // 2
        top = 0
    elif position == "bottom":
        left = (w - size) // 2
        top = h - size
    else:
        left = (w - size) // 2
        top = (h - size) // 2

    cropped = img.crop((left, top, left + size, top + size))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(str(out))
    print(f"Cropped: {w}x{h} -> {size}x{size} ({position})")
    print(f"Saved: {out}")
    return True


def add_padding(input_path, output_path, target_ratio="1:1", pad_color="white"):
    """Add padding to match target aspect ratio (useful for e-commerce platforms)."""
    if not HAS_PIL:
        print("Error: Pillow is required", file=sys.stderr)
        return False

    img = Image.open(input_path).convert("RGBA")
    w, h = img.size

    # Parse ratio and expand the canvas in whichever direction is needed.
    try:
        parts = target_ratio.split(":")
        rw, rh = int(parts[0]), int(parts[1])
        if rw <= 0 or rh <= 0:
            raise ValueError
    except (IndexError, ValueError):
        print(f"Error: invalid ratio '{target_ratio}', expected W:H (e.g. 1:1)", file=sys.stderr)
        return False

    target_ratio_value = rw / rh
    current_ratio = w / h
    if current_ratio > target_ratio_value:
        target_w = w
        target_h = int(round(w / target_ratio_value))
    else:
        target_w = int(round(h * target_ratio_value))
        target_h = h

    # Create padded canvas
    color_map = {
        "white": (255, 255, 255, 255),
        "black": (0, 0, 0, 255),
        "transparent": (0, 0, 0, 0),
    }
    color = color_map.get(pad_color, (255, 255, 255, 255))

    canvas = Image.new("RGBA", (target_w, target_h), color)
    offset_x = (target_w - w) // 2
    offset_y = (target_h - h) // 2
    canvas.paste(img, (offset_x, offset_y), img if img.mode == "RGBA" else None)

    out = Path(output_path)
    if pad_color == "transparent" and out.suffix.lower() in (".jpg", ".jpeg"):
        out = out.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix.lower() in (".jpg", ".jpeg"):
        canvas = canvas.convert("RGB")

    canvas.save(str(out))
    print(f"Padded: {w}x{h} -> {target_w}x{target_h} (ratio {target_ratio}, {pad_color})")
    print(f"Saved: {out}")
    return True


def batch_process(input_dir, output_dir, operation, **kwargs):
    """Apply an operation to all images in a directory."""
    operation = operation.replace("-", "_")
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    images = [f for f in input_dir.iterdir()
              if f.is_file() and f.suffix.lower() in image_extensions]

    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"Processing {len(images)} images...")
    success = 0
    for img_path in sorted(images):
        out_path = output_dir / img_path.name
        print(f"\n[{img_path.name}]")
        try:
            if operation == "remove_bg":
                out_path = out_path.with_suffix(".png")
                if remove_background(str(img_path), str(out_path)):
                    success += 1
            elif operation == "resize":
                if resize_image(str(img_path), str(out_path), **kwargs):
                    success += 1
            elif operation == "enhance":
                if enhance_image(str(img_path), str(out_path), **kwargs):
                    success += 1
            elif operation == "square":
                if crop_to_square(str(img_path), str(out_path)):
                    success += 1
            elif operation == "convert":
                if convert_format(str(img_path), str(out_path), kwargs.get("quality", 95)):
                    success += 1
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nBatch complete: {success}/{len(images)} succeeded")


def main():
    parser = argparse.ArgumentParser(description="Image Post-Processing for ComfyUI")
    subparsers = parser.add_subparsers(dest="command")

    # deps - Check dependencies
    subparsers.add_parser("deps", help="Check available dependencies")

    # remove-bg - Remove background
    bg_parser = subparsers.add_parser("remove-bg", help="Remove image background")
    bg_parser.add_argument("--input", required=True, help="Input image path")
    bg_parser.add_argument("--output", required=True, help="Output image path")
    bg_parser.add_argument("--alpha-matting", action="store_true", help="Use alpha matting for better edges")

    # resize - Resize image
    resize_parser = subparsers.add_parser("resize", help="Resize image")
    resize_parser.add_argument("--input", required=True, help="Input image path")
    resize_parser.add_argument("--output", required=True, help="Output image path")
    resize_parser.add_argument("--width", type=int, help="Target width")
    resize_parser.add_argument("--height", type=int, help="Target height")
    resize_parser.add_argument("--scale", type=float, help="Scale factor (e.g., 2.0)")
    resize_parser.add_argument("--resample", default="lanczos",
                               choices=["nearest", "bilinear", "bicubic", "lanczos"])

    # convert - Convert format
    conv_parser = subparsers.add_parser("convert", help="Convert image format")
    conv_parser.add_argument("--input", required=True, help="Input image path")
    conv_parser.add_argument("--output", required=True, help="Output image path")
    conv_parser.add_argument("--quality", type=int, default=95, help="Quality (1-100)")

    # enhance - Enhance image
    enh_parser = subparsers.add_parser("enhance", help="Enhance image quality")
    enh_parser.add_argument("--input", required=True, help="Input image path")
    enh_parser.add_argument("--output", required=True, help="Output image path")
    enh_parser.add_argument("--brightness", type=float, default=1.0, help="Brightness factor")
    enh_parser.add_argument("--contrast", type=float, default=1.0, help="Contrast factor")
    enh_parser.add_argument("--saturation", type=float, default=1.0, help="Saturation factor")
    enh_parser.add_argument("--sharpness", type=float, default=1.0, help="Sharpness factor")
    enh_parser.add_argument("--denoise", action="store_true", help="Apply denoise filter")

    # square - Crop to square
    sq_parser = subparsers.add_parser("square", help="Crop image to square")
    sq_parser.add_argument("--input", required=True, help="Input image path")
    sq_parser.add_argument("--output", required=True, help="Output image path")
    sq_parser.add_argument("--position", default="center",
                           choices=["center", "top", "bottom"])

    # pad - Add padding for aspect ratio
    pad_parser = subparsers.add_parser("pad", help="Add padding for target aspect ratio")
    pad_parser.add_argument("--input", required=True, help="Input image path")
    pad_parser.add_argument("--output", required=True, help="Output image path")
    pad_parser.add_argument("--ratio", default="1:1", help="Target ratio (e.g., 1:1, 4:3, 16:9)")
    pad_parser.add_argument("--color", default="white", choices=["white", "black", "transparent"])

    # batch - Batch process directory
    batch_parser = subparsers.add_parser("batch", help="Batch process all images in directory")
    batch_parser.add_argument("--input-dir", required=True, help="Input directory")
    batch_parser.add_argument("--output-dir", required=True, help="Output directory")
    batch_parser.add_argument("--operation", required=True,
                              choices=["remove-bg", "resize", "enhance", "square", "convert"])
    batch_parser.add_argument("--width", type=int, help="Target width (for resize)")
    batch_parser.add_argument("--height", type=int, help="Target height (for resize)")
    batch_parser.add_argument("--scale", type=float, help="Scale factor (for resize)")
    batch_parser.add_argument("--quality", type=int, default=95, help="Quality (for convert)")
    batch_parser.add_argument("--brightness", type=float, default=1.0)
    batch_parser.add_argument("--contrast", type=float, default=1.0)
    batch_parser.add_argument("--saturation", type=float, default=1.0)
    batch_parser.add_argument("--sharpness", type=float, default=1.0)

    args = parser.parse_args()

    if args.command == "deps":
        deps = check_dependencies()
        print("Dependencies:")
        for name, available in deps.items():
            status = "OK" if available else "MISSING"
            print(f"  {name}: {status}")

    elif args.command == "remove-bg":
        remove_background(args.input, args.output, args.alpha_matting)

    elif args.command == "resize":
        resize_image(args.input, args.output, args.width, args.height, args.scale, resample=args.resample)

    elif args.command == "convert":
        convert_format(args.input, args.output, args.quality)

    elif args.command == "enhance":
        enhance_image(args.input, args.output, args.brightness, args.contrast,
                      args.saturation, args.sharpness, args.denoise)

    elif args.command == "square":
        crop_to_square(args.input, args.output, args.position)

    elif args.command == "pad":
        add_padding(args.input, args.output, args.ratio, args.color)

    elif args.command == "batch":
        batch_process(args.input_dir, args.output_dir, args.operation,
                      width=args.width, height=args.height, scale=args.scale,
                      quality=args.quality, brightness=args.brightness,
                      contrast=args.contrast, saturation=args.saturation,
                      sharpness=args.sharpness)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
