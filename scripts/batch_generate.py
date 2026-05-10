#!/usr/bin/env python3
"""
Batch Generation Script - Submit multiple image generation jobs to ComfyUI
Supports multi-angle, multi-scene, multi-color/style variants
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add parent scripts dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from comfyui_client import ComfyUIClient, fill_workflow
from generate_prompts import generate_prompts, ECOMMERCE_SCENE_TEMPLATES

# --- Batch variant definitions ---

BATCH_ANGLES = {
    "front": "front view, straight-on, centered, facing camera",
    "side": "side view, profile view, lateral view, 90 degree angle",
    "quarter": "three-quarter view, 45 degree angle, angled perspective",
    "top": "top-down view, overhead shot, bird's eye view, flat lay",
    "back": "back view, rear view, reverse side",
    "detail": "close-up, macro shot, detail view, texture detail, material close-up",
}

BATCH_SCENES = {
    "white_bg": "product_white_bg",
    "lifestyle": "product_lifestyle",
    "luxury": "product_luxury",
    "flatlay": "product_flatlay",
    "studio": "model_studio",
    "outdoor": "model_outdoor",
    "street": "model_street",
}

BATCH_COLORS = {
    "red": "red color, crimson, scarlet",
    "blue": "blue color, navy, cobalt",
    "black": "black color, dark, ebony",
    "white": "white color, ivory, cream",
    "green": "green color, emerald, forest green",
    "pink": "pink color, rose, blush",
    "gold": "gold color, golden, metallic gold",
    "silver": "silver color, metallic silver, chrome",
}


def generate_batch_variants(base_subject, batch_type, variants=None):
    """Generate a list of (subject, extra_keywords) tuples for batch generation."""
    results = []

    if batch_type == "angles":
        angle_keys = variants or ["front", "side", "quarter", "top"]
        for key in angle_keys:
            if key in BATCH_ANGLES:
                results.append({
                    "variant_name": f"angle_{key}",
                    "subject": f"{base_subject}, {BATCH_ANGLES[key]}",
                    "extra_positive": BATCH_ANGLES[key],
                })

    elif batch_type == "scenes":
        scene_keys = variants or ["white_bg", "lifestyle", "luxury"]
        for key in scene_keys:
            if key in BATCH_SCENES:
                scene = BATCH_SCENES[key]
                results.append({
                    "variant_name": f"scene_{key}",
                    "subject": base_subject,
                    "ecommerce_scene": scene,
                    "extra_positive": "",
                })

    elif batch_type == "colors":
        color_keys = variants or ["red", "blue", "black", "white"]
        for key in color_keys:
            if key in BATCH_COLORS:
                results.append({
                    "variant_name": f"color_{key}",
                    "subject": f"{base_subject}, {BATCH_COLORS[key]}",
                    "extra_positive": BATCH_COLORS[key],
                })

    elif batch_type == "custom":
        # User provides a list of custom variant subjects
        for i, variant in enumerate(variants or []):
            results.append({
                "variant_name": f"custom_{i+1}",
                "subject": variant,
                "extra_positive": "",
            })

    return results


def run_batch(client, workflow, variants, model_name, output_dir, timeout=600,
              steps=None, cfg=None, width=None, height=None, sampler=None, scheduler=None):
    """Submit batch jobs and collect results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(variants)

    for i, variant in enumerate(variants):
        variant_name = variant.get("variant_name", f"variant_{i+1}")
        subject = variant["subject"]
        ecommerce_scene = variant.get("ecommerce_scene")
        extra_positive = variant.get("extra_positive", "")

        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] Generating: {variant_name}")
        print(f"  Subject: {subject}")

        # Generate prompts
        positive, negative = generate_prompts(
            subject=subject,
            ecommerce_scene=ecommerce_scene,
            extra_positive=extra_positive,
        )

        print(f"  Positive: {positive[:100]}...")

        # Fill workflow
        filled = fill_workflow(
            workflow,
            model_name=model_name,
            positive_prompt=positive,
            negative_prompt=negative,
            steps=steps,
            cfg=cfg,
            width=width,
            height=height,
            sampler=sampler,
            scheduler=scheduler,
        )

        # Set unique filename prefix
        for node_id, node in filled.items():
            if node.get("class_type") == "SaveImage":
                node["inputs"]["filename_prefix"] = f"batch/{variant_name}"

        # Submit
        prompt_id = client.submit_workflow(filled)
        if not prompt_id:
            print(f"  FAILED: Could not submit workflow for {variant_name}")
            results.append({"variant": variant_name, "status": "failed", "error": "submit_failed"})
            continue

        print(f"  Submitted: {prompt_id}")

        # Wait for completion
        result = client.monitor_progress(prompt_id, timeout=timeout)

        if result.get("status") == "success":
            images = result.get("images", [])
            print(f"  Completed! {len(images)} image(s) generated")
            for img in images:
                print(f"    - {img.get('filename', 'unknown')}")
            results.append({
                "variant": variant_name,
                "status": "success",
                "prompt_id": prompt_id,
                "images": images,
                "positive": positive,
                "negative": negative,
            })
        else:
            print(f"  FAILED: {result.get('status')} - {result.get('message', '')}")
            results.append({
                "variant": variant_name,
                "status": result.get("status", "unknown"),
                "prompt_id": prompt_id,
                "error": result.get("message", ""),
            })

        # Small delay between submissions to avoid overwhelming the queue
        if i < total - 1:
            time.sleep(1)

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch generation for ComfyUI")
    parser.add_argument("--api-url", default="http://127.0.0.1:8188", help="ComfyUI API URL")
    parser.add_argument("--workflow", required=True, help="Path to workflow JSON template")
    parser.add_argument("--model", required=True, help="Checkpoint model name")
    parser.add_argument("--subject", required=True, help="Base subject description")
    parser.add_argument("--batch-type", required=True,
                        choices=["angles", "scenes", "colors", "custom"],
                        help="Type of batch variants")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Specific variant keys (e.g., front side quarter) or custom subjects")
    parser.add_argument("--steps", type=int, help="Sampling steps")
    parser.add_argument("--cfg", type=float, help="CFG scale")
    parser.add_argument("--width", type=int, help="Image width")
    parser.add_argument("--height", type=int, help="Image height")
    parser.add_argument("--sampler", help="Sampler name")
    parser.add_argument("--scheduler", help="Scheduler name")
    parser.add_argument("--timeout", type=int, default=600, help="Per-image timeout in seconds")
    parser.add_argument("--output", default=None, help="Output directory (default: ComfyUI output)")
    parser.add_argument("--dry-run", action="store_true", help="Show variants without submitting")

    args = parser.parse_args()

    # Load workflow template
    with open(args.workflow, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    # Generate variants
    variants = generate_batch_variants(args.subject, args.batch_type, args.variants)

    if not variants:
        print("No variants generated. Check your --variants arguments.", file=sys.stderr)
        sys.exit(1)

    print(f"Batch Generation Plan:")
    print(f"  Subject: {args.subject}")
    print(f"  Batch type: {args.batch_type}")
    print(f"  Variants: {len(variants)}")
    print(f"  Model: {args.model}")
    for v in variants:
        print(f"    - {v['variant_name']}: {v['subject'][:60]}...")

    if args.dry_run:
        print("\n[DRY RUN] No jobs submitted.")
        print(json.dumps(variants, indent=2, ensure_ascii=False))
        return

    # Initialize client
    client = ComfyUIClient(args.api_url)

    # Check connection
    stats = client.get_system_stats()
    if not stats:
        print(f"Cannot connect to ComfyUI at {args.api_url}", file=sys.stderr)
        sys.exit(1)

    print(f"\nConnected to ComfyUI")
    output_dir = args.output or "."

    # Run batch
    results = run_batch(
        client=client,
        workflow=workflow,
        variants=variants,
        model_name=args.model,
        output_dir=output_dir,
        timeout=args.timeout,
        steps=args.steps,
        cfg=args.cfg,
        width=args.width,
        height=args.height,
        sampler=args.sampler,
        scheduler=args.scheduler,
    )

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] != "success")

    print(f"\n{'='*60}")
    print(f"Batch Complete: {success} succeeded, {failed} failed out of {len(results)}")

    # Save batch results
    results_file = Path(output_dir) / "batch_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "subject": args.subject,
            "batch_type": args.batch_type,
            "model": args.model,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
