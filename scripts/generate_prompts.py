#!/usr/bin/env python3
"""
Prompt Generator - Generate positive and negative prompts for ComfyUI
Supports general art and e-commerce scenarios
"""

import argparse
import json
import sys

QUALITY_BOOSTERS = {
    "photorealistic": [
        "masterpiece", "best quality", "ultra detailed", "high resolution",
        "8k", "photorealistic", "RAW photo", "detailed skin texture",
        "natural lighting", "depth of field", "film grain",
    ],
    "anime": [
        "masterpiece", "best quality", "highly detailed", "vibrant colors",
        "sharp focus", "illustration", "anime style", "detailed eyes",
        "beautiful shading",
    ],
    "art": [
        "masterpiece", "best quality", "highly detailed", "oil painting",
        "artistic", "expressive brushstrokes", "rich colors",
        "dramatic lighting", "fine art",
    ],
    "concept_art": [
        "masterpiece", "best quality", "concept art", "matte painting",
        "detailed", "epic", "cinematic", "volumetric lighting",
        "trending on artstation",
    ],
    "ecommerce_product": [
        "masterpiece", "best quality", "ultra detailed", "high resolution",
        "8k", "commercial photography", "product photography",
        "studio lighting", "clean background", "sharp focus",
        "professional", "catalog photo", "white background",
    ],
    "ecommerce_model": [
        "masterpiece", "best quality", "ultra detailed", "high resolution",
        "8k", "fashion photography", "editorial", "professional model",
        "studio lighting", "detailed skin texture", "magazine quality",
        "vogue", "high fashion",
    ],
    "ecommerce_lifestyle": [
        "masterpiece", "best quality", "ultra detailed", "high resolution",
        "8k", "lifestyle photography", "natural setting", "aspirational",
        "warm lighting", "cozy atmosphere", "realistic", "inviting",
    ],
}

NEGATIVE_TEMPLATES = {
    "photorealistic": [
        "painting", "drawing", "illustration", "cartoon", "anime",
        "3d render", "cgi", "sketch", "blurry", "low quality", "lowres",
        "bad anatomy", "watermark", "signature", "text", "deformed",
        "distorted", "disfigured", "bad proportions", "duplicate",
        "out of frame",
    ],
    "anime": [
        "photorealistic", "3d", "western cartoon", "realistic",
        "low quality", "lowres", "blurry", "bad anatomy", "bad hands",
        "distorted", "deformed", "mutation", "watermark", "signature",
        "text", "worst quality", "jpeg artifacts",
    ],
    "art": [
        "photograph", "3d render", "cgi", "blurry", "low quality",
        "lowres", "bad anatomy", "watermark", "signature", "text",
        "deformed", "distorted", "worst quality", "jpeg artifacts",
    ],
    "concept_art": [
        "blurry", "low quality", "lowres", "bad anatomy", "watermark",
        "signature", "text", "deformed", "distorted", "worst quality",
        "jpeg artifacts", "photograph",
    ],
    "ecommerce_product": [
        "blurry", "low quality", "lowres", "watermark", "signature",
        "text", "logo", "brand name", "deformed", "distorted",
        "cropped", "out of frame", "background noise", "cluttered",
        "dusty", "scratched", "damaged", "worst quality", "jpeg artifacts",
        "overexposed", "underexposed", "color cast",
    ],
    "ecommerce_model": [
        "blurry", "low quality", "lowres", "bad anatomy", "bad hands",
        "extra fingers", "missing fingers", "deformed", "distorted",
        "disfigured", "bad proportions", "watermark", "signature",
        "text", "logo", "worst quality", "jpeg artifacts",
        "unnatural pose", "stiff", "awkward",
    ],
    "ecommerce_lifestyle": [
        "blurry", "low quality", "lowres", "watermark", "signature",
        "text", "logo", "deformed", "distorted", "worst quality",
        "jpeg artifacts", "artificial", "fake", "staged looking",
        "unrealistic", "overprocessed",
    ],
}

STYLE_KEYWORDS = {
    "cyberpunk": "cyberpunk, neon lights, futuristic, dystopian, rain, holographic, tech noir",
    "fantasy": "fantasy, magical, ethereal, mystical, enchanted, fairy tale, otherworldly",
    "horror": "horror, dark, eerie, creepy, gothic, macabre, unsettling, shadowy",
    "scifi": "sci-fi, space, futuristic, alien, technology, starship, cosmic",
    "steampunk": "steampunk, victorian, brass, gears, clockwork, industrial, steam-powered",
    "watercolor": "watercolor, soft edges, flowing, delicate, pastel, wet on wet technique",
    "oil_painting": "oil painting, rich colors, textured, classical, chiaroscuro, impasto",
    "pixel_art": "pixel art, 16-bit, retro, 8-bit, sprite, nostalgic",
    "minimalist": "minimalist, clean, simple, geometric, modern, sparse, negative space",
    "baroque": "baroque, ornate, dramatic, gilded, elaborate, theatrical, grandiose",
    "impressionist": "impressionist, light and shadow, brushstrokes, en plein air, luminous",
    "art_nouveau": "art nouveau, organic forms, flowing lines, decorative, floral, elegant",
    "vaporwave": "vaporwave, retro aesthetic, pastel colors, glitch, 80s, neon",
    "noir": "film noir, black and white, high contrast, shadows, moody, detective",
    "ukiyo_e": "ukiyo-e, japanese woodblock print, flat colors, bold outlines, edo period",
    "chinese_ink": "chinese ink painting, ink wash, traditional, xuan paper, landscape, calligraphy",
    "ecommerce_clean": "clean product shot, white background, isolated, catalog style, minimal, professional",
    "ecommerce_luxury": "luxury product photography, premium, elegant, dark background, dramatic spotlight, high-end",
    "ecommerce_flatlay": "flat lay photography, top-down view, arranged composition, aesthetic, organized, trendy",
    "ecommerce_outdoor": "outdoor lifestyle photography, natural environment, adventure, active, fresh, vibrant",
    "ecommerce_minimal": "minimalist product photography, simple, clean lines, negative space, modern, scandinavian",
}

LIGHTING_KEYWORDS = {
    "natural": "natural lighting, golden hour, sunlight, soft light, ambient",
    "dramatic": "dramatic lighting, chiaroscuro, high contrast, rim lighting, spotlight",
    "cinematic": "cinematic lighting, volumetric lighting, god rays, lens flare, bokeh",
    "studio": "studio lighting, three-point lighting, softbox, key light, fill light",
    "neon": "neon lighting, colorful lights, glow, luminescent, fluorescent",
    "moody": "moody lighting, low key, shadows, atmospheric, fog, mist",
    "backlight": "backlighting, silhouette, rim light, translucent, glow",
    "ecommerce_studio": "professional studio lighting, soft even lighting, no harsh shadows, color accurate, clean highlights",
    "ecommerce_window": "window light, soft directional light, natural feel, warm tones, gentle shadows",
    "ecommerce_dramatic": "dramatic product lighting, spotlight, dark background, rim light, premium feel",
    "ecommerce_ring": "ring light, beauty lighting, even illumination, catchlight, flattering",
}

COMPOSITION_KEYWORDS = {
    "portrait": "portrait, close-up, face focus, shallow depth of field",
    "landscape": "landscape, wide angle, panoramic, scenic vista",
    "full_body": "full body, standing, head to toe, full figure",
    "close_up": "close-up, detailed, macro, intimate",
    "wide_shot": "wide shot, establishing shot, environmental, context",
    "aerial": "aerial view, bird's eye, top-down, overhead",
    "low_angle": "low angle, looking up, heroic, imposing",
    "symmetrical": "symmetrical composition, balanced, centered, harmonious",
    "ecommerce_center": "centered composition, product centered, straight-on view, symmetrical, balanced",
    "ecommerce_45deg": "45 degree angle view, three-quarter view, dynamic angle, dimensional",
    "ecommerce_detail": "detail shot, macro, texture close-up, material detail, stitching detail",
    "ecommerce_group": "group shot, multiple products, collection display, variety, color options",
    "ecommerce_lifestyle": "lifestyle composition, in-context, natural usage, environment, story telling",
    "ecommerce_flatlay_comp": "flat lay composition, top-down, arranged, organized, aesthetic layout",
}

ECOMMERCE_SCENE_TEMPLATES = {
    "product_white_bg": {
        "positive_suffix": "on pure white background, isolated product shot, clean, professional e-commerce photo, no shadows on background, catalog standard",
        "negative_extra": ["colored background", "pattern background", "textured background", "props", "distracting elements"],
        "quality_style": "ecommerce_product",
        "lighting": "ecommerce_studio",
        "composition": "ecommerce_center",
    },
    "product_lifestyle": {
        "positive_suffix": "in natural lifestyle setting, aspirational scene, warm and inviting, real-world context, premium lifestyle",
        "negative_extra": ["studio", "white background", "artificial", "staged", "fake looking"],
        "quality_style": "ecommerce_lifestyle",
        "lighting": "ecommerce_window",
        "composition": "ecommerce_lifestyle",
    },
    "product_luxury": {
        "positive_suffix": "luxury presentation, dark elegant background, premium feel, high-end product photography, sophisticated, exclusive",
        "negative_extra": ["cheap", "casual", "bright background", "amateur", "low quality"],
        "quality_style": "ecommerce_product",
        "lighting": "ecommerce_dramatic",
        "composition": "ecommerce_center",
    },
    "product_flatlay": {
        "positive_suffix": "flat lay arrangement, top-down view, aesthetically organized, complementary props, trendy composition, instagram style",
        "negative_extra": ["vertical", "standing", "messy", "cluttered", "random"],
        "quality_style": "ecommerce_product",
        "lighting": "ecommerce_studio",
        "composition": "ecommerce_flatlay_comp",
    },
    "model_studio": {
        "positive_suffix": "fashion model, professional studio shoot, clean background, editorial pose, high fashion, magazine cover quality, detailed fabric texture",
        "negative_extra": ["casual snapshot", "selfie", "amateur", "low quality", "unprofessional"],
        "quality_style": "ecommerce_model",
        "lighting": "ecommerce_studio",
        "composition": "full_body",
    },
    "model_outdoor": {
        "positive_suffix": "fashion model in outdoor setting, natural environment, lifestyle fashion, candid pose, natural movement, editorial outdoor",
        "negative_extra": ["studio", "indoor", "stiff pose", "unnatural", "white background"],
        "quality_style": "ecommerce_model",
        "lighting": "natural",
        "composition": "ecommerce_lifestyle",
    },
    "model_street": {
        "positive_suffix": "street fashion photography, urban environment, trendy outfit, street style, candid, dynamic pose, city background",
        "negative_extra": ["studio", "formal", "indoor", "plain background", "stiff"],
        "quality_style": "ecommerce_model",
        "lighting": "natural",
        "composition": "full_body",
    },
    "model_detail": {
        "positive_suffix": "fashion detail shot, fabric close-up, texture detail, stitching detail, material quality, craftsmanship",
        "negative_extra": ["full body", "wide shot", "blurry detail", "low resolution"],
        "quality_style": "ecommerce_product",
        "lighting": "ecommerce_studio",
        "composition": "ecommerce_detail",
    },
    "tryon_virtual": {
        "positive_suffix": "virtual try-on, model wearing the outfit, natural fit, realistic draping, accurate fabric representation, proper proportions",
        "negative_extra": ["ill-fitting", "floating clothes", "unnatural fit", "distorted proportions", "bad body proportions"],
        "quality_style": "ecommerce_model",
        "lighting": "ecommerce_studio",
        "composition": "full_body",
    },
    "bg_replace_clean": {
        "positive_suffix": "clean background replacement, seamless integration, natural edge blending, consistent lighting, professional compositing",
        "negative_extra": ["visible cutout", "halo", "edge artifacts", "inconsistent lighting", "obvious editing"],
        "quality_style": "ecommerce_product",
        "lighting": "ecommerce_studio",
        "composition": "ecommerce_center",
    },
    "bg_replace_scene": {
        "positive_suffix": "in scenic background, naturally integrated, realistic environment, matching lighting, cohesive composition",
        "negative_extra": ["floating", "obvious composite", "mismatched lighting", "fake background", "cutout look"],
        "quality_style": "ecommerce_lifestyle",
        "lighting": "natural",
        "composition": "ecommerce_lifestyle",
    },
}


def generate_prompts(subject, style=None, lighting=None, composition=None,
                     quality_style="photorealistic", extra_positive=None,
                     extra_negative=None, ecommerce_scene=None):
    parts = [subject]

    if ecommerce_scene and ecommerce_scene in ECOMMERCE_SCENE_TEMPLATES:
        template = ECOMMERCE_SCENE_TEMPLATES[ecommerce_scene]
        parts.append(template["positive_suffix"])

        effective_quality = template.get("quality_style", quality_style)
        effective_lighting = template.get("lighting", lighting)
        effective_composition = template.get("composition", composition)
    else:
        effective_quality = quality_style
        effective_lighting = lighting
        effective_composition = composition

    if style and style in STYLE_KEYWORDS:
        parts.append(STYLE_KEYWORDS[style])
    elif style:
        parts.append(style)

    if effective_quality in QUALITY_BOOSTERS:
        parts.append(", ".join(QUALITY_BOOSTERS[effective_quality]))

    if effective_lighting and effective_lighting in LIGHTING_KEYWORDS:
        parts.append(LIGHTING_KEYWORDS[effective_lighting])
    elif effective_lighting:
        parts.append(effective_lighting)

    if effective_composition and effective_composition in COMPOSITION_KEYWORDS:
        parts.append(COMPOSITION_KEYWORDS[effective_composition])
    elif effective_composition:
        parts.append(effective_composition)

    if extra_positive:
        parts.append(extra_positive)

    positive = ", ".join(parts)

    neg_parts = []
    if ecommerce_scene and ecommerce_scene in ECOMMERCE_SCENE_TEMPLATES:
        template = ECOMMERCE_SCENE_TEMPLATES[ecommerce_scene]
        effective_neg_quality = template.get("quality_style", quality_style)
        if effective_neg_quality in NEGATIVE_TEMPLATES:
            neg_parts.extend(NEGATIVE_TEMPLATES[effective_neg_quality])
        neg_parts.extend(template.get("negative_extra", []))
    else:
        if quality_style in NEGATIVE_TEMPLATES:
            neg_parts.extend(NEGATIVE_TEMPLATES[quality_style])

    if extra_negative:
        neg_parts.extend([x.strip() for x in extra_negative.split(",") if x.strip()])

    negative = ", ".join(neg_parts)

    return positive, negative


def list_ecommerce_scenes():
    return list(ECOMMERCE_SCENE_TEMPLATES.keys())


def main():
    parser = argparse.ArgumentParser(description="Generate ComfyUI prompts (supports e-commerce scenes)")
    parser.add_argument("--subject", required=False, help="Subject description (use English for best results)")
    parser.add_argument("--style", default=None,
                       help="Art style (general or ecommerce_*)")
    parser.add_argument("--lighting", default=None,
                       help="Lighting style (general or ecommerce_*)")
    parser.add_argument("--composition", default=None,
                       help="Composition type (general or ecommerce_*)")
    parser.add_argument("--quality-style", default="photorealistic",
                       help="Quality booster style (general or ecommerce_*)")
    parser.add_argument("--ecommerce-scene", default=None,
                       choices=list(ECOMMERCE_SCENE_TEMPLATES.keys()),
                       help="E-commerce scene template (overrides style/lighting/composition)")
    parser.add_argument("--extra-positive", default=None, help="Extra positive keywords")
    parser.add_argument("--extra-negative", default=None, help="Extra negative keywords")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    parser.add_argument("--list-scenes", action="store_true",
                       help="List available e-commerce scene templates")

    args = parser.parse_args()

    if args.list_scenes:
        print("Available e-commerce scene templates:")
        print("-" * 60)
        for name, template in ECOMMERCE_SCENE_TEMPLATES.items():
            print(f"  {name}")
            print(f"    Positive: {template['positive_suffix'][:80]}...")
            print()
        return

    if not args.subject:
        parser.error("--subject is required when not using --list-scenes")

    positive, negative = generate_prompts(
        subject=args.subject,
        style=args.style,
        lighting=args.lighting,
        composition=args.composition,
        quality_style=args.quality_style,
        extra_positive=args.extra_positive,
        extra_negative=args.extra_negative,
        ecommerce_scene=args.ecommerce_scene,
    )

    result = {
        "positive": positive,
        "negative": negative,
        "settings": {
            "subject": args.subject,
            "style": args.style,
            "lighting": args.lighting,
            "composition": args.composition,
            "quality_style": args.quality_style,
            "ecommerce_scene": args.ecommerce_scene,
        },
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved to: {args.output}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
