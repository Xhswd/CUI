# ComfyUI Assistant Technical Guidelines

This document provides the technical reference for the ComfyUI Assistant skill. The agent MUST read this before executing any ComfyUI operations.

## 1. ComfyUI API Reference

### Base URL
Default: `http://127.0.0.1:8188`

### Key Endpoints

#### System Info
```
GET /system_stats
```
Response:
```json
{
  "system": {
    "os": "linux",
    "python_version": "3.10.x",
    "devices": [
      {
        "name": "NVIDIA GeForce RTX 4090",
        "type": "cuda",
        "vram_total": 25757220864,
        "vram_free": 23456789012
      }
    ]
  }
}
```

#### Queue Prompt (Submit Workflow)
```
POST /prompt
Content-Type: application/json

{
  "prompt": { <workflow_json> },
  "client_id": "comfyui-assistant"
}
```
Response:
```json
{
  "prompt_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "number": 1,
  "node_errors": {}
}
```

#### Get Queue Status
```
GET /queue
```

#### Get History
```
GET /history/{prompt_id}
```

#### Get Image
```
GET /view?filename={filename}&subfolder={subfolder}&type={type}
```

#### Upload Image
```
POST /upload/image
Content-Type: multipart/form-data

image: <file>
overwrite: true
```

#### Object Info (Get Node Schemas)
```
GET /object_info/{node_class}
```

#### List Models
```
GET /object_info/CheckpointLoaderSimple
GET /object_info/LoraLoader
GET /object_info/ControlNetLoader
```

### WebSocket Monitoring
Connect to `ws://127.0.0.1:8188/ws?clientId=comfyui-assistant` to receive progress updates:
```json
{"type": "progress", "data": {"value": 5, "max": 20}}
{"type": "executing", "data": {"node": "3", "prompt_id": "xxx"}}
{"type": "executed", "data": {"node": "9", "output": {"images": [{"filename": "output.png", "subfolder": "", "type": "output"}]}}}
```

## 2. Workflow JSON Structure

ComfyUI workflows use a node-based JSON format. Each node has:
- `class_type`: The node type (e.g., "KSampler", "CLIPTextEncode")
- `inputs`: Input values and connections to other nodes

### Node Connection Format
```json
"inputs": {
  "model": ["4", 0],    // [source_node_id, output_index]
  "text": "a beautiful landscape"
}
```

### Common Node Types

| Node Class | Purpose | Key Inputs |
|------------|---------|------------|
| `CheckpointLoaderSimple` | Load checkpoint model | `ckpt_name` |
| `CLIPTextEncode` | Encode text prompt | `text`, `clip` |
| `KSampler` | Main sampling node | `model`, `positive`, `negative`, `latent_image`, `seed`, `steps`, `cfg`, `sampler_name`, `scheduler` |
| `VAEDecode` | Decode latent to image | `samples`, `vae` |
| `SaveImage` | Save output image | `images`, `filename_prefix` |
| `LoadImage` | Load input image | `image` |
| `VAEEncode` | Encode image to latent | `pixels`, `vae` |
| `LoraLoader` | Load LoRA model | `lora_name`, `strength_model`, `strength_clip` |
| `ControlNetApply` | Apply ControlNet | `conditioning`, `control_net`, `image` |
| `ControlNetLoader` | Load ControlNet model | `control_net_name` |
| `UpscaleLatent` | Upscale latent | `samples`, `upscale_method`, `crop` |
| `LatentUpscale` | Upscale latent image | `samples`, `upscale_method`, `width`, `height`, `crop` |
| `ImageScale` | Scale image | `image`, `upscale_method`, `width`, `height`, `crop` |

### Dynamic Workflow Builder

Prefer dynamic workflows for new generation sessions. The builder stores reusable node pointers and workflow blueprints in code, then assembles JSON at runtime.

**Model-aware menu for TUI:**
```bash
python3 {skill_dir}/scripts/workflow_builder.py menu \
  --model "{checkpoint_name}" \
  --image-type "{image_type}" \
  --json
```

The response is grouped into tiers:
- `basic`: txt2img, txt2img + LoRA, SD3/SD3.5 txt2img, FLUX txt2img
- `image`: img2img, inpainting
- `control`: ControlNet variants
- `advanced`: high-res fix and latent upscale
- `ecommerce`: product/model/background/try-on workflows

Only show workflows returned by the menu. This prevents SD1.5/SD2.x/SDXL/SD3/SD3.5/FLUX incompatible options from appearing in the TUI. SD3/SD3.5 workflows require separate text encoders (`clip_l`, `clip_g`, `t5xxl`) under ComfyUI's `models/text_encoders`. FLUX workflows require a diffusion model, CLIP-L, T5XXL, and VAE in their matching ComfyUI model folders.

**Build final workflow JSON after prompts are known:**
```bash
python3 {skill_dir}/scripts/workflow_builder.py build \
  --workflow-id "{workflow_id}" \
  --model "{checkpoint_name}" \
  --positive "{positive}" \
  --negative "{negative}" \
  --output "{comfyui_path}/user/workflows/{name}.json" \
  {workflow_specific_args}
```

For SD3/SD3.5 dynamic workflows, pass text encoder filenames when they differ from defaults:
```bash
--sd3-clip-l "sdv3/clip_l.safetensors" --sd3-clip-g "sdv3/clip_g.safetensors" --sd3-t5xxl "sdv3/t5xxl_fp16.safetensors"
```

For FLUX dynamic workflows, pass the split model component filenames:
```bash
--flux-unet "flux1-dev.safetensors" --flux-clip-l "clip_l.safetensors" --flux-t5xxl "t5xxl_fp16.safetensors" --flux-vae "ae.safetensors"
```

**Validate any generated or static workflow:**
```bash
python3 {skill_dir}/scripts/workflow_builder.py validate --workflow "{workflow_path}"
```

Use `--allow-placeholders` only for template inspection. Before submitting to ComfyUI, validation should pass without unresolved placeholders.

### Recommended Sampler Settings

| Style | Sampler | Scheduler | Steps | CFG |
|-------|---------|-----------|-------|-----|
| General | euler_ancestral | normal | 20-30 | 7-8 |
| Photorealistic | dpmpp_2m | karras | 25-35 | 7 |
| Anime | euler_ancestral | normal | 20-25 | 7 |
| Fast | euler | normal | 15-20 | 7 |
| Quality | dpmpp_2m_sde | karras | 30-50 | 5-7 |

## 3. Prompt Engineering Templates

### Positive Prompt Structure
```
[subject description], [style/mood], [quality boosters], [lighting], [composition/camera], [detail modifiers]
```

### Quality Booster Sets

#### Photorealistic
```
masterpiece, best quality, ultra detailed, high resolution, 8k, photorealistic, RAW photo, detailed skin texture, natural lighting, depth of field, film grain
```

#### Anime/Illustration
```
masterpiece, best quality, highly detailed, vibrant colors, sharp focus, illustration, anime style, detailed eyes, beautiful shading
```

#### Art/Painting
```
masterpiece, best quality, highly detailed, oil painting, artistic, expressive brushstrokes, rich colors, dramatic lighting, fine art
```

#### Concept Art
```
masterpiece, best quality, concept art, matte painting, detailed, epic, cinematic, volumetric lighting, trending on artstation
```

### Negative Prompt Templates

#### General
```
blurry, low quality, lowres, bad anatomy, bad hands, distorted, deformed, mutation, mutated, extra limbs, extra fingers, missing fingers, watermark, signature, text, logo, cropped, worst quality, jpeg artifacts, duplicate, morbid, mutilated
```

#### Photorealistic
```
painting, drawing, illustration, cartoon, anime, 3d render, cgi, sketch, blurry, low quality, lowres, bad anatomy, watermark, signature, text, deformed, distorted, disfigured, bad proportions, duplicate, out of frame, out of bounds
```

#### Anime
```
photorealistic, 3d, western cartoon, realistic, low quality, lowres, blurry, bad anatomy, bad hands, distorted, deformed, mutation, watermark, signature, text, worst quality, jpeg artifacts
```

### Style Keywords Reference

| Style | Keywords |
|-------|----------|
| Cyberpunk | cyberpunk, neon lights, futuristic, dystopian, rain, holographic, tech noir |
| Fantasy | fantasy, magical, ethereal, mystical, enchanted, fairy tale, otherworldly |
| Horror | horror, dark, eerie, creepy, gothic, macabre, unsettling, shadowy |
| Sci-Fi | sci-fi, space, futuristic, alien, technology, starship, cosmic |
| Steampunk | steampunk, victorian, brass, gears, clockwork, industrial, steam-powered |
| Watercolor | watercolor, soft edges, flowing, delicate, pastel, wet on wet technique |
| Oil Painting | oil painting, rich colors, textured, classical, chiaroscuro, impasto |
| Pixel Art | pixel art, 16-bit, retro, 8-bit, sprite, nostalgic, low resolution charm |
| Minimalist | minimalist, clean, simple, geometric, modern, sparse, negative space |
| Baroque | baroque, ornate, dramatic, gilded, elaborate, theatrical, grandiose |

### Lighting Keywords

| Lighting | Keywords |
|----------|----------|
| Natural | natural lighting, golden hour, sunlight, soft light, ambient |
| Dramatic | dramatic lighting, chiaroscuro, high contrast, rim lighting, spotlight |
| Cinematic | cinematic lighting, volumetric lighting, god rays, lens flare, bokeh |
| Studio | studio lighting, three-point lighting, softbox, key light, fill light |
| Neon | neon lighting, colorful lights, glow, luminescent, fluorescent |
| Moody | moody lighting, low key, shadows, atmospheric, fog, mist |

## 4. Model Download Reference

### HuggingFace Mirror
Primary mirror: `https://hf-mirror.com`

### Popular Model Repositories

| Model | Repo ID | Type |
|-------|---------|------|
| SDXL Base 1.0 | stabilityai/stable-diffusion-xl-base-1.0 | Checkpoint |
| SDXL Refiner 1.0 | stabilityai/stable-diffusion-xl-refiner-1.0 | Checkpoint |
| SD 1.5 | runwayml/stable-diffusion-v1-5 | Checkpoint |
| SD 2.1 | stabilityai/stable-diffusion-2-1 | Checkpoint |
| SD 3 Medium | stabilityai/stable-diffusion-3-medium | Checkpoint |
| FLUX.1-schnell | black-forest-labs/FLUX.1-schnell | Checkpoint |
| FLUX.1-dev | black-forest-labs/FLUX.1-dev | Checkpoint |

### Download URL Pattern
```
https://hf-mirror.com/{repo_id}/resolve/main/{filename}
```

Example:
```
https://hf-mirror.com/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
```

### Using huggingface-cli with Mirror
```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download {repo_id} {filename} --local-dir {output_dir}
```

### Using wget/curl with Mirror
```bash
wget -c https://hf-mirror.com/{repo_id}/resolve/main/{filename} -O {output_path}
```

## 5. ComfyUI Directory Structure

```
ComfyUI/
├── main.py                    # Entry point
├── models/
│   ├── checkpoints/           # Main model files (.safetensors, .ckpt)
│   ├── loras/                 # LoRA models
│   ├── controlnet/            # ControlNet models
│   ├── upscale_models/        # Upscale models (ESRGAN, etc.)
│   ├── vae/                   # VAE models
│   ├── embeddings/            # Textual inversion embeddings
│   └── clip/                  # CLIP models
├── output/                    # Generated images
├── input/                     # Input images
├── user/
│   └── workflows/             # Saved workflows
├── web/                       # Frontend files
└── custom_nodes/              # Custom node extensions
```

## 6. Troubleshooting

### ComfyUI Won't Start
- Check Python version: `python3 --version` (needs 3.9+)
- Check dependencies: `pip install -r requirements.txt`
- Check GPU: `nvidia-smi`
- Check port conflict: `lsof -i :8188`

### Out of VRAM
- Reduce image resolution (512x512 instead of 1024x1024)
- Use `--lowvram` flag: `python main.py --lowvram`
- Use `--cpu` flag for CPU-only mode: `python main.py --cpu`

### Model Not Found
- Ensure model is in `models/checkpoints/` directory
- Check file extension (.safetensors, .ckpt)
- Refresh model list via API: `POST /refresh_models`

### Generation Stuck
- Check queue: `GET /queue`
- Interrupt current: `POST /interrupt`
- Clear queue: `POST /queue`, `{"delete": ["all"]}`

## 7. E-Commerce Scene Reference

### E-Commerce Workflow Templates

| Template File | Function | Description |
|---------------|----------|-------------|
| `ecommerce_product.json` | 电商商品图 | Product photography, studio lighting, clean output |
| `ecommerce_model.json` | 电商模特图 | Fashion model photography, img2img based |
| `ecommerce_bg_replace.json` | 电商换背景 | Background replacement via img2img |
| `ecommerce_tryon.json` | 虚拟试穿 | Virtual try-on via ControlNet |

### E-Commerce Scene Templates

| Scene ID | Name | Quality Style | Lighting | Composition |
|----------|------|---------------|----------|-------------|
| `product_white_bg` | 白底商品图 | ecommerce_product | ecommerce_studio | ecommerce_center |
| `product_lifestyle` | 生活场景图 | ecommerce_lifestyle | ecommerce_window | ecommerce_lifestyle |
| `product_luxury` | 奢侈品展示 | ecommerce_product | ecommerce_dramatic | ecommerce_center |
| `product_flatlay` | 平铺展示图 | ecommerce_product | ecommerce_studio | ecommerce_flatlay_comp |
| `model_studio` | 棚拍模特图 | ecommerce_model | ecommerce_studio | full_body |
| `model_outdoor` | 户外模特图 | ecommerce_model | natural | ecommerce_lifestyle |
| `model_street` | 街拍模特图 | ecommerce_model | natural | full_body |
| `model_detail` | 细节特写图 | ecommerce_product | ecommerce_studio | ecommerce_detail |
| `tryon_virtual` | 虚拟试穿 | ecommerce_model | ecommerce_studio | full_body |
| `bg_replace_clean` | 换纯色/白底 | ecommerce_product | ecommerce_studio | ecommerce_center |
| `bg_replace_scene` | 换场景背景 | ecommerce_lifestyle | natural | ecommerce_lifestyle |

### E-Commerce Quality Boosters

**ecommerce_product**: `masterpiece, best quality, ultra detailed, high resolution, 8k, commercial photography, product photography, studio lighting, clean background, sharp focus, professional, catalog photo, white background`

**ecommerce_model**: `masterpiece, best quality, ultra detailed, high resolution, 8k, fashion photography, editorial, professional model, studio lighting, detailed skin texture, magazine quality, vogue, high fashion`

**ecommerce_lifestyle**: `masterpiece, best quality, ultra detailed, high resolution, 8k, lifestyle photography, natural setting, aspirational, warm lighting, cozy atmosphere, realistic, inviting`

### E-Commerce Negative Prompts

**ecommerce_product**: `blurry, low quality, lowres, watermark, signature, text, logo, brand name, deformed, distorted, cropped, out of frame, background noise, cluttered, dusty, scratched, damaged, worst quality, jpeg artifacts, overexposed, underexposed, color cast`

**ecommerce_model**: `blurry, low quality, lowres, bad anatomy, bad hands, extra fingers, missing fingers, deformed, distorted, disfigured, bad proportions, watermark, signature, text, logo, worst quality, jpeg artifacts, unnatural pose, stiff, awkward`

**ecommerce_lifestyle**: `blurry, low quality, lowres, watermark, signature, text, logo, deformed, distorted, worst quality, jpeg artifacts, artificial, fake, staged looking, unrealistic, overprocessed`

### E-Commerce Recommended Settings

| Product Type | Recommended Model | Resolution | Steps | CFG |
|-------------|-------------------|------------|-------|-----|
| Fashion/Clothing | SDXL / FLUX | 1024x1536 | 30 | 7 |
| Electronics | SDXL | 1024x1024 | 25 | 7 |
| Cosmetics | SDXL | 1024x1024 | 30 | 7 |
| Food/Beverage | SDXL / FLUX | 1024x1024 | 25 | 7 |
| Jewelry/Accessories | SDXL | 1024x1024 | 30 | 7 |
| Home/Furniture | SDXL | 1024x768 | 25 | 7 |
| Sports/Outdoors | SDXL | 1024x1024 | 25 | 7 |

### E-Commerce Batch Generation Angles

| Angle | Prompt Keywords | Use Case |
|-------|----------------|----------|
| Front | `front view, straight-on, centered` | Main product image |
| Side | `side view, profile, lateral` | Detail showcase |
| 45-degree | `three-quarter view, 45 degree angle` | Dimensional display |
| Top-down | `top view, overhead, bird's eye` | Flat lay / layout |
| Back | `back view, rear, reverse side` | Full product view |
| Detail | `close-up, macro, detail shot, texture` | Material/quality showcase |

### E-Commerce Platform Image Specs

| Platform | Main Image | Detail Image | Format |
|----------|-----------|--------------|--------|
| 淘宝/天猫 | 800x800+ | 750xauto | JPG/PNG |
| 京东 | 800x800+ | 750xauto | JPG/PNG |
| 拼多多 | 750xauto | 750xauto | JPG/PNG |
| 抖音 | 800x800+ | 750xauto | JPG/PNG |
| 小红书 | 1080x1440 | 1080xauto | JPG/PNG |
| Amazon | 1000x1000+ | 1000xauto | JPG/PNG |
| Shopify | 2048x2048 | 2048xauto | JPG/PNG |

## 8. Model Presets & Recommendations

### Type-Aware Recommendation Rule

Always ask for the target image type before showing model choices. Local model lists must be filtered by image type:

```bash
python3 {skill_dir}/scripts/model_presets.py recommend-type \
  --image-type "{image_type}" \
  --comfyui-path "{comfyui_path}" \
  --family any \
  --json
```

Only display models returned in `local_checkpoints`; do not show unrelated local checkpoints just because they exist. If no local match is found for a custom "Other" type, search online:

```bash
python3 {skill_dir}/scripts/model_presets.py recommend-type \
  --image-type "{custom_image_type}" \
  --comfyui-path "{comfyui_path}" \
  --search-if-missing \
  --mirror hf-mirror \
  --json
```

Use `recommended_combos` to ask whether the user wants the checkpoint + LoRA + VAE + ControlNet combination. Download only after asking.

### Recommended Checkpoint Models

| Model | Family | Best For | Resolution | Steps | CFG | Sampler |
|-------|--------|----------|------------|-------|-----|---------|
| SDXL Base 1.0 | SDXL | 通用, 电商, 风景 | 1024x1024 | 25 | 7 | dpmpp_2m/karras |
| DreamShaper XL Turbo | SDXL | 快速生成, 创意 | 1024x1024 | 8 | 2 | dpmpp_sde/karras |
| Juggernaut XL v9 | SDXL | 电商, 写实照片 | 1024x1024 | 20 | 4.5 | dpmpp_2m/karras |
| RealVis XL V4.0 | SDXL | 写实人物, 电商模特 | 1024x1024 | 25 | 7 | dpmpp_2m/karras |
| SD 1.5 | SD15 | 通用, LoRA兼容 | 512x512 | 25 | 7 | euler_ancestral/normal |
| DreamShaper 8 | SD15 | 通用, 动漫, 写实 | 512x512 | 25 | 7 | euler_ancestral/normal |
| ReV Animated | SD15 | 动漫, 插画 | 512x512 | 25 | 7 | euler_ancestral/normal |
| FLUX.1 Dev | FLUX | 通用, 文字渲染 | 1024x1024 | 20 | 3.5 | euler/simple |
| FLUX.1 Schnell | FLUX | 极速出图, 批量 | 1024x1024 | 4 | 1 | euler/simple |

### Recommended Model Combos

**电商商品图:**
- SDXL: Juggernaut XL + product-photo-xl LoRA (strength 0.7) + SDXL VAE
- FLUX: FLUX Dev (no LoRA needed)

**电商模特图:**
- SDXL: RealVis XL + detail-tweaker-xl LoRA (strength 0.8) + SDXL VAE

**动漫/插画:**
- SDXL: DreamShaper XL + anime-style-xl LoRA (strength 0.7)
- SD15: ReV Animated + SD 1.5 VAE

**写实照片:**
- SDXL: Juggernaut XL + detail-tweaker-xl LoRA (strength 0.8)

**快速生成:**
- SDXL: DreamShaper XL Turbo (8步) + SDXL VAE FP16
- FLUX: FLUX Schnell (1-4步)

### Recommended VAE Models

| VAE | Family | Best For |
|-----|--------|----------|
| SDXL VAE | SDXL | SDXL通用, 色彩改善 |
| SDXL VAE FP16 | SDXL | 低显存场景 |
| SD VAE FT MSE | SD15 | SD1.5通用, 色彩改善 |

### Recommended Upscale Models

| Model | Scale | Best For |
|-------|-------|----------|
| 4x UltraSharp | 4x | 通用放大, 细节保留 |
| Real-ESRGAN x4 | 4x | 真实照片, 老照片修复 |
| 4x NMKD Superscale | 4x | 高质量放大 |
| 2x Digital Art | 2x | 插画, 动漫 |

## 9. LoRA Usage Guide

### LoRA Workflow
Use `workflows/txt2img_lora.json` which includes a `LoraLoader` node (node 10).

### LoRA Loader Node Structure
```json
"10": {
  "class_type": "LoraLoader",
  "inputs": {
    "lora_name": "lora_file.safetensors",
    "strength_model": 0.8,
    "strength_clip": 0.8,
    "model": ["4", 0],
    "clip": ["4", 1]
  }
}
```

### LoRA Strength Guidelines
| Strength | Effect |
|----------|--------|
| 0.3-0.5 | Subtle effect, barely noticeable |
| 0.5-0.7 | Moderate effect, good for style LoRAs |
| 0.7-0.9 | Strong effect, recommended for detail LoRAs |
| 0.9-1.0 | Maximum effect, may cause artifacts |

### Popular LoRA Presets

| Preset | Family | Strength | Best For |
|--------|--------|----------|----------|
| detail-tweaker-xl | SDXL | 0.8 | 细节增强, 纹理提升 |
| add-details-xl | SDXL | 0.6 | 细节, 锐度 |
| product-photo-xl | SDXL | 0.7 | 电商产品摄影 |
| anime-style-xl | SDXL | 0.7 | 动漫风格 |
| oil-painting-xl | SDXL | 0.7 | 油画风格 |
| flux-realism | FLUX | 0.7 | FLUX写实增强 |

## 10. Image Post-Processing

### Available Operations
- **Remove Background** (`remove-bg`): Uses rembg for AI-powered background removal, outputs transparent PNG
- **Resize** (`resize`): Resize with lanczos/bicubic/bilinear/nearest resampling
- **Format Convert** (`convert`): PNG/JPG/WEBP/BMP/TIFF conversion
- **Enhance** (`enhance`): Brightness, contrast, saturation, sharpness adjustment + denoise
- **Crop Square** (`square`): Center/top/bottom crop to 1:1 ratio
- **Add Padding** (`pad`): Add padding for target ratio (1:1, 4:3, 16:9) with white/black/transparent
- **Batch Process** (`batch`): Apply any operation to all images in a directory

### Dependencies
- Install the full helper dependency set from this repo: `pip install -r requirements.txt`
- Minimum for non-background image operations: `pip install Pillow`
- Background removal requires: `pip install rembg`

## 11. History Management

### Database
History is stored in `{comfyui_path}/user/history.json` (JSON format).

### Features
- **List & Search**: Browse history with filters (function, model, tag, favorites, text search)
- **Favorites**: Star/unstar generations for quick access
- **Rating**: Rate generations 1-5 stars
- **Tags**: Add custom tags for organization
- **Notes**: Add free-text notes to any generation
- **Compare**: Side-by-side comparison of multiple generations
- **Statistics**: View generation stats (total count, top models, top functions, date range)
- **Export**: Export to JSON or CSV format
- **Import**: Import from ComfyUI output directory
