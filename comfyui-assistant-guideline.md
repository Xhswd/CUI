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
