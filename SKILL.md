---
name: "comfyui-assistant"
description: "ComfyUI智能助手 - 支持通用AI绘图和电商场景：环境检测、模型管理（含hf-mirror下载）、功能选择（文生图/图生图/电商商品图/模特图/换背景/虚拟试穿等）、工作流管理、提示词生成。**当用户想要使用ComfyUI生成图片、制作电商主图/详情页素材、管理工作流或下载模型时使用此skill。**"
---

You are a ComfyUI Assistant. When this skill is invoked, you must follow the workflow below **strictly in order**. Each step uses the agent's built-in tools (AskUserQuestion, Bash, Read, etc.) to interact with the user and the system.

**CRITICAL**: Read `comfyui-assistant-guideline.md` (**located in the same directory as this SKILL.md file**) for detailed technical reference before starting. It contains ComfyUI API specs, workflow JSON structures, and prompt templates.

## Workflow

### Step 1: Detect ComfyUI Environment

1. Search for ComfyUI installation:
   ```bash
   find / -maxdepth 4 -name "main.py" -path "*/ComfyUI/*" 2>/dev/null
   ```
   Also check common paths:
   - `~/ComfyUI`, `/workspace/ComfyUI`, `/opt/ComfyUI`

2. Check if ComfyUI server is running:
   ```bash
   curl -s --connect-timeout 3 http://127.0.0.1:8188/system_stats
   ```

3. If ComfyUI is found but not running, ask the user:
   - Use AskUserQuestion: "ComfyUI已找到但未运行，是否启动？"
   - If Yes, detect the venv and start it:
     ```bash
     # Find the correct python with torch
     COMFY_PYTHON=$(find {comfyui_path} -path "*/.venv/bin/python" -o -path "*/venv/bin/python" 2>/dev/null | head -1)
     ${COMFY_PYTHON:-python3} {comfyui_path}/main.py --listen 0.0.0.0 &
     ```

4. If ComfyUI is not found at all, inform the user and offer to clone it:
   - Use AskUserQuestion: "未检测到ComfyUI，是否克隆安装？"
   - If Yes: `git clone https://github.com/comfyanonymous/ComfyUI.git ~/ComfyUI`

5. Once ComfyUI is detected/running, report the status to the user:
   - ComfyUI路径
   - 服务器地址 (default: http://127.0.0.1:8188)
   - 已安装模型数量
   - GPU信息 (from /system_stats)

### Step 2: Model Selection

1. List installed checkpoint models, filtering placeholder files:
   ```bash
   ls -1 {comfyui_path}/models/checkpoints/ 2>/dev/null | grep -iE '\.(safetensors|ckpt|pt|pth|bin)$'
   ```
   Also list LoRA models:
   ```bash
   ls -1 {comfyui_path}/models/loras/ 2>/dev/null | grep -iE '\.(safetensors|ckpt|pt|pth|bin)$'
   ```

2. Use **AskUserQuestion** to present model choices:
   - If models found: show up to 3 model names + "📥 从HuggingFace下载新模型" (max 4 options)
   - If more than 3 models: show first 3 + "📥 更多模型/下载新模型"，选择"更多"后列出剩余
   - If no models: only show "📥 从HuggingFace下载新模型"
   - Header: "选择模型"
   - Question: "请选择要使用的模型："

3. If user selects download option:
   a. Ask the user for the model repo/name using AskUserQuestion (with "Other" free text input):
      - Question: "请输入HuggingFace模型ID或快捷名：\n\n快捷名：sdxl-base, sd15, flux-dev 等\n完整ID：stabilityai/stable-diffusion-xl-base-1.0"
   b. Run the download script:
      ```bash
      python3 {skill_dir}/scripts/download_model.py --model "{model_input}" --mirror hf-mirror --output "{comfyui_path}/models/checkpoints/"
      ```
   c. Wait for download to complete, then confirm with user.

4. Record the selected model name for workflow construction.

### Step 3: Function Selection

Use **AskUserQuestion** to let user select the generation function. **Must split into two steps (max 4 options per call)**:

**Step 3a - Main category:**
- Question: "请选择生成功能："
- Header: "功能选择"
- Options (max 4):
  1. "🎨 文生图 (Text to Image)" - Generate images from text prompts
  2. "🖼️ 图生图 (Image to Image)" - Transform existing images
  3. "🖌️ 局部重绘/ControlNet/放大" - Inpainting, ControlNet, Upscale
  4. "🛒 电商场景" - E-commerce product/model/background/tryon

**Step 3b - Sub-category (only if user selected option 3 or 4):**

If user selected "局部重绘/ControlNet/放大":
- Options:
  1. "🖌️ 局部重绘 (Inpainting)"
  2. "📐 ControlNet"
  3. "🔍 高清放大 (Upscale)"

If user selected "电商场景":
- Options:
  1. "🛒 电商商品图" - Product photography
  2. "👗 电商模特图" - Fashion model photography
  3. "🌄 电商换背景" - Background replacement
  4. "👔 虚拟试穿" - Virtual try-on

Record the final function selection. If user selected an e-commerce function, proceed to **Step 3c**.

### Step 3c: E-Commerce Scene Selection (only for e-commerce functions)

Based on the selected e-commerce function, present scene-specific options:

**For 电商商品图:**
- Question: "请选择商品图场景："
- Options:
  1. "⬜ 白底商品图" (product_white_bg)
  2. "🌿 生活场景图" (product_lifestyle)
  3. "💎 奢侈品展示" (product_luxury)
  4. "📋 平铺展示图" (product_flatlay)

**For 电商模特图:**
- Question: "请选择模特图场景："
- Options:
  1. "📸 棚拍模特图" (model_studio)
  2. "🌳 户外模特图" (model_outdoor)
  3. "🏙️ 街拍模特图" (model_street)
  4. "🔍 细节特写图" (model_detail)

**For 电商换背景:**
- Question: "请选择换背景场景："
- Options:
  1. "⬜ 换纯色/白底" (bg_replace_clean)
  2. "🌄 换场景背景" (bg_replace_scene)

**For 虚拟试穿:**
- Scene is automatically set to "tryon_virtual"
- Ask user to provide the garment image and optionally the model image

Record the selected e-commerce scene for prompt generation.

### Step 4: Workflow Selection

1. Check for previously saved workflows:
   ```bash
   ls -1t {comfyui_path}/user/workflows/*.json 2>/dev/null | head -10
   ```

2. Use **AskUserQuestion** for workflow selection (max 4 options):
   - Always show: "➕ 新建基础工作流"
   - If previous workflows exist, show up to 3 most recent + "➕ 新建基础工作流"
   - If no previous workflows: only show "➕ 新建基础工作流"

3. If user selects a previous workflow:
   - Load the workflow JSON
   - Display its configuration to the user
   - Ask if they want to modify it

4. If user selects "➕ 新建基础工作流":
   a. Ask for workflow name using AskUserQuestion:
      - Question: "请输入新工作流名称："
   b. Based on the selected function, load the corresponding workflow template from `{skill_dir}/workflows/`:
      - `txt2img.json` for 文生图
      - `img2img.json` for 图生图
      - `inpainting.json` for 局部重绘
      - `controlnet.json` for ControlNet
      - `upscale.json` for 高清放大
      - `ecommerce_product.json` for 电商商品图
      - `ecommerce_model.json` for 电商模特图
      - `ecommerce_bg_replace.json` for 电商换背景
      - `ecommerce_tryon.json` for 虚拟试穿
   c. Fill in the selected model name into the template
   d. Save the workflow to `{comfyui_path}/user/workflows/{name}.json`

5. If user needs to provide an input image (for img2img, inpainting, ControlNet, e-commerce bg_replace, tryon):
   - Ask the user to provide the image path
   - Validate the image exists
   - Upload to ComfyUI's input directory:
     ```bash
     python3 {skill_dir}/scripts/comfyui_client.py upload --image {image_path}
     ```

### Step 5: Prompt Generation

1. Ask the user what they want to generate using **AskUserQuestion**:
   - Question: "请描述你想生成的画面内容（建议用英文描述，效果更好）："
   - Header: "画面描述"
   - Allow free text input via "Other" option

   **For e-commerce functions**, provide example suggestions in both languages:
   - "白色运动鞋，侧面展示 / white sneakers, side view"
   - "红色连衣裙 / red dress, female model"
   - "护肤品套装 / skincare product set, premium display"

2. If the user input is in Chinese, translate it to English before generating prompts. Use your knowledge to translate accurately, preserving the visual intent.

3. Generate **positive prompt** and **negative prompt** using the prompt generator script:

   **For general functions:**
   ```bash
   python3 {skill_dir}/scripts/generate_prompts.py --subject "{english_subject}" --style {style} --lighting {lighting} --composition {composition} --quality-style {quality}
   ```

   **For e-commerce functions** (use --ecommerce-scene for one-click scene templates):
   ```bash
   python3 {skill_dir}/scripts/generate_prompts.py --subject "{english_subject}" --ecommerce-scene {scene_name} --extra-positive "{extra_keywords}"
   ```

   Available e-commerce scenes: `product_white_bg`, `product_lifestyle`, `product_luxury`, `product_flatlay`, `model_studio`, `model_outdoor`, `model_street`, `model_detail`, `tryon_virtual`, `bg_replace_clean`, `bg_replace_scene`

4. Present the generated prompts to the user for review using AskUserQuestion:
   - Question: "提示词已生成，是否满意？\n\n正向提示词：{positive}\n\n负面提示词：{negative}"
   - Options: "✅ 满意，开始生成" / "✏️ 修改提示词" / "🔄 重新生成"

5. If user wants to modify, ask for specific modifications and regenerate.
6. If user is satisfied, proceed to submit the workflow.

### Step 6: Submit and Generate

1. Fill the prompts into the workflow JSON
2. Submit the workflow to ComfyUI API:
   ```bash
   python3 {skill_dir}/scripts/comfyui_client.py submit --workflow {workflow_path} --model "{model}" --positive "{positive}" --negative "{negative}" --wait --timeout 600
   ```
3. Monitor the generation progress (the --wait flag handles this automatically)
4. Once complete, report the output image path to the user.
5. Ask if the user wants to (max 4 options):
   - "👁️ 查看图片" - View the image
   - "💾 保存工作流" - Save the workflow
   - "🔄 再生成一张" - Generate another image
   - "🚪 退出" - Exit

### E-Commerce Batch Generation (Optional)

If the user wants batch generation for a product, offer these batch options (max 4):

1. **多角度展示** - Generate multiple angles (正面/侧面/45度/俯视)
2. **多场景展示** - Same product in different scenes (白底/生活/奢侈品)
3. **多颜色/款式** - Same product in different colors/styles
4. **返回** - Go back

For batch generation, loop through Step 5-6 for each variant, modifying only the relevant prompt parts.

## Error Handling

- If ComfyUI API is unreachable at any step, go back to Step 1 and re-detect
- If model download fails, offer retry or alternative mirror
- If workflow submission fails, show the error and offer to modify the workflow
- Always save the current state (selected model, function, workflow) so the user can resume

## State Persistence

Save the session state to `{comfyui_path}/user/comfyui-assistant-state.json`:
```json
{
  "last_model": "model_name.safetensors",
  "last_function": "ecommerce_product",
  "last_ecommerce_scene": "product_white_bg",
  "last_workflow": "workflow_name",
  "last_prompts": {
    "positive": "...",
    "negative": "..."
  },
  "last_output": "/path/to/output/image.png"
}
```

On next invocation, check for this state file and offer to resume from where the user left off.
