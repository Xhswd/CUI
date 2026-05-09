#!/usr/bin/env python3
"""
ComfyUI API Client - Submit workflows and monitor generation progress
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
import uuid
from pathlib import Path

try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False


class ComfyUIClient:
    def __init__(self, api_url="http://127.0.0.1:8188"):
        self.api_url = api_url.rstrip("/")
        self.client_id = str(uuid.uuid4())

    def _request(self, method, path, data=None):
        url = f"{self.api_url}{path}"
        headers = {"Content-Type": "application/json"} if data else {}
        body = json.dumps(data).encode("utf-8") if data else None

        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")
            print(f"HTTP Error {e.code}: {body_text}", file=sys.stderr)
            return None
        except urllib.error.URLError as e:
            print(f"Connection Error: {e.reason}", file=sys.stderr)
            return None

    def get_system_stats(self):
        return self._request("GET", "/system_stats")

    def get_queue(self):
        return self._request("GET", "/queue")

    def get_history(self, prompt_id=None):
        if prompt_id:
            return self._request("GET", f"/history/{prompt_id}")
        return self._request("GET", "/history")

    def interrupt(self):
        return self._request("POST", "/interrupt")

    def clear_queue(self):
        return self._request("POST", "/queue", {"delete": ["all"]})

    def refresh_models(self):
        return self._request("POST", "/refresh_models")

    def submit_workflow(self, workflow_data):
        payload = {
            "prompt": workflow_data,
            "client_id": self.client_id,
        }
        result = self._request("POST", "/prompt", payload)
        if result and "prompt_id" in result:
            return result["prompt_id"]
        return None

    def upload_image(self, image_path, subfolder="", overwrite=True):
        url = f"{self.api_url}/upload/image"
        filepath = Path(image_path)
        if not filepath.exists():
            print(f"Image not found: {image_path}", file=sys.stderr)
            return None

        boundary = uuid.uuid4().hex
        lines = []

        lines.append(f"--{boundary}")
        lines.append(f'Content-Disposition: form-data; name="image"; filename="{filepath.name}"')
        lines.append("Content-Type: application/octet-stream")
        lines.append("")
        lines.append("")

        lines.append(f"--{boundary}")
        lines.append(f'Content-Disposition: form-data; name="subfolder"')
        lines.append("")
        lines.append(subfolder)

        lines.append(f"--{boundary}")
        lines.append(f'Content-Disposition: form-data; name="overwrite"')
        lines.append("")
        lines.append(str(overwrite).lower())

        lines.append(f"--{boundary}--")

        header_part = "\r\n".join(lines[:-1]) + "\r\n\r\n"
        footer_part = "\r\n" + lines[-1]

        with open(filepath, "rb") as f:
            image_data = f.read()

        body = header_part.encode("utf-8") + image_data + footer_part.encode("utf-8")

        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"Upload failed: {e}", file=sys.stderr)
            return None

    def get_image(self, filename, subfolder="", img_type="output"):
        url = f"{self.api_url}/view?filename={urllib.parse.quote(filename)}&subfolder={urllib.parse.quote(subfolder)}&type={img_type}"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read()
        except Exception as e:
            print(f"Failed to get image: {e}", file=sys.stderr)
            return None

    def wait_for_completion(self, prompt_id, timeout=600, poll_interval=2):
        start_time = time.time()
        while time.time() - start_time < timeout:
            history = self.get_history(prompt_id)
            if history and prompt_id in history:
                status = history[prompt_id].get("status", {})
                if status.get("completed", False) or status.get("status_str") == "success":
                    outputs = history[prompt_id].get("outputs", {})
                    images = []
                    for node_id, node_output in outputs.items():
                        if "images" in node_output:
                            for img in node_output["images"]:
                                images.append(img)
                    return {"status": "success", "images": images}
                if status.get("status_str") == "error":
                    return {"status": "error", "message": status}
            time.sleep(poll_interval)
        return {"status": "timeout"}

    def monitor_progress(self, prompt_id, timeout=600):
        if HAS_WEBSOCKET:
            return self._monitor_ws(prompt_id, timeout)
        else:
            return self.wait_for_completion(prompt_id, timeout)

    def _monitor_ws(self, prompt_id, timeout=600):
        ws_url = f"ws://{self.api_url.replace('http://', '').replace('https://', '')}/ws?clientId={self.client_id}"
        try:
            ws = websocket.create_connection(ws_url, timeout=timeout)
        except Exception as e:
            print(f"WebSocket connection failed: {e}, falling back to polling", file=sys.stderr)
            return self.wait_for_completion(prompt_id, timeout)

        start_time = time.time()
        try:
            while time.time() - start_time < timeout:
                try:
                    msg = json.loads(ws.recv())
                except websocket.WebSocketTimeoutException:
                    continue
                except Exception:
                    break

                msg_type = msg.get("type")
                data = msg.get("data", {})

                if msg_type == "progress":
                    value = data.get("value", 0)
                    max_val = data.get("max", 1)
                    pct = int(value / max_val * 100) if max_val > 0 else 0
                    print(f"\r进度: {value}/{max_val} ({pct}%)", end="", flush=True)

                elif msg_type == "executing":
                    if data.get("node") is None and data.get("prompt_id") == prompt_id:
                        print("\n生成完成!")
                        ws.close()
                        return self._get_result_from_history(prompt_id)

                elif msg_type == "execution_error":
                    print(f"\n执行错误: {data}")
                    ws.close()
                    return {"status": "error", "message": data}

                elif msg_type == "execution_cached":
                    pass

        except Exception as e:
            print(f"\n监控异常: {e}", file=sys.stderr)
        finally:
            try:
                ws.close()
            except Exception:
                pass

        return {"status": "timeout"}

    def _get_result_from_history(self, prompt_id):
        history = self.get_history(prompt_id)
        if history and prompt_id in history:
            outputs = history[prompt_id].get("outputs", {})
            images = []
            for node_id, node_output in outputs.items():
                if "images" in node_output:
                    for img in node_output["images"]:
                        images.append(img)
            return {"status": "success", "images": images}
        return {"status": "unknown"}


def fill_workflow(workflow, model_name, positive_prompt, negative_prompt,
                  seed=None, steps=None, cfg=None, width=None, height=None,
                  sampler=None, scheduler=None, input_image=None):
    result = json.loads(json.dumps(workflow))

    # Build a map of which CLIPTextEncode nodes are positive/negative
    # by tracing KSampler's positive/negative input connections
    positive_node_ids = set()
    negative_node_ids = set()
    for node_id, node in result.items():
        if node.get("class_type") == "KSampler":
            pos_input = node.get("inputs", {}).get("positive", [])
            if isinstance(pos_input, list) and len(pos_input) > 0:
                positive_node_ids.add(str(pos_input[0]))
            neg_input = node.get("inputs", {}).get("negative", [])
            if isinstance(neg_input, list) and len(neg_input) > 0:
                negative_node_ids.add(str(neg_input[0]))

    for node_id, node in result.items():
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})

        if class_type == "CheckpointLoaderSimple":
            if model_name is not None:
                inputs["ckpt_name"] = model_name

        elif class_type == "CLIPTextEncode":
            if node_id in positive_node_ids:
                inputs["text"] = positive_prompt
            elif node_id in negative_node_ids:
                inputs["text"] = negative_prompt

        elif class_type == "KSampler":
            if seed is not None:
                inputs["seed"] = seed
            if steps is not None:
                inputs["steps"] = steps
            if cfg is not None:
                inputs["cfg"] = cfg
            if sampler is not None:
                inputs["sampler_name"] = sampler
            if scheduler is not None:
                inputs["scheduler"] = scheduler

        elif class_type == "EmptyLatentImage":
            if width is not None:
                inputs["width"] = width
            if height is not None:
                inputs["height"] = height

        elif class_type in ("LoadImage",):
            if input_image:
                inputs["image"] = input_image

    return result


def main():
    parser = argparse.ArgumentParser(description="ComfyUI API Client")
    parser.add_argument("--api-url", default="http://127.0.0.1:8188", help="ComfyUI API URL")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("stats", help="Get system stats")

    submit_parser = subparsers.add_parser("submit", help="Submit a workflow")
    submit_parser.add_argument("--workflow", required=True, help="Path to workflow JSON")
    submit_parser.add_argument("--model", help="Model name")
    submit_parser.add_argument("--positive", help="Positive prompt")
    submit_parser.add_argument("--negative", help="Negative prompt")
    submit_parser.add_argument("--seed", type=int, help="Random seed")
    submit_parser.add_argument("--steps", type=int, help="Sampling steps")
    submit_parser.add_argument("--cfg", type=float, help="CFG scale")
    submit_parser.add_argument("--width", type=int, help="Image width")
    submit_parser.add_argument("--height", type=int, help="Image height")
    submit_parser.add_argument("--sampler", help="Sampler name")
    submit_parser.add_argument("--scheduler", help="Scheduler name")
    submit_parser.add_argument("--input-image", help="Input image filename")
    submit_parser.add_argument("--wait", action="store_true", help="Wait for completion")
    submit_parser.add_argument("--timeout", type=int, default=600, help="Wait timeout in seconds")

    monitor_parser = subparsers.add_parser("monitor", help="Monitor a running prompt")
    monitor_parser.add_argument("--prompt-id", required=True, help="Prompt ID to monitor")
    monitor_parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds")

    upload_parser = subparsers.add_parser("upload", help="Upload an image")
    upload_parser.add_argument("--image", required=True, help="Path to image file")

    subparsers.add_parser("queue", help="Get queue status")

    history_parser = subparsers.add_parser("history", help="Get generation history")
    history_parser.add_argument("--prompt-id", help="Specific prompt ID")

    subparsers.add_parser("interrupt", help="Interrupt current generation")

    subparsers.add_parser("refresh", help="Refresh model list")

    args = parser.parse_args()
    client = ComfyUIClient(args.api_url)

    if args.command == "stats":
        result = client.get_system_stats()
        print(json.dumps(result, indent=2))

    elif args.command == "submit":
        with open(args.workflow, "r", encoding="utf-8") as f:
            workflow = json.load(f)

        if any([args.model, args.positive, args.negative, args.seed, args.steps,
                args.cfg, args.width, args.height, args.sampler, args.scheduler, args.input_image]):
            workflow = fill_workflow(
                workflow,
                model_name=args.model,
                positive_prompt=args.positive or "",
                negative_prompt=args.negative or "",
                seed=args.seed,
                steps=args.steps,
                cfg=args.cfg,
                width=args.width,
                height=args.height,
                sampler=args.sampler,
                scheduler=args.scheduler,
                input_image=args.input_image,
            )

        prompt_id = client.submit_workflow(workflow)
        if prompt_id:
            print(f"Submitted: {prompt_id}")
            if args.wait:
                result = client.monitor_progress(prompt_id, timeout=args.timeout)
                print(json.dumps(result, indent=2))
        else:
            print("Failed to submit workflow", file=sys.stderr)
            sys.exit(1)

    elif args.command == "monitor":
        result = client.monitor_progress(args.prompt_id, timeout=args.timeout)
        print(json.dumps(result, indent=2))

    elif args.command == "upload":
        result = client.upload_image(args.image)
        print(json.dumps(result, indent=2))

    elif args.command == "queue":
        result = client.get_queue()
        print(json.dumps(result, indent=2))

    elif args.command == "history":
        result = client.get_history(args.prompt_id)
        print(json.dumps(result, indent=2))

    elif args.command == "interrupt":
        result = client.interrupt()
        print(json.dumps(result, indent=2))

    elif args.command == "refresh":
        result = client.refresh_models()
        print(json.dumps(result, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
