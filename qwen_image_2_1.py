#####################################################
# Test script for QwenImage21Pipeline
# This script demonstrates how to use the QwenImage21Pipeline to generate images from text prompts and edit existing images based on a given prompt. 
# It loads an input image, applies the editing pipeline, and saves the output image.
# usage: python test_qwen_image_2_1.py --mode t2i
# usage: python test_qwen_image_2_1.py --mode edit --image data/ll.jpg
#####################################################
import argparse
import os
import time
from pathlib import Path

import torch
from PIL import Image
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import QwenImage21Pipeline

MODEL_ID = "Qwen/Qwen-Image-2.1"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def get_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return "cpu", torch.float32


def build_pipeline():
    device, dtype = get_device_and_dtype()
    print(f"Loading {MODEL_ID} from local cache on {device} with dtype={dtype}...")
    pipe = QwenImage21Pipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipe = pipe.to(device)
    return pipe, device


def write_generation_log(out_path: Path, prompt: str, elapsed_seconds: float, mode: str):
    """Write metadata for the generated image beside the image file."""
    log_path = out_path.with_suffix(".log")
    log_path.write_text(
        f"Mode: {mode}\n"
        f"Model: {MODEL_ID}\n"
        f"Prompt: {prompt}\n"
        f"Generation time: {elapsed_seconds:.3f} seconds\n",
        encoding="utf-8",
    )
    print(f"Generation log saved to: {log_path.resolve()}")


def run_text_to_image(pipe, device):
    #prompt = 'A fat yellow cat holding a mobile phone filming a cute girl wearing a red dress and a red hat and dancing in a colorful garden. The image is in 4K resolution, ultra-detailed.'
    prompt = '远处的天宫在飘渺的云中，近景的山崖上一对男女在眺望远处的天宫。整个画面要有仙境的感觉。超高清，超精细，超写实，电影感，光影感。'
    generator = torch.Generator(device=device).manual_seed(42) if device == "cuda" else torch.Generator(device="cpu").manual_seed(42)
    start_time = time.perf_counter()
    image = pipe(
        prompt=prompt,
        width=1024,
        height=1024,
        num_inference_steps=35,
        generator=generator,
    ).images[0]
    elapsed_seconds = time.perf_counter() - start_time

    out_path = OUTPUT_DIR / "output_t2i.png"
    image.save(out_path)
    print(f"Text-to-image saved to: {out_path.resolve()}")
    write_generation_log(out_path, prompt, elapsed_seconds, "text-to-image")


def run_edit(pipe, device, image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    input_image = Image.open(image_path).convert("RGB")
    prompt = "Change the background to a sunset beach while preserving the main subject."
    generator = torch.Generator(device=device).manual_seed(42) if device == "cuda" else torch.Generator(device="cpu").manual_seed(42)
    start_time = time.perf_counter()
    image = pipe(
        prompt=prompt,
        image=input_image,
        num_inference_steps=20,
        generator=generator,
    ).images[0]
    elapsed_seconds = time.perf_counter() - start_time

    out_path = OUTPUT_DIR / "qwen_image_2_1_edit.png"
    image.save(out_path)
    print(f"Image edit saved to: {out_path.resolve()}")
    write_generation_log(out_path, prompt, elapsed_seconds, "image-edit")


def main():
    parser = argparse.ArgumentParser(description="Local-cache smoke test for Qwen/Qwen-Image-2.1")
    parser.add_argument("--mode", choices=["t2i", "edit"], default="t2i", help="Generation mode to test")
    parser.add_argument("--image", default="", help="Input image path for --mode edit")
    args = parser.parse_args()

    pipe, device = build_pipeline()

    if args.mode == "t2i":
        run_text_to_image(pipe, device)
    else:
        if not args.image:
            raise ValueError("Please provide --image for edit mode, e.g. --image data/ll.jpg")
        run_edit(pipe, device, args.image)


if __name__ == "__main__":
    main()
