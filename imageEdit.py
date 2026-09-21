###################################################
# Test script for QwenImageEditPlusPipeline
#
# This script demonstrates how to use the QwenImageEditPlusPipeline to edit images based on a given prompt. 
# It loads two images, applies the editing pipeline, and saves the output image.
# usage: python imageEdit.py
###################################################
import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline


def main():
    """Load the Qwen image editing pipeline and generate an edited image."""
    # Load the pretrained pipeline
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        local_files_only=True,  # Use local files only to avoid downloading
    )
    print("pipeline loaded")

    # Set the progress bar configuration
    pipeline.set_progress_bar_config(disable=None)

    # Load the input images
    image1 = Image.open("data/Lindsay_Lohan.jpg")
    image2 = Image.open("data/cat.jpg")

    # Prepare output directory
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_image_name = "output_ll.png"
    output_image_path = os.path.join(output_dir, output_image_name)

    # Prepare pipeline inputs
    prompt = "Let the lady hold the cat. Do not change the lady's face!"
    inputs = {
        "image": [image1, image2],
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 40,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
    }

    # Run inference and save output
    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]
        output_image.save(output_image_path)
        print("image saved at", os.path.abspath(output_image_path))


if __name__ == "__main__":
    main()

