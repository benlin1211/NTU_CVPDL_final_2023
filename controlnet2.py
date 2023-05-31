# Ref: https://github.com/lllyasviel/ControlNet-v1-1-nightly/blob/main/gradio_seg.py
# from share import *

import cv2
import einops
# import gradio as gr
import numpy as np
import torch
import random
import os
from PIL import Image
import matplotlib.pyplot as plt

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.uniformer import UniformerDetector
from diffusers.utils import load_image
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler


apply_uniformer = UniformerDetector() # for deteting semantic segmentations.
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe_control = StableDiffusionControlNetInpaintPipeline.from_pretrained(
     "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16
 )

# speed up diffusion process with faster scheduler and memory optimization
pipe_control.scheduler = UniPCMultistepScheduler.from_config(pipe_control.scheduler.config)
# remove following line if xformers is not installed
# pipe.enable_xformers_memory_efficient_attention()
model = pipe_control.to('cuda')

output_dir = "./demo_output"

def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map = apply_uniformer(resize_image(input_image, detect_resolution))
        plt.imsave(os.path.join(output_dir, "Uniformer_Detector.png"), detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
        # control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        # control = torch.stack([control for _ in range(num_samples)], dim=0)
        # control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        image = pipe_control(prompt=prompt, 
                    negative_prompt=n_prompt,
                    controlnet_hint=detected_map, 
                    image=image,
                    mask_image=mask,
                    num_inference_steps=100).images[0]
    return [detected_map] + results


if __name__ == "__main__":
    local_image_path = './demo/Distracted_Boyfriend.png'

    input_image = np.array(Image.open(local_image_path)) # gr.Image(source='upload', type="numpy")
    prompt = "A asian, rich, old woman." # 'Chimpanzee style' # gr.Textbox(label="Prompt")
    num_samples = 1  #gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
    image_resolution = 512 # gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
    strength = 1 # gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
    guess_mode = False # gr.Checkbox(label='Guess Mode', value=False)
    detect_resolution = 512 # gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
    ddim_steps = 20 # gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
    scale = 9.0 # gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
    seed = 1211 # gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
    eta = 0.0 # gr.Number(label="eta (DDIM)", value=0.0)
    a_prompt = 'best quality, extremely detailed' # gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
    n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
    # gr.Textbox(label="Negative Prompt",  value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')

    detected_maps, result = process(input_image, prompt, a_prompt, n_prompt, 
                     num_samples, image_resolution, detect_resolution, 
                     ddim_steps, guess_mode, 
                     strength, scale, seed, eta)
    # print(len(detected_maps))
    print(result.shape)
    plt.imsave(os.path.join(output_dir, f"controlnet_{prompt}_test.png"), result)