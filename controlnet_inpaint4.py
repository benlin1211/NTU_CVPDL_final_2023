#!/usr/bin/env python3
# https://github.com/mikonvergence/ControlNetInpaint
# git clone https://github.com/mikonvergence/ControlNetInpaint

from diffusers import ControlNetModel, UniPCMultistepScheduler
from ControlNetInpaint.src.pipeline_stable_diffusion_controlnet_inpaint import *
# from ControlNetInpaint.src.pipeline_stable_diffusion_controlnet_inpaint import *
from diffusers.utils import load_image
import torch
import cv2
import numpy as np
import urllib
from PIL import Image
import os
import diffusers
print(diffusers.__version__)

device = "cuda"
output_dir = "./demo_output"
# image = load_image('./demo/dog.png')
# mask = load_image('./demo/dog_mask.png')
# control_image = load_image('./demo/dog_control.png')

# 
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_seg", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)


image = load_image('./demo/dog.png')
mask = load_image('./demo/dog_mask.png')
#print( np.array(mask).shape)
control_image = load_image('./demo/dog_control.png')
#print( np.array(control_image).shape)
print(image.size, mask.size, control_image.size)
generator = torch.manual_seed(0)
image = pipe(
    "a red panda sitting on a bench",
    num_inference_steps=100,
    generator=generator,
    image=image,
    control_image=control_image,
    controlnet_conditioning_scale = 0.5,
    mask_image=mask,
).images[0]

image.save(os.path.join(output_dir,"inpaint_seg4.png"))