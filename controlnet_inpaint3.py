#!/usr/bin/env python3
# https://github.com/huggingface/diffusers/issues/3582
# https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet#diffusers.StableDiffusionControlNetInpaintPipeline
# https://github.com/haofanwang/ControlNet-for-Diffusers/blob/main/README.md

from diffusers import ControlNetModel, DDIMScheduler, DiffusionPipeline
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionControlNetInpaintPipeline
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
    "runwayml/stable-diffusion-inpainting",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16
)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

def decode_image(image_url):
    print("Decode image: " + image_url)
    req = urllib.request.urlopen(image_url)
    arr = np.array(bytearray(req.read()), dtype=np.uint8)
    imageBGR = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGRA2RGB)

    return Image.fromarray(imageRGB)

# img_url = "https://i.ibb.co/0ZK7yL0/img.png"
# img_red_bg_url = "https://i.ibb.co/zHHZTkZ/img-red-bg.png"
# canny_url = "https://i.ibb.co/rp9FYCX/canny.png"
# mask_url = "https://i.ibb.co/FK4DNNK/mask.png"
# img = decode_image(img_url)
# img_red_bg = decode_image(img_red_bg_url)
# canny = decode_image(canny_url)
# mask = decode_image(mask_url)
image = load_image('./demo/dog.png')
mask = load_image('./demo/dog_mask.png')
print( np.array(mask).shape)
control_image = load_image('./demo/dog_control.png')
print( np.array(control_image).shape)
# image = pipe(prompt="a red panda sitting on a bench",
#             negative_prompt="lowres, bad anatomy, worst quality, low quality",
#             controlnet_hint=control_image, 
#             image=image,
#             mask_image=mask,
#             num_inference_steps=100).images[0]

image = pipe("a red panda sitting on a bench",
            negative_prompt="lowres, bad anatomy, worst quality, low quality",
            controlnet_hint=control_image, 
            image=image,
            mask_image=mask,
            controlnet_conditioning_scale=0.5,
            num_inference_steps=100).images[0]

# generator = torch.manual_seed(0)
# image = pipe(
#     "a red panda sitting on a bench",
#     num_inference_steps=100,
#     generator=generator,
#     image=image,
#     control_image=control_image,
#     controlnet_conditioning_scale = 0.5,
#     mask_image=mask,
# ).images[0]

image.save(os.path.join(output_dir,"inpaint_seg3.png"))