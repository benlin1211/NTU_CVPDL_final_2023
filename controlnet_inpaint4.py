import torch
import cv2
import numpy as np
import urllib
from PIL import Image
import os
from diffusers.utils import load_image

device = "cuda"
output_dir = "./demo_output"
image = load_image('./demo/dog.png')
mask = load_image('./demo/dog_mask.png')
control_image = load_image('./demo/dog_control.png')

from diffusers import StableDiffusionInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from pipeline_stable_diffusion_controlnet_inpaint import *

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
     "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16
 )

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

image.save(os.path.join(output_dir,"inpaint_seg.jpg"))