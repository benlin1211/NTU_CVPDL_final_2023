# https://github.com/mikonvergence/ControlNetInpaint
import torch
from diffusers.utils import load_image
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
import os

# load control net and stable diffusion v1-5
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe_control = StableDiffusionControlNetInpaintPipeline.from_pretrained(
     "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16
 )

# speed up diffusion process with faster scheduler and memory optimization
pipe_control.scheduler = UniPCMultistepScheduler.from_config(pipe_control.scheduler.config)
# remove following line if xformers is not installed
# pipe.enable_xformers_memory_efficient_attention()

pipe_control.to('cuda')

output_dir = "./demo_output"

# we also the same example as stable-diffusion-inpainting
image = load_image('./demo/dog.png')
mask = load_image('./demo/dog_mask.png')
# the segmentation result is generated from https://huggingface.co/spaces/hysts/ControlNet
# upload your image, and then download the generated mask to ./demo_output
control_image = load_image('./demo/dog_control.png')

image = pipe_control(
        "a red panda sitting on a bench",
        num_inference_steps=20,
        generator=torch.manual_seed(0),
        image=image,
        control_image=control_image,
        mask_image=mask,
        ).images[0]

image.save(os.path.join(output_dir,"inpaint_seg.jpg"))