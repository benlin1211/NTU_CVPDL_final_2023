import torch
from diffusers.utils import load_image
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionControlNetInpaintPipeline, ControlNetModel
import os
# https://github.com/huggingface/diffusers/pull/2407
# https://github.com/mikonvergence/ControlNetInpaint

output_dir = "./demo_output"
ROOT = os.getcwd()
pretrain_control_seg = os.path.join(ROOT, "models/control_sd15_seg.pth")

# we have downloaded models locally, you can also load from huggingface
# control_sd15_seg is converted from control_sd15_seg.safetensors using instructions above
import diffusers
print(os.path.dirname(diffusers.__file__))
pipe_control = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    # pretrain_control_seg,
    #"lllyasviel/control_v11p_sd15_seg",
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16, 
    local_files_only=True
).to('cuda')
# pipe_control = ControlNetModel.from_pretrained(
#     # pretrain_control_seg,
#     #"lllyasviel/control_v11p_sd15_seg",
#     "lllyasviel/sd-controlnet-canny",
#     torch_dtype=torch.float16, 
#     local_files_only=True
# ).to('cuda')

pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", # https://huggingface.co/runwayml/stable-diffusion-inpainting
    revision="fp16",
    torch_dtype=torch.float16,
)
# yes, we can directly replace the UNet
pipe_control.unet = pipe_inpaint.unet
pipe_control.unet.in_channels = 4

# we also the same example as stable-diffusion-inpainting
image = load_image("https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png")
mask = load_image("https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png")

# the segmentation result is generated from https://huggingface.co/spaces/hysts/ControlNet
control_image = load_image('tmptvkkr0tg.png')

image = pipe_control(prompt="Face of a yellow cat, high resolution, sitting on a park bench", 
                     negative_prompt="lowres, bad anatomy, worst quality, low quality",
                     controlnet_hint=control_image, 
                     image=image,
                     mask_image=mask,
                     num_inference_steps=100).images[0]

image.save(os.path.join(output_dir,"inpaint_seg.jpg"))