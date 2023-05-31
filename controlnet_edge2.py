# https://huggingface.co/blog/controlnet
from diffusers.utils import load_image
import cv2
from PIL import Image
import numpy as np
import os
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
import torch

# download an image
image = load_image(
     "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
)
image = np.array(image)
mask_image = load_image(
     "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
)
mask_image = np.array(mask_image)
print(mask_image.shape)

# get canny image
canny_image = cv2.Canny(image, 100, 200)
canny_image = canny_image[:, :, None]
canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
print(mask_image.shape)
image=Image.fromarray(image)
mask_image=Image.fromarray(mask_image)
canny_image = Image.fromarray(canny_image)



# controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
# pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
#      "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16
# )
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.to('cuda')

# text_prompt="a red panda sitting on a bench"
# # generate image
# generator = torch.manual_seed(0)
# result = pipe(
#     text_prompt,
#     num_inference_steps=20,
#     generator=generator,
#     image=image,
#     control_hint=canny_image,
#     controlnet_conditioning_scale = 0.5,
#     mask_image=mask_image
# ).images[0]


output_dir = "./demo_output"
result.save(os.path.join(output_dir,"inpaint_seg2.png"))

