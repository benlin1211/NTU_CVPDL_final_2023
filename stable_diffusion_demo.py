from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import requests
from io import BytesIO
import matplotlib.pyplot as plt

def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


if __name__ == "__main__":
    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

    init_image = download_image(img_url).resize((512, 512))
    mask_image = download_image(mask_url).resize((512, 512))
    #plt.imsave(os.path.join(output_dir,"init_image.png"), init_image) 
    init_image.save("init_image_pil.png")
    init_image = np.array(init_image)
    plt.imsave("init_image_np.png", init_image) 


    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
    image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
    image.save("init_image_pil.png")