import argparse
import os
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert
import random

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict

import supervision as sv

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

# control net
# https://github.com/haofanwang/ControlNet-for-Diffusers

# assume you already know the absolute path of installed diffusers
import diffusers
# PATH = os.path.dirname(diffusers.__file__)
# (/home/pywu_server/anaconda3/envs/gsam/lib/python3.9/site-packages/diffusers/pipelines/stable_diffusion/__init__.py)
#   cp ./pipeline_stable_diffusion_controlnet_inpaint.py PATH/pipelines/stable_diffusion
# Then, import this new added pipeline in corresponding files
#   in PATH/__init__.py, line 131: add "StableDiffusionControlNetInpaintPipeline,"
#   in PATH/pipelines/__init__.py, line 65: add "StableDiffusionControlNetInpaintPipeline,"
#   in PATH/pipelines/stable_diffusion/__init__.py, line 50: add "from .pipeline_stable_diffusion_controlnet_inpaint import StableDiffusionControlNetInpaintPipeline"
# See https://github.com/haofanwang/ControlNet-for-Diffusers/tree/main for detail.
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionControlNetInpaintPipeline
from cldm.model import create_model, load_state_dict

# Directly copy from SAM examples: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/onnx_model_example.ipynb
def show_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model  


if __name__ == "__main__":
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    sam_checkpoint = './models/sam_vit_h_4b8939.pth'
    device = "cuda"
    TARGET_PROMPT = "woman"
    BOX_TRESHOLD = 0.3 
    TEXT_TRESHOLD = 0.25
    INPAINT_PROMPT = "A asian, rich, old woman."
    NEGATIVE_PROMPT = "lowres, bad anatomy, worst quality, low quality"

    local_image_path = './demo/Distracted_Boyfriend.png'
    # 
    output_dir = "./demo_output"
    os.makedirs(output_dir, exist_ok=True)

    # Load Dino
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

    # Load SAM
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    # Load stable diffusion inpainting models
    pipe_control = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "lllyasviel/sd-controlnet-seg", # https://huggingface.co/lllyasviel/control_v11p_sd15_seg
        torch_dtype=torch.float16,
    )
    #pipe_control = create_model('./models/cldm_v15.yaml').cpu()
    #pipe_control.load_state_dict(load_state_dict('./models/control_sd15_seg.pth', location='cuda'))

    pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", # https://huggingface.co/runwayml/stable-diffusion-inpainting
        revision="fp16",
        torch_dtype=torch.float16,
    )

    # yes, we can directly replace the UNet
    pipe_control.unet = pipe_inpaint.unet
    pipe_control.unet.in_channels = 4
    pipe_control = pipe_control.to(device)

    # Load demo image
    image_source, image = load_image(local_image_path)

    # Step 1: Run Grounding DINO for detection
    boxes, logits, phrases = predict(
        model=groundingdino_model, 
        image=image, 
        caption=TARGET_PROMPT, 
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB
    plt.imsave(os.path.join(output_dir,"origin.png"), image_source)
    plt.imsave(os.path.join(output_dir,"dino.png"), annotated_frame)

    # Run segmentation anything model
    sam_predictor.set_image(image_source)
    # box: normalized box xywh -> unnormalized xyxy
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    # Step 2: SAM
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
    masks, _, _ = sam_predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes, # box prompt from dino
        multimask_output = False, # only one mask is returned.
    )
    # annotated_frame_with_mask = show_mask(masks[0][0], annotated_frame)
    for i, mask in enumerate(masks):
        annotated_frame_with_mask = show_mask(mask[0].detach().cpu(), annotated_frame)
        plt.imsave(os.path.join(output_dir,f"sam_{i}.png"), annotated_frame_with_mask)

    # Get mask
    image_mask = masks[0][0].detach().cpu()
    # Convert True/False into PIL RGB
    image_mask_pil = Image.fromarray((image_mask.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")
    # Convert PIL to cv2
    image_mask_cv2 = cv2.cvtColor(np.array(image_mask_pil), cv2.COLOR_RGB2BGR)
    # Step 3: Run Control net for Image Inpainting
    result = pipe_control(prompt=INPAINT_PROMPT, 
                         negative_prompt=NEGATIVE_PROMPT,
                         controlnet_hint=image_mask, # segmentation result 
                         image=image_source, # segmentation result 
                         mask_image=image_mask, # mask result (telling model where to inpaint)
                         num_inference_steps=100).images[0]
    # Un-resize 
    result = Image.fromarray(result)
    image_source_pil = Image.fromarray(image_source)
    result = result.resize((image_source_pil.size[0], image_source_pil.size[1]))
    result.save(os.path.join(output_dir,"diffusion.png"))

    #plt.imsave(os.path.join(output_dir,"example.png"), image) 