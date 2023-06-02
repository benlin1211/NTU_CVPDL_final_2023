import argparse
import os
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert
import random
import argparse

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model 
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, predict
from groundingdino.util.inference import load_image as load_image_dino

import supervision as sv

# segment anything
from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

# control net
# DO NOT use this: https://github.com/haofanwang/ControlNet-for-Diffusers
# Use this: https://github.com/mikonvergence/ControlNetInpaint
import diffusers
from diffusers import ControlNetModel, UniPCMultistepScheduler
from ControlNetInpaint.src.pipeline_stable_diffusion_controlnet_inpaint import *

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


def convert_anns(anns):
    # Convert mask_generator result into RGB  
    # Ref: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    semantic_mask = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.random.random(3)
        semantic_mask[m] = color_mask
    return semantic_mask


def make_parser():
    parser = argparse.ArgumentParser(description="hw 4-2 train",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("image_path", help="Input image path") 
    parser.add_argument("target_prompt", help="Which object to be inpainted") 
    parser.add_argument("inpaint_prompt", help="What object to inpaint") 
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25) 
    parser.add_argument("--negative_prompt", default="lowres, bad anatomy, worst quality, low quality")
    
    parser.add_argument("--output_dir", default="./demo_output")
    parser.add_argument("--device_dino", default="cpu")
    parser.add_argument("--device_sam", default="cuda:0")
    parser.add_argument("--device_pipe", default="cuda:1")
    args = parser.parse_args()  

    return args


if __name__ == "__main__":
    assert torch.cuda.device_count()==2
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    sam_checkpoint = './models/sam_vit_h_4b8939.pth'
    turn_on_semantic_guidance = True
    args = make_parser()
    TARGET_PROMPT = args.target_prompt
    BOX_TRESHOLD = args.box_threshold
    TEXT_TRESHOLD = args.text_threshold
    INPAINT_PROMPT = args.inpaint_prompt
    NEGATIVE_PROMPT = args.negative_prompt
    local_image_path = args.image_path

    device_dino = args.device_dino
    device_sam = args.device_sam
    device_pipe = args.device_pipe
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load Dino to "cuda:1"
    print(f"Load Dino to {device_dino}")
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device=device_dino)

    # Load SAM to "cuda:0"
    print(f"Load SAM to {device_sam}")
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=device_sam)
    sam_predictor = SamPredictor(sam)
    # Segment anything setting.
    # See https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb 
    mask_generator = SamAutomaticMaskGenerator( 
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    ) 

    # Load stable diffusion controlnet inpainting model to "cuda:1"
    print(f"Load controlnet to {device_pipe}")
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_seg", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=controlnet, 
        torch_dtype=torch.float16,
        safety_checker=None, # Disable NSFW checker for academic use. # https://github.com/huggingface/diffusers/issues/3422
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # Disable NSFW checker for academic use.
    # pipe.safety_checker = lambda images, clip_input: (images, False)
    pipe = pipe.to(device_pipe) #device

    # Load demo image
    image_source, image = load_image_dino(local_image_path)

    # ============== Step 1: Run Grounding DINO for detection ==============
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
    # ========================== Step 2: SAM ==========================
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device=device_sam) # device
    # with torch.no_grad():
    masks, _, _ = sam_predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes, # box prompt from dino
        multimask_output = False, # only one mask is returned.
    )
    if turn_on_semantic_guidance:
        sementic_mask = mask_generator.generate(np.array(image_source))
        sementic_mask = convert_anns(sementic_mask)

    # Save mask
    for i, mask in enumerate(masks):
        annotated_frame_with_mask = show_mask(mask[0].detach().cpu(), annotated_frame)
        plt.imsave(os.path.join(output_dir,f"sam_{i}.png"), annotated_frame_with_mask)
    plt.imsave(os.path.join(output_dir,"sam_all.png"), sementic_mask)

    # Get mask
    image_mask = masks[0][0].detach().cpu()
    # Convert into PIL RGB
    image_source_pil = Image.fromarray(image_source).convert("RGB")
    image_mask_pil = Image.fromarray((image_mask.cpu().numpy() * 255).astype(np.uint8)).convert("RGB")
    sementic_mask_pil = Image.fromarray((sementic_mask * 255).astype(np.uint8)).convert("RGB")
    # dino mask 
    boxes_mask_pil = Image.fromarray((image_mask.cpu().numpy() * 255).astype(np.uint8)).convert("RGB")

    # Resize so that CUDA out of memory won't happen.
    image_source_for_inpaint = image_source_pil.resize((512, 512))
    image_mask_for_inpaint = image_mask_pil.resize((512, 512))
    #image_mask_for_inpaint = boxes_mask_pil.resize((512, 512))
    sementic_mask_for_inpaint = image_mask_pil.resize((512, 512))
    # print(image_source.shape, image_mask.shape, sementic_mask.shape)
    # print(image_source_pil.size, image_mask_pil.size, sementic_mask_pil.size)
    # print(image_source_for_inpaint.size, image_mask_for_inpaint.size, sementic_mask_for_inpaint.size)

    # ============== Step 3: Run Control net for Image Inpainting ==============
    generator = torch.manual_seed(0)
    result = pipe(
        INPAINT_PROMPT,
        num_inference_steps=100,
        generator=generator,
        image=image_source_for_inpaint,
        control_image=sementic_mask_for_inpaint,
        controlnet_conditioning_scale = 0.5,
        mask_image=image_mask_for_inpaint,
        negative_prompt=NEGATIVE_PROMPT,
    ).images[0]

    # Un-resize 
    #result = Image.fromarray(result)
    result = result.resize((image_source_pil.size[0], image_source_pil.size[1]))
    result.save(os.path.join(output_dir,"diffusion_controlnet.png"))