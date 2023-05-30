import argparse
import os
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

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


# diffusers
import PIL
import torch
from diffusers import StableDiffusionInpaintPipeline

from huggingface_hub import hf_hub_download


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
    sam_checkpoint = './ckpts/sam_vit_h_4b8939.pth'
    device = "cuda"
    TARGET_PROMPT = "background"
    BOX_TRESHOLD = 0.3 
    TEXT_TRESHOLD = 0.25
    INPAINT_PROMPT = "On a bus."

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
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to(device)

    # Load demo image
    image_source, image = load_image(local_image_path)

    # Run Grounding DINO for detection
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
    # SAM
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

    # Run Stable Diffusion for Image Inpainting
    image_mask = masks[0][0].cpu().numpy()

    # resize for inpaint
    image_source_pil = Image.fromarray(image_source)
    image_source_for_inpaint = image_source_pil.resize((512, 512))
    image_mask_pil = Image.fromarray(image_mask)
    image_mask_for_inpaint = image_mask_pil.resize((512, 512))
    #annotated_frame_pil = Image.fromarray(annotated_frame)
    #annotated_frame_with_mask_pil = Image.fromarray(annotated_frame_with_mask)

    # Stable Diffusion. Input and output: PIL.Image
    image_inpainting = pipe(prompt=INPAINT_PROMPT, image=image_source_for_inpaint, mask_image=image_mask_for_inpaint).images[0]
    #image_inpainting = pipe(prompt=INPAINT_PROMPT, image=np.array(image_source_for_inpaint), mask_image=np.array(image_mask_for_inpaint)).images[0]

    # Un-resize 
    image_inpainting = image_inpainting.resize((image_source_pil.size[0], image_source_pil.size[1]))
    image_inpainting.save(os.path.join(output_dir,"diffusion.png"))

    #plt.imsave(os.path.join(output_dir,"example.png"), image) 