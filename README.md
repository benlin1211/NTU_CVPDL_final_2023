# Grounded sam 
https://github.com/datarootsio/workshop-image-segmentation-style-transfer/blob/main/notebooks/tutorial.ipynb

    conda create -n gsam python=3.10 -y 
    conda activate gsam

# export CUDA_HOME=/usr/local/cuda-11.1 

    export PATH=/usr/local/cuda/bin:$PATH
    export CUDA_HOME=/usr/local/cuda
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Choose your cuda version, mine is 11.3

    conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch 
    pip install -r requirements.txt
    pip install opencv-contrib-python==4.7.0.72

<!-- pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
pip install diffusers transformers accelerate scipy safetensors -->

## Download pre-trained model

stable-diffusion-inpainting: https://huggingface.co/runwayml/stable-diffusion-inpainting

SAM: https://huggingface.co/spaces/abhishek/StableSAM/blob/main/sam_vit_h_4b8939.pth

    gdown 1LcxjMT_vjTMSee75mRARWr539b20h2-w -O ./models/sam_vit_h_4b8939.pth 

ControlNet (segmentation): https://huggingface.co/lllyasviel/ControlNet

    gdown 1FYnsRuRjaxMjfxPnYI5pqVKccwFZthy5 -O ./models/control_sd15_seg.pth

## Control nat inpaint:
assume you already know the absolute path of installed diffusers
    
    import diffusers
    PATH = os.path.dirname(diffusers.__file__)

(/home/pywu_server/anaconda3/envs/gsam/lib/python3.9/site-packages/diffusers/pipelines/stable_diffusion/__init__.py)
    
    cd grounded_sam
    cp ./pipeline_stable_diffusion_controlnet_inpaint.py PATH/pipelines/stable_diffusion

Then, import this new added pipeline in corresponding files
- in PATH/__init__.py, line 122: add "StableDiffusionControlNetInpaintPipeline,"
- in PATH/pipelines/__init__.py, line 53: add "StableDiffusionControlNetInpaintPipeline,"
- in PATH/pipelines/stable_diffusion/__init__.py, line 50: add "from .pipeline_stable_diffusion_controlnet_inpaint import StableDiffusionControlNetInpaintPipeline"
See https://github.com/haofanwang/ControlNet-for-Diffusers/tree/main for more detail.
