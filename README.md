# Grounded sam 
https://github.com/datarootsio/workshop-image-segmentation-style-transfer/blob/main/notebooks/tutorial.ipynb

conda create -n gsam python=3.9.16 -y
conda activate 
# export CUDA_HOME=/usr/local/cuda-11.1 
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# Choose your cuda version, mine is 11.3
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch 
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
pip install diffusers transformers accelerate scipy safetensors