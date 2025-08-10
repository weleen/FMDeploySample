#!/usr/bin/env python
# coding: utf-8

# In[]:

import torch
from pathlib import Path
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from diffusers.models.attention_processor import AttnProcessor
from safetensors import safe_open
import gc
import requests
import os
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('..')
from utils.video_logging import save_video_frames_as_frames
import argparse

parser = argparse.ArgumentParser("cdfvd evaluation")
parser.add_argument('--model', default='svd-xt', choices=['svd-xt', 'svd-xt-1-1'])
parser.add_argument('--ucf_frame_path', default='/mnt/dongxu-fs1/data-hdd/yiming/dataset/UCF101/ucf-101_first_frames_png')
parser.add_argument('--num_inference_steps', default=4, type=int, choices=[1, 2, 4, 8], help='Number inference steps.')
parser.add_argument('--resize', action='store_true', help='Resize to (240, 320).')
parser.add_argument('--no_compressed', action='store_true', help='store to no compressed png, or compressed 95% jpg')
parser.add_argument('--output_dir', default='./')
args = parser.parse_args()

device = 'cuda'
weight_dtype = torch.float16

# download scheduler file
lcm_scheduler_url = "https://huggingface.co/spaces/wangfuyun/AnimateLCM-SVD/raw/main/lcm_scheduler.py"
if not os.path.exists('lcm_scheduler.py'):
    r = requests.get(lcm_scheduler_url)
    with open("lcm_scheduler.py", "w") as f:
        f.write(r.text)

from lcm_scheduler import AnimateLCMSVDStochasticIterativeScheduler

model = args.model
if model == 'svd-xt':
    pretrained_model_path = '/home/yiming/pretrained_models/stable-video-diffusion-img2vid-xt'
    online_model_path = 'stabilityai/stable-video-diffusion-img2vid-xt'
    model_path = '/home/yiming/pretrained_models/AnimateLCM/AnimateLCM_SVD/AnimateLCM-SVD-xt.safetensors'
elif model == 'svd-xt-1-1':
    pretrained_model_path = '/home/yiming/pretrained_models/stable-video-diffusion-img2vid-xt-1-1'
    online_model_path = 'stabilityai/stable-video-diffusion-img2vid-xt-1-1'
    model_path = '/home/yiming/pretrained_models/AnimateLCM/AnimateLCM_SVD/AnimateLCM-SVD-xt-1.1.safetensors'
else:
    raise ValueError(f"Model {model} not found")

noise_scheduler = AnimateLCMSVDStochasticIterativeScheduler(
    num_train_timesteps=40,
    sigma_min=0.002,
    sigma_max=700.0,
    sigma_data=1.0,
    s_noise=1.0,
    rho=7,
    clip_denoised=False,
)

if os.path.exists(pretrained_model_path):
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        pretrained_model_path,
        variant="fp16",
        scheduler=noise_scheduler,
    ).to(device)
else:
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        online_model_path,
        variant="fp16",
        scheduler=noise_scheduler,
    ).to(device)

# Load finetuned unet
state_dict = {}
LCM_LORA_PATH = Path(model_path)
with safe_open(LCM_LORA_PATH, framework="pt", device="cpu") as f:
    for key in f.keys():
        state_dict[key] = f.get_tensor(key)
missing, unexpected = pipe.unet.load_state_dict(state_dict, strict=True)
del state_dict

pipe.unet.eval()
pipe.vae.eval()
pipe.image_encoder.eval()

# # Load the conditioning image
# img_dir = '/home/yiming/project/MyProjects/VisionGen/assets/images'
# img_names = os.listdir(img_dir)
# image_files = [os.path.join(img_dir, img_name) for img_name in img_names]

# Get a list of all image files in the ucf_frames directory
image_files = []
for root, dirs, files in os.walk(args.ucf_frame_path):
    for file in files:
        if file.endswith('.png'):  # Assuming the frames are saved as .png
            image_files.append(os.path.join(root, file))
image_files = sorted(image_files)

ucf_generated_svd = os.path.join(args.output_dir, f'ucf_generated_{"95compressed_jpg" if not args.no_compressed else "no_compressed_png"}_{"animatelcmsvdxt" if args.model == "svd-xt" else "animatelcmsvdxt11"}_{args.num_inference_steps}steps_{"320x240" if args.resize else "1024x576"}')

# Create the output directory if it doesn't exist
os.makedirs(ucf_generated_svd, exist_ok=True)

for img_name in tqdm(image_files):
    out_dir = os.path.join(ucf_generated_svd, f"{os.path.basename(img_name)[:-4]}")
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) == 25:
        continue

    image = load_image(img_name)
    image = image.resize((1024, 576))
    with torch.no_grad(), torch.autocast(device, dtype=weight_dtype):
        frames = pipe(
            image,
            num_inference_steps=args.num_inference_steps,
            motion_bucket_id=127,
            num_frames=25,
            height=576,
            width=1024,
            generator=torch.manual_seed(12342)
        ).frames[0]

        if args.resize:
            resized_frames = [frame.resize((320, 240)) for frame in frames]
        else:
            resized_frames = frames
        resized_frames = [torch.tensor(np.array(frame)).permute(2, 0, 1) for frame in resized_frames]

        save_video_frames_as_frames(
            resized_frames,
            out_dir,
            format='png' if args.no_compressed else 'jpg'
        )
