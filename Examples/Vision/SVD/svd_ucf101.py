#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_gif, export_to_video, load_image
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser("cdfvd evaluation")
parser.add_argument('--ucf_frame_path', default='/mnt/dongxu-fs1/data-hdd/yiming/dataset/UCF101/ucf-101_first_frames_png')
parser.add_argument('--num_inference_steps', default=4, choices=[1, 2, 4, 8], help='Number inference steps.')
parser.add_argument('--resize', action='store_true', help='Resize to (240, 320).')
parser.add_argument('--save_as_mp4', action='store_true', help='Save as mp4')
parser.add_argument('--use_subset', action='store_true', help='Only inference on a subset of images.')
parser.add_argument('--no_compressed', action='store_true', help='store to no compressed png, or compressed 95% jpg')
parser.add_argument('--max_processes', default=200000, help='Max image processed here.')
args = parser.parse_args()

device='cuda'
weight_dtype = torch.float16
torch.backends.cuda.matmul.allow_tf32 = True

pipeline = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid", variant="fp16"
).to(device, weight_dtype)

# In[]:
# # https://huggingface.co/docs/diffusers/v0.28.2/en/tutorials/fast_diffusion
# if is_xformers_available():
#     print("enable xformers memory efficient attention")
#     pipeline.unet.enable_xformers_memory_efficient_attention()
# else:
#     print("install xformers to enable memory efficient attention")

# pipeline.enable_model_cpu_offload()
# 20-25% speedup
# torch._inductor.config.conv_1x1_as_mm = True
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.epilogue_fusion = False
# torch._inductor.config.coordinate_descent_check_all_directions = True
# pipeline.unet.to(memory_format=torch.channels_last)
# pipeline.vae.to(memory_format=torch.channels_last)
# pipeline.unet = torch.compile(pipeline.unet, mode="max-autotune", fullgraph=True)
# pipeline.vae.decode = torch.compile(pipeline.vae.decode, mode="max-autotune", fullgraph=True)


# In[2]:

import os
from PIL import Image
import sys
sys.path.append('..')
from utils.video_logging import save_video_frames_as_frames, save_video_frames_as_mp4

# 30 sub classes
saved_subset_names = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress', 'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats', 'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth', 'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen', 'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics']

ucf_origin_folder = '/mnt/dongxu-fs1/data-hdd/yiming/dataset/UCF101/UCF-101'

def process_ucf_frames(ucf_frames, num_inference_steps=4, resize=False, save_as_mp4=False, use_subset=False, no_compressed=False, max_process=200000):
    print(f'Arguments: {ucf_frames}, {num_inference_steps}, {resize}, {save_as_mp4}, {use_subset}, {no_compressed}, {max_process}')
    ucf_generated_svd = f'./ucf_generated_{"95compressed_jpg" if not no_compressed else "no_compressed_png"}_svd_{num_inference_steps}steps_{"320x240" if resize else "1024x576"}{"_subset" if use_subset else ""}'

    # Create the output directory if it doesn't exist
    os.makedirs(ucf_generated_svd, exist_ok=True)

    # Get a list of all image files in the ucf_frames directory
    image_files = []
    for root, dirs, files in os.walk(ucf_frames):
        for file in files:
            if file.endswith('.png'):  # Assuming the frames are saved as .png
                image_files.append(os.path.join(root, file))

    if use_subset:
        all_subset_names = sorted(os.listdir(ucf_origin_folder))
        subset_names = all_subset_names[:30] # take 30 subsets from 101 folders
        assert subset_names == saved_subset_names 
        subset_image_files = [f for f in image_files if any(name in f for name in subset_names)]
        print(f"Using {len(subset_image_files)}/{len(image_files)} images")
        image_files = subset_image_files

    num_processed = 0
    # Process each image file
    for image_file in tqdm(image_files):
        img = Image.open(image_file).resize((1024, 576), Image.Resampling.BILINEAR)
        generator = torch.manual_seed(0)
        if save_as_mp4 and os.path.exists(os.path.join(ucf_generated_svd, f"{os.path.basename(image_file)[:-4]}")):
            continue
        elif not save_as_mp4 and os.path.exists(os.path.join(ucf_generated_svd, f"{os.path.basename(image_file)[:-4]}")):
            continue

        print(image_file)

        with torch.no_grad(), torch.autocast(device, dtype=weight_dtype):
            frames = pipeline(img, decode_chunk_size=8, generator=generator, motion_bucket_id=127, fps=7, num_inference_steps=num_inference_steps).frames[0]
            if resize:
                resized_frames = [frame.resize((320, 240)) for frame in frames]
            else:
                resized_frames = frames
            resized_frames = [torch.tensor(np.array(frame)).permute(2, 0, 1) for frame in resized_frames]

            if save_as_mp4:
                save_video_frames_as_mp4(
                    resized_frames,
                    7,
                    os.path.join(ucf_generated_svd, f"{os.path.basename(image_file)[:-4]}.mp4")
                )
            else:
                save_video_frames_as_frames(
                    resized_frames,
                    os.path.join(ucf_generated_svd, f"{os.path.basename(image_file)[:-4]}"),
                    format='png' if no_compressed else 'jpg'
                )
        num_processed += 1
        if num_processed > max_process:
            break

process_ucf_frames(args.ucf_frame_path,
                   num_inference_steps=args.num_inference_steps,
                   resize=args.resize,
                   save_as_mp4=args.save_as_mp4,
                   use_subset=args.use_subset,
                   no_compressed=args.no_compressed,
                   max_process=args.max_processes)