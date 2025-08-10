#!/usr/bin/env python
# coding: utf-8

# In[1]:
import torch
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_gif, export_to_video, load_image
from diffusers.utils.import_utils import is_xformers_available
import packaging

import torch._dynamo
torch._dynamo.reset()


if packaging.version.parse(torch.__version__) >= packaging.version.parse('1.12.0'):
    torch.backends.cuda.matmul.allow_tf32 = True
    
# weight_dtype = torch.bfloat16
weight_dtype = torch.float16

pipeline = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid", variant="fp16"
).to("cuda", dtype=weight_dtype)

# if is_xformers_available():
#     print("enable xformers memory efficient attention")
#     pipeline.unet.enable_xformers_memory_efficient_attention()
# else:
#     print("install xformers to enable memory efficient attention")

# pipeline.enable_model_cpu_offload() # slow but save memory
# 20-25% speedup

# Combine attention projection matrices.
# pipeline.fuse_qkv_projections()

# # Change the memory layout.
# pipeline.unet.to(memory_format=torch.channels_last)
# pipeline.vae.to(memory_format=torch.channels_last)

# # compile
# pipeline.unet = torch.compile(pipeline.unet, mode="max-autotune", fullgraph=True)
# pipeline.vae.decode = torch.compile(pipeline.vae.decode, mode="max-autotune", fullgraph=True)

# torch._inductor.config.conv_1x1_as_mm = True
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.epilogue_fusion = False
# torch._inductor.config.coordinate_descent_check_all_directions = True
# pipeline.unet.to(memory_format=torch.channels_last)
# pipeline.vae.to(memory_format=torch.channels_last)
# pipeline.unet = torch.compile(pipeline.unet, mode="max-autotune", fullgraph=True)
# pipeline.vae.decode = torch.compile(pipeline.vae.decode, mode="max-autotune", fullgraph=True)


# In[2]:

# test 1-step inference speed
img = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
generator = torch.manual_seed(42)
for _ in range(10):
    with torch.no_grad(), torch.autocast("cuda", dtype=weight_dtype):
        frames = pipeline(img, decode_chunk_size=7, generator=generator, motion_bucket_id=127, fps=7, max_guidance_scale=1.0, num_inference_steps=1).frames[0]
# export_to_gif(frames, "generated_1step.gif")
