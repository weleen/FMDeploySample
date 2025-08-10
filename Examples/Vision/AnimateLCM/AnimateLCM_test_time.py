import torch
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

from diffusers.utils.import_utils import is_xformers_available

import torch._dynamo
torch._dynamo.reset()

# baseline
device = "cuda"
dtype = torch.float16
adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=dtype).to(device)
pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=dtype).to(device)

# if is_xformers_available():
#     print("enable xformers memory efficient attention")
#     pipe.unet.enable_xformers_memory_efficient_attention()
# else:
#     print("install xformers to enable memory efficient attention")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
pipe.set_adapters(["lcm-lora"], [0.8])

# # Combine attention projection matrices.
# pipe.fuse_qkv_projections()

# # Change the memory layout.
# pipe.unet.to(memory_format=torch.channels_last)
# pipe.vae.to(memory_format=torch.channels_last)

# # compile
# pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
# pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

# pipe.enable_vae_slicing()
# pipe.enable_model_cpu_offload()

import os
UCF_yaml = '../assets/UCF101_prompts.yaml'
import yaml

with open(UCF_yaml, 'r') as stream:
    try:
        UCF_prompts = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

print(UCF_prompts)
prompts  =list(UCF_prompts.values())
# prompts = prompts[:10]
os.makedirs("UCF101-2steps-seed200-guidance1", exist_ok=True)

for prompt in prompts:
    output = pipe(prompt=prompt,
                  negative_prompt="semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
                  num_frames=16,
                  guidance_scale=1.0,
                  num_inference_steps=1,
                  generator=torch.Generator("cpu").manual_seed(200),
                #   return_dict=False
                )
    # export_to_gif(output.frames[0], "UCF101-2steps-seed200-guidance1/AnimateLCM_{}.gif".format(prompt[:30].replace(" ", "_")))


# # bf16
# device = "cuda"
# dtype = torch.bfloat16
# adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=dtype).to(device)
# pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=dtype).to(device)

# # if is_xformers_available():
# #     print("enable xformers memory efficient attention")
# #     pipe.unet.enable_xformers_memory_efficient_attention()
# # else:
# #     print("install xformers to enable memory efficient attention")

# pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

# pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
# pipe.set_adapters(["lcm-lora"], [0.8])

# # # Combine attention projection matrices.
# # pipe.fuse_qkv_projections()

# # # Change the memory layout.
# # pipe.unet.to(memory_format=torch.channels_last)
# # pipe.vae.to(memory_format=torch.channels_last)

# # # compile
# # pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
# # pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

# # pipe.enable_vae_slicing()
# # pipe.enable_model_cpu_offload()

# os.makedirs("UCF101-debug/baseline-bf16", exist_ok=True)

# for prompt in prompts:
#     output = pipe(prompt=prompt,
#                   negative_prompt="semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
#                   num_frames=16,
#                   guidance_scale=0.0,
#                   num_inference_steps=1,
#                   generator=torch.Generator("cpu").manual_seed(0),
#                 #   return_dict=False
#                 )
#     export_to_gif(output.frames[0], "UCF101-debug/baseline-bf16/AnimateLCM_{}.gif".format(prompt[:30].replace(" ", "_")))

# # bf16+xformers
# device = "cuda"
# dtype = torch.bfloat16
# adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=dtype).to(device)
# pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=dtype).to(device)

# if is_xformers_available():
#     print("enable xformers memory efficient attention")
#     pipe.unet.enable_xformers_memory_efficient_attention()
# else:
#     print("install xformers to enable memory efficient attention")

# pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

# pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
# pipe.set_adapters(["lcm-lora"], [0.8])

# # # Combine attention projection matrices.
# # pipe.fuse_qkv_projections()

# # # Change the memory layout.
# # pipe.unet.to(memory_format=torch.channels_last)
# # pipe.vae.to(memory_format=torch.channels_last)

# # # compile
# # pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
# # pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

# # pipe.enable_vae_slicing()
# # pipe.enable_model_cpu_offload()

# os.makedirs("UCF101-debug/baseline-bf16-xformers", exist_ok=True)

# for prompt in prompts:
#     output = pipe(prompt=prompt,
#                   negative_prompt="semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
#                   num_frames=16,
#                   guidance_scale=0.0,
#                   num_inference_steps=1,
#                   generator=torch.Generator("cpu").manual_seed(0),
#                 #   return_dict=False
#                 )
#     export_to_gif(output.frames[0], "UCF101-debug/baseline-bf16-xformers/AnimateLCM_{}.gif".format(prompt[:30].replace(" ", "_")))

# # bf16+xformers+qkv_fuse+memory_last+compile
# device = "cuda"
# dtype = torch.bfloat16
# adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=dtype).to(device)
# pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=dtype).to(device)

# if is_xformers_available():
#     print("enable xformers memory efficient attention")
#     pipe.unet.enable_xformers_memory_efficient_attention()
# else:
#     print("install xformers to enable memory efficient attention")

# pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

# pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
# pipe.set_adapters(["lcm-lora"], [0.8])

# # Combine attention projection matrices.
# pipe.fuse_qkv_projections()

# # Change the memory layout.
# pipe.unet.to(memory_format=torch.channels_last)
# pipe.vae.to(memory_format=torch.channels_last)

# # compile
# pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
# pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

# # pipe.enable_vae_slicing()
# # pipe.enable_model_cpu_offload()

# os.makedirs("UCF101-debug/baseline-bf16-xformers-compile_etc", exist_ok=True)

# for prompt in prompts:
#     output = pipe(prompt=prompt,
#                   negative_prompt="semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
#                   num_frames=16,
#                   guidance_scale=0.0,
#                   num_inference_steps=1,
#                   generator=torch.Generator("cpu").manual_seed(0),
#                 #   return_dict=False
#                 )
#     export_to_gif(output.frames[0], "UCF101-debug/baseline-bf16-xformers-compile_etc/AnimateLCM_{}.gif".format(prompt[:30].replace(" ", "_")))