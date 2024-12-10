"""
This script demonstrates how to generate multi-camera synchronized videos using SynCamMaster + CogVideoX.
"""

import argparse
from typing import Literal
import torch
import sys
import os
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..'))

from diffusers.models.normalization import CogVideoXLayerNormZero
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0
from diffusers import (
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
)
from syncammaster.pipeline_syncammaster import CogVideoXPipeline
from syncammaster.transformer_3d import CogVideoXTransformer3DModel
from diffusers.utils import export_to_video
from torch.utils.data import DataLoader
from torch import nn
from einops import rearrange, repeat
import subprocess
import json
import csv
from torch.utils.data import DataLoader, Dataset


class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)

class ValData(Dataset):
    def __init__(
            self,
            view_num: int,
            prompt_file: str = None,
            camera_type: str = "az",
    ):  
        self.view_num = view_num
        self.prompt_file = prompt_file
        self.camera_type = camera_type
        self.prompts = []
        with open(self.prompt_file, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                self.prompts.append(row['caption'])
        self.length = len(self.prompts)

    def get_relative_pose(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]

        cam_to_origin = 0
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -cam_to_origin],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

    def parse_matrix(self, matrix_str):
        rows = matrix_str.strip().split('] [')
        matrix = []
        for row in rows:
            row = row.replace('[', '').replace(']', '')
            matrix.append(list(map(float, row.split())))
        return np.array(matrix)

    def get_batch(self, idx):
        return_dict = {}
        train_pose_scale = 200
        t_scale = 1  
        cameras = []
        if self.camera_type == 'az':
            dis = 4
            el = 15
            with open(f"cameras/Hemi36_{dis}m_{el}/Hemi36_{dis}m_{el}.json", 'r') as file:
                cam_data = json.load(file)
            for view_idx in range(self.view_num):
                view_idx *= 2  # 10° * n
                key = f"C_{view_idx + 1:02d}_18mm"
                data = self.parse_matrix(cam_data[key])
                cameras.append(data)
            cameras.reverse()
        elif self.camera_type == 'el':
            dis = 5
            for view_idx in range(self.view_num):
                el = view_idx * 15
                with open(f"cameras/Hemi36_{dis}m_{el}/Hemi36_{dis}m_{el}.json", 'r') as file:
                    cam_data = json.load(file)
                key = f"C_01_35mm"
                data = self.parse_matrix(cam_data[key])
                cameras.append(data)
        elif self.camera_type == 'dis':
            el = 15
            for view_idx in range(self.view_num):
                dis = 4 + view_idx
                with open(f"cameras/Hemi36_{dis}m_{el}/Hemi36_{dis}m_{el}.json", 'r') as file:
                    cam_data = json.load(file)
                key = f"C_01_35mm"
                data = self.parse_matrix(cam_data[key])
                cameras.append(data)

        cameras = np.stack(cameras)
        cameras = np.transpose(cameras, (0,2,1))
        c2ws = []
        for i, data in enumerate(cameras):
            if data.shape[0] == 3:
                data = np.vstack((data, np.array([[0, 0, 0, 1]])))
            data = data[:, [1,2,0,3]]
            data[:3,1] *= -1.
            c2ws.append(data)

        for c2w in c2ws:
            c2w[:3, 3] = (c2w[:3, 3] / train_pose_scale) * t_scale

        cam_params = [Camera(cam_param) for cam_param in c2ws]
        c2w_poses = self.get_relative_pose(cam_params)
        c2w = torch.as_tensor(c2w_poses)[None]
        pose_embedding = c2w[:,:,:3,:][0]
        return_dict.update(
            {
                "prompts": [self.prompts[idx%len(self.prompts)]],
                "poses": pose_embedding,
            }
        )
        return return_dict

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
            return_dict = self.get_batch(idx % self.length)
            return return_dict

    def collate_fn(self, examples):
        examples = list(filter(lambda x: "failed" not in x, examples))  # filter out all the Nones
        return_dict = {}
        prompts = [example["prompts"][0] for example in examples]
        poses = torch.stack([example["poses"] for example in examples])
        return_dict.update(
            {
            'prompts': prompts,
            'poses' : poses,
            }
        )
        return return_dict


def generate_video(
    prompt: str,
    cogvideo_model_path: str,
    view_num: int,
    ckpt_path: str,
    prompt_file: str,
    camera_type: str = "az",
    save_dir: str = "output",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
):

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    pipe = CogVideoXPipeline.from_pretrained(cogvideo_model_path, torch_dtype=dtype)
    val_dataset = ValData(view_num=view_num, prompt_file=prompt_file, camera_type=camera_type)
    val_dataloader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=val_dataset.collate_fn,
    num_workers=1,
    )
    
    pipe.transformer = CogVideoXTransformer3DModel.from_pretrained(
        cogvideo_model_path,
        subfolder="transformer",
        torch_dtype=dtype,
    )

    # 1.1 INSERT MVS Module in SynCamMaster
    for block in pipe.transformer.transformer_blocks:
        dim = block.attn1.query_dim
        qk_norm = True
        block.norm_syncam = CogVideoXLayerNormZero(pipe.transformer.config.time_embed_dim, dim, pipe.transformer.config.norm_elementwise_affine, pipe.transformer.config.norm_eps, bias=True)
        block.attn_syncam = Attention(
            query_dim=dim,
            dim_head=pipe.transformer.config.attention_head_dim,
            heads=pipe.transformer.config.num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=pipe.transformer.config.attention_bias,
            out_bias=True,
            processor=CogVideoXAttnProcessor2_0(),
        )
        block.cam_encoder = nn.Linear(12, dim)
        block.projector = nn.Linear(dim, dim) 

    # 1.2 Load SynCamMaster Pretrained Checkpoint
    state_dict = torch.load(ckpt_path)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[len("module."):]
        else:
            new_key = key

        new_key = new_key.replace("pose_fuse_layer_1", "cam_encoder")
        new_key = new_key.replace("pose_fuse_layer_2", "projector")
        new_state_dict[new_key] = value
    pipe.transformer.load_state_dict(new_state_dict,strict=True)

    pipe.transformer = pipe.transformer.to(dtype)

    # 2. Set Scheduler.
    if "2b" in cogvideo_model_path:
        pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    else:
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # 3. Enable CPU offload for the model.
    # pipe.enable_sequential_cpu_offload()
    # pipe.vae.enable_slicing()
    # pipe.vae.enable_tiling()
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")
    pipe.to("cuda")

    for step, batch in enumerate(val_dataloader):
        print(batch["prompts"])
        prompt = batch["prompts"] * view_num
        pose = rearrange(batch["poses"], "b v m n -> b v (m n)").to(dtype=dtype, device="cuda")
        
        # 4. Generate the video frames based on the prompt and camera poses.
        video_generate = pipe(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=49,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
            pose=pose,
        ).frames

        # 5. Export the generated frames to a video file.
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        video_files = []
        for i, video in enumerate(video_generate):
            video_file = os.path.join(save_dir, f"temp{step}_{i}.mp4")
            export_to_video(video, video_file)
            video_files.append(video_file)
        output_file = os.path.join(save_dir, f"output{step}.mp4")
        ffmpeg_command = [
            "ffmpeg",
            "-y",
            "-loglevel", "quiet"
        ]
        for video_file in video_files:
            ffmpeg_command.extend(["-i", video_file])
        ffmpeg_command.extend([
            "-filter_complex",
            f"hstack=inputs={view_num}",
            output_file
        ])
        subprocess.run(ffmpeg_command)
        for video_file in video_files:
            os.remove(video_file)
        print(f"Concatenated video saved as {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, default="a bot", help="The description of the video to be generated")
    parser.add_argument(
        "--cogvideo_model_path", type=str, default="THUDM/CogVideoX-2b", help="The path of the pre-trained model to be used"
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')"
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--view_num", type=int, default=2)
    parser.add_argument("--ckpt_path", type=str, default="transformer_20000.pth")
    parser.add_argument("--save_dir", type=str, default="output")
    parser.add_argument("--prompt_file", type=str, default="caption/prompts.csv")
    parser.add_argument("--camera_type", type=str, choices=["az", "el", "dis"], default="az")

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    generate_video(
        prompt=args.prompt,
        cogvideo_model_path=args.cogvideo_model_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        seed=args.seed,
        view_num=args.view_num,
        ckpt_path=args.ckpt_path,
        save_dir=args.save_dir,
        prompt_file=args.prompt_file,
        camera_type=args.camera_type,
    )
