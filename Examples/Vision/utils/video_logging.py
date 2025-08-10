import torchvision.transforms.functional as TVF
from typing import List
from torch import Tensor
from multiprocessing.pool import ThreadPool
import numpy as np
import os
from PIL import Image
import cv2
from tqdm import tqdm


# save video frames as frames, adapt from https://github.com/universome/stylegan-v/blob/3fecd69c602e1cda204357201461c0fb0a634909/src/training/logging.py,
def save_video_frames_as_mp4(frames: List[Tensor], fps: int, save_path: os.PathLike, verbose: bool=False):
    # Load data
    frame_h, frame_w = frames[0].shape[1:]
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(save_path, fourcc, fps, (frame_w, frame_h))
    frames = tqdm(frames, desc='Saving videos') if verbose else frames
    for frame in frames:
        assert frame.shape[0] == 3, "RGBA/grayscale images are not supported"
        frame = np.array(TVF.to_pil_image(frame))
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Uncomment this line to release the memory.
    # It didn't work for me on centos and complained about installing additional libraries (which requires root access)
    # cv2.destroyAllWindows()
    video.release()


def save_video_frames_as_frames(frames: List[Tensor], save_dir: os.PathLike, time_offset: int=0, format='jpg'):
    os.makedirs(save_dir, exist_ok=True)

    for i, frame in enumerate(frames):
        if format == 'jpg': # stylegan-v uses q=95
            save_path = os.path.join(save_dir, f'{i + time_offset:06d}.jpg')
            TVF.to_pil_image(frame).save(save_path, q=95)
        elif format == 'png':
            save_path = os.path.join(save_dir, f'{i + time_offset:06d}.png')
            TVF.to_pil_image(frame).save(save_path, compress_level=0)
        else:
            raise ValueError(f'Unsupported format: {format}')


def save_video_frames_as_frames_parallel(frames: List[np.ndarray], save_dir: os.PathLike, time_offset: int=0, num_processes: int=1):
    assert num_processes > 1, "Use `save_video_frames_as_frames` if you do not plan to use num_processes > 1."
    os.makedirs(save_dir, exist_ok=True)
    # We are fine with the ThreadPool instead of Pool since most of the work is I/O
    pool = ThreadPool(processes=num_processes)
    save_paths = [os.path.join(save_dir, f'{i + time_offset:06d}.jpg') for i in range(len(frames))]
    pool.map(save_jpg_mp_proxy, [(f, p) for f, p in zip(frames, save_paths)])


def save_jpg_mp_proxy(args):
    return save_jpg(*args)


def save_jpg(x: np.ndarray, save_path: os.PathLike):
    Image.fromarray(x).save(save_path, q=95)
