import os
import subprocess

import cv2
import ffmpeg
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm


def create_video_clip(frames, fps, output_file, video_file_temp='test.mp4'):
    print('frame shape', frames.shape)

    # Create video using OpenCV
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_file_temp, fourcc, fps, (width, height))

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()

    name = output_file.split('.')[0] + '.mp4'

    # Merge video and audio using ffmpeg
    # Adding '-y' to overwrite existing files without asking
    ffmpeg.output(ffmpeg.input(video_file_temp), name, vcodec='copy', acodec='copy', strict='experimental', y=None).run()

    # Remove temporary video file
    os.remove(video_file_temp)
    print(name)

    # Set the codec parameters for Discord compatibility
    ffmpeg_command = [
        'ffmpeg',
        '-y',  # Overwrite output files without asking
        '-i', name,
        '-c:v', 'libx264',  # H.264 video codec
        '-c:a', 'aac',  # AAC audio codec
        '-strict', 'experimental',  # Necessary for the 'aac' codec
        '-threads', '4',  # Number of CPU threads to use for encoding (adjust as needed)
        output_file.split('.')[0]+'_discord.mp4'
    ]

    # Run FFmpeg command
    subprocess.call(ffmpeg_command)


def save_video(generated_video, fps, filename):
    # 3, num_frames, width, height
    # num_frames, width, height, 3
    generated_video = (generated_video.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 3, 0)
    create_video_clip(generated_video, fps, f'{filename}.mp4')

















class VideoPreprocessor(nn.Module):
    def __init__(self):
        super(VideoPreprocessor, self).__init__()

    def forward(self, x):
        # Normalize pixel values to [0, 1]
        x_normalized = x / 255.0
        return x_normalized


videoPreprocessor = VideoPreprocessor()


import numpy as np





def split_video_into_patches(video_tensor, patch_size=16, overlap=4):
    batch_size, num_frames, height, width, channels = video_tensor.shape
    effective_patch_size = patch_size + 2 * overlap  # Include overlap on both sides

    # Calculate padding to ensure that patches fit evenly into the video tensor
    pad_height = (effective_patch_size - (height % patch_size)) % effective_patch_size
    pad_width = (effective_patch_size - (width % patch_size)) % effective_patch_size
    video_tensor_padded = torch.nn.functional.pad(video_tensor, (0, 0, overlap, pad_width + overlap, overlap, pad_height + overlap, 0, 0), mode='constant', value=0)

    # Update dimensions with padding
    padded_height, padded_width = height + pad_height + 2 * overlap, width + pad_width + 2 * overlap
    num_patches_h = (padded_height - effective_patch_size) // (patch_size) + 1
    num_patches_w = (padded_width - effective_patch_size) // (patch_size) + 1

    patches = torch.zeros((batch_size, num_frames * num_patches_h * num_patches_w, effective_patch_size, effective_patch_size, channels))

    patch_idx = 0
    for frame in range(num_frames):
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                start_i = i * patch_size
                start_j = j * patch_size
                patch = video_tensor_padded[:, frame, start_i:start_i + effective_patch_size, start_j:start_j + effective_patch_size, :]
                patches[:, patch_idx, :, :, :] = patch
                patch_idx += 1

    return patches


def split_video_into_patches_mem(video_tensor, patch_size=24, overlap=4):
    batch_size, num_frames, channels, height, width = video_tensor.shape
    # Calculate the size of padding based on the overlap
    pad_size = overlap * 2

    # Pad the tensor on the spatial dimensions
    padded_video = F.pad(video_tensor, (overlap, overlap, overlap, overlap), mode='constant', value=0)

    # Calculate the number of patches in each dimension
    vertical_patches = (height + pad_size - patch_size) // (patch_size - overlap*2) + 1
    horizontal_patches = (width + pad_size - patch_size) // (patch_size - overlap*2) + 1
    total_patches = vertical_patches * horizontal_patches * num_frames

    # Generator to yield patches to save memory
    def patch_generator():
        for frame_index in range(num_frames):
            for i in range(0, height + pad_size - patch_size + 1, patch_size - overlap*2):
                for j in range(0, width + pad_size - patch_size + 1, patch_size - overlap*2):
                    # Calculate the window of the patch
                    yield padded_video[:, frame_index:frame_index + 1, :, i:i + patch_size, j:j + patch_size].squeeze(1)

    # Initialize the tensor for patches
    patches = torch.zeros((batch_size, total_patches, channels, patch_size, patch_size), device=video_tensor.device)

    # Use the generator to fill in the patches tensor
    for batch_index in range(batch_size):
        patch_index = 0
        for patch_tensor in patch_generator():
            patches[batch_index, patch_index] = patch_tensor
            patch_index += 1
            if patch_index >= total_patches:
                break

    return patches





def reconstruct_video_from_patches_improved_fade(patches, original_video_shape, patch_size=16, overlap=4):
    batch_size, num_frames, height, width, channels = original_video_shape
    # Calculate the actual patch size, including the overlap
    actual_patch_size = patch_size + overlap * 2

    # Calculate number of patches per dimension, accounting for the overlap
    num_patches_h = (height + overlap*2) // patch_size
    num_patches_w = (width + overlap*2) // patch_size

    reconstructed_video = torch.zeros((batch_size, num_frames, height, width, channels), device=patches.device)
    weight_matrix = torch.zeros_like(reconstructed_video, device=patches.device)

    # Generate the weight mask with a Gaussian-like transition for smoother blending
    linear_weight = torch.linspace(1, 0, overlap, device=patches.device)
    gaussian_weight = torch.exp(-5 * linear_weight ** 2)
    smooth_transition = torch.cat(
        (gaussian_weight, torch.ones(patch_size, device=patches.device), gaussian_weight.flip(0)))
    full_weight_mask = torch.outer(smooth_transition, smooth_transition).unsqueeze(-1).expand(-1, -1, channels)

    patch_idx = 0
    for frame in range(num_frames):
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                if patch_idx >= patches.shape[1]:  # Check to avoid going out of bounds
                    print(f"Reached the end of patches at index {patch_idx}")
                    break

                # Calculate the starting indices considering the overlap
                start_i = i * patch_size
                start_j = j * patch_size
                end_i = start_i + actual_patch_size
                end_j = start_j + actual_patch_size

                # Adjust for boundary conditions
                patch_height = min(end_i - start_i, height - start_i)
                patch_width = min(end_j - start_j, width - start_j)

                # Extract the current patch and corresponding weight mask
                patch = patches[:, patch_idx, :patch_height, :patch_width, :]
                mask = full_weight_mask[:patch_height, :patch_width, :]

                #print(f"Applying patch {patch_idx} to position ({start_i}, {start_j}) with size {patch.shape}")

                # Apply patch and mask to reconstructed video
                reconstructed_video[:, frame, start_i:start_i + patch_height, start_j:start_j + patch_width,
                :] += patch * mask
                weight_matrix[:, frame, start_i:start_i + patch_height, start_j:start_j + patch_width, :] += mask
                patch_idx += 1

    # Avoid division by zero
    weight_matrix[weight_matrix == 0] = 1
    # Normalize the reconstruction by the weight matrix to average the overlaps
    reconstructed_video /= weight_matrix

    reconstructed_video = reconstructed_video[:, :, overlap:, overlap:-overlap]

    return reconstructed_video



def reconstruct_video_from_patches_improved_fade2(patches, original_video_shape, patch_size=16, overlap=4, temporal_depth=10):
    frames = original_video_shape[0]
    print(frames)
    num_frames = (frames // temporal_depth) * temporal_depth
    print(num_frames, frames // temporal_depth, frames % temporal_depth)
    if (frames % temporal_depth) > 0:
        num_frames += temporal_depth
    batch_size, _, height, width, channels = original_video_shape
    # Calculate the actual patch size, including the overlap
    actual_patch_size = patch_size + overlap * 2

    # Calculate number of patches per dimension, accounting for the overlap
    num_patches_h = (height + overlap*2) // patch_size
    num_patches_w = (width + overlap*2) // patch_size

    reconstructed_video = torch.zeros((batch_size, num_frames, height, width, channels), device=patches.device)
    weight_matrix = torch.zeros_like(reconstructed_video, device=patches.device)

    # Generate the weight mask with a Gaussian-like transition for smoother blending
    linear_weight = torch.linspace(1, 0, overlap, device=patches.device)
    gaussian_weight = torch.exp(-5 * linear_weight ** 2)
    smooth_transition = torch.cat(
        (gaussian_weight, torch.ones(patch_size, device=patches.device), gaussian_weight.flip(0)))
    full_weight_mask = torch.outer(smooth_transition, smooth_transition).unsqueeze(-1).expand(-1, -1, channels)

    patch_idx = 0
    for frame in range(num_frames):
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                if patch_idx >= patches.shape[1]:  # Check to avoid going out of bounds
                    print(f"Reached the end of patches at index {patch_idx}")
                    break

                # Calculate the starting indices considering the overlap
                start_i = i * patch_size
                start_j = j * patch_size
                end_i = start_i + actual_patch_size
                end_j = start_j + actual_patch_size

                # Adjust for boundary conditions
                patch_height = min(end_i - start_i, height - start_i)
                patch_width = min(end_j - start_j, width - start_j)

                # Extract the current patch and corresponding weight mask
                patch = patches[:, patch_idx, :patch_height, :patch_width, :]
                mask = full_weight_mask[:patch_height, :patch_width, :]

                #print(f"Applying patch {patch_idx} to position ({start_i}, {start_j}) with size {patch.shape}")

                # Apply patch and mask to reconstructed video
                reconstructed_video[:, frame, start_i:start_i + patch_height, start_j:start_j + patch_width,
                :] += patch * mask
                weight_matrix[:, frame, start_i:start_i + patch_height, start_j:start_j + patch_width, :] += mask
                patch_idx += 1

    # Avoid division by zero
    weight_matrix[weight_matrix == 0] = 1
    # Normalize the reconstruction by the weight matrix to average the overlaps
    reconstructed_video /= weight_matrix

    reconstructed_video = reconstructed_video[:, :, overlap:, overlap:-overlap]

    return reconstructed_video


def pad_video_to_divisible_by_size(video_tensor, size=16):
    """
    Pads the video tensor's width and height to make them divisible by 32.

    Parameters:
    video_tensor (torch.Tensor): A video tensor of shape [batch_size, num_frames, width, height, 3].

    Returns:
    torch.Tensor: The padded video tensor.
    """
    _, _, width, height, _ = video_tensor.shape
    # Calculate the required padding for width and height
    pad_width = (size - width % size) % size
    pad_height = (size - height % size) % size

    # Pad the tensor. PyTorch's F.pad expects padding as (left, right, top, bottom),
    # but since we're only padding right and bottom, left and top paddings are 0.
    padded_video = F.pad(video_tensor, (0, 0, 0, pad_height, 0, pad_width), mode='constant', value=0)

    return padded_video





def crop_video_to_divisible_by_size(video_tensor, size=16):
    """
    Crops the video tensor's width and height to make them divisible by 32,
    handling edge cases where the video is too small or an uneven number of pixels needs to be cropped.

    Parameters:
    video_tensor (torch.Tensor): A video tensor of shape [batch_size, num_frames, width, height, 3].

    Returns:
    torch.Tensor: The cropped video tensor, or the original tensor if it cannot be cropped as specified.
    """
    _, _, width, height, _ = video_tensor.shape

    # Ensure we don't crop if width or height is already less than or equal to 32
    if width <= size or height <= size:
        #print("Width or height is less than or equal to 32 pixels; returning original video tensor.")
        return video_tensor

    # Calculate the required cropping for width and height
    crop_width = width % size
    crop_height = height % size

    # Calculate how much to crop from each side
    crop_left = crop_width // 2
    crop_right = crop_width - crop_left
    crop_top = crop_height // 2
    crop_bottom = crop_height - crop_top

    # Adjust for uneven cropping
    # If after cropping, the dimensions are less than 32 or not divisible by 32 (which shouldn't happen with correct calculation), adjust accordingly
    if width - crop_left - crop_right <= 0 or height - crop_top - crop_bottom <= 0:
        print("Adjustment needed to prevent empty tensor.")
        # Adjust the cropping logic as needed
        # This block is more of a safeguard and shouldn't be reached with correct initial calculations

    # Crop the tensor
    cropped_video = video_tensor[:, :, crop_left:width - crop_right, crop_top:height - crop_bottom, :]

    return cropped_video



