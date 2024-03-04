import os
import subprocess

import cv2
import ffmpeg
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# this file is a mess lmao

def create_refined_edge_weight_map(patch_size, edge_weight=10):
    H, W = patch_size
    # Create a meshgrid for the patch dimensions
    y = torch.linspace(-1, 1, steps=H) ** 2  # Squaring to increase gradient towards edges
    x = torch.linspace(-1, 1, steps=W) ** 2  # Squaring to increase gradient towards edges
    xv, yv = torch.meshgrid(x, y)  # Create a meshgrid
    weight_map = 1 + (edge_weight - 1) * torch.sqrt(xv**2 + yv**2)  # Calculate distance-based weight
    return weight_map


def create_edge_weight_map(patch_size, edge_weight=10):
    """
    Create a weight map for a patch, with higher weights on the edges.

    Args:
        patch_size (tuple): The size (H, W) of the patch.
        edge_weight (float): The weight to apply to the edges.

    Returns:
        torch.Tensor: A weight map with higher values at the edges.
    """
    H, W = patch_size
    weight_map = torch.ones((H, W))

    # Apply higher weights to the edges
    weight_map[:, :1] *= edge_weight  # Left edge
    weight_map[:, -1:] *= edge_weight  # Right edge
    weight_map[:1, :] *= edge_weight  # Top edge
    weight_map[-1:, :] *= edge_weight  # Bottom edge

    return weight_map

class EdgeWeightedLoss(nn.Module):
    def __init__(self, base_loss_func=nn.MSELoss(reduction='none'), edge_weight=10, patch_size=64):
        """
        Initializes the edge-weighted loss module.

        Args:
            base_loss_func (callable): The base loss function (e.g., nn.MSELoss).
            edge_weight (float): The weight to apply to the edges.
        """
        super(EdgeWeightedLoss, self).__init__()
        self.base_loss_func = base_loss_func
        self.edge_weight = edge_weight
        # Assuming fixed patch size for simplicity; dynamically adjust if necessary
        self.weight_map = create_refined_edge_weight_map((patch_size, patch_size), edge_weight=edge_weight).unsqueeze(0).unsqueeze(-1)  # Add batch and channel dims
        self.weight_map = self.weight_map + create_edge_weight_map((patch_size, patch_size), edge_weight=edge_weight//2).unsqueeze(0).unsqueeze(-1)

    def forward(self, prediction, target):
        """
        Calculate the edge-weighted loss between prediction and target.

        Args:
            prediction (torch.Tensor): The predicted patches.
            target (torch.Tensor): The ground truth patches.

        Returns:
            torch.Tensor: The calculated loss.
        """
        # Ensure weight_map is on the same device as the input tensors
        weight_map = self.weight_map.to(prediction.device)

        # Calculate base loss per element
        per_element_loss = self.base_loss_func(prediction, target)

        # Apply weight map
        weighted_loss = per_element_loss * weight_map

        # Average over the batch and channel dimensions
        loss = weighted_loss.mean(dim=(0, 1, -1))  # Average over batch, temporal, and channel dims

        return loss.mean()  # Further average over spatial dimensions


def color_consistency_loss(output, target):
    # Calculate the mean color of each channel in the output and target
    output_mean_color = output.reshape(output.size(0), output.size(1), -1).mean(2)
    target_mean_color = target.reshape(target.size(0), target.size(1), -1).mean(2)

    # The loss is the L1 distance between the mean colors
    loss = F.l1_loss(output_mean_color, target_mean_color)
    return loss


def temporal_consistency_loss2(output, target):
    # Calculate the difference between consecutive frames in the output and target
    output_diff = output[:, :, 1:] - output[:, :, :-1]
    target_diff = target[:, :, 1:] - target[:, :, :-1]

    # The loss is the L1 distance between the differences of consecutive frames
    loss = F.l1_loss(output_diff, target_diff)
    return loss


from torchvision.models import vgg16
from pytorch_msssim import ssim, ms_ssim

class SSIMLoss(torch.nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, x, y):
        # Calculate SSIM loss between two sets of images
        return 1 - ssim(x, y, data_range=1, size_average=True)  # Assuming x and y are in [0, 1]

class FrequencyDomainLoss(torch.nn.Module):
    def __init__(self):
        super(FrequencyDomainLoss, self).__init__()

    def forward(self, x, y):
        # Convert images to frequency domain using FFT
        x_freq = torch.fft.fft2(x)
        y_freq = torch.fft.fft2(y)

        # Calculate L1 loss in the frequency domain
        loss = F.l1_loss(x_freq.abs(), y_freq.abs())
        return loss

class FeatureConsistencyLoss(torch.nn.Module):
    def __init__(self):
        super(FeatureConsistencyLoss, self).__init__()
        self.vgg = vgg16(pretrained=True).features[:16].to(device)  # Use the first few layers of VGG16 for feature extraction
        for param in self.vgg.parameters():
            param.requires_grad = False  # Freeze VGG parameters

    def forward(self, x, y):
        # Reshape and permute inputs to match VGG input format
        batch_size, temporal_dimension, height, width, channels = x.shape
        x = x.permute(0, 1, 4, 2, 3).reshape(batch_size * temporal_dimension, channels, height, width)
        y = y.permute(0, 1, 4, 2, 3).reshape(batch_size * temporal_dimension, channels, height, width)

        # Normalize inputs
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1).to(x.device)
        x_vgg = (x - mean) / std
        y_vgg = (y - mean) / std

        # Process in smaller batches
        sub_batch_size = 16  # Adjust based on your VRAM capacity
        total_loss = 0.0
        num_batches = max(x_vgg.size(0) // sub_batch_size, 1)

        for i in range(0, x_vgg.size(0), sub_batch_size):
            x_sub = x_vgg[i:i+sub_batch_size]
            y_sub = y_vgg[i:i+sub_batch_size]

            x_features = self.vgg(x_sub)
            y_features = self.vgg(y_sub)

            # Calculate L2 loss for the current sub-batch and accumulate
            total_loss += F.mse_loss(x_features, y_features) * (x_sub.size(0) / x_vgg.size(0))

        return total_loss




































from encoder_decoder2 import UnifiedModel
from modelfunny import EnhancedDecoder, init_weights2
from t10 import EnhancedVideoEncoder2, EnhancedVideoDecoder2, patches_to_video222, video_to_patches, adjust_video_size, patches_to_video_debug, Discriminator, weights_init_d


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from rand_vid_dataset import RandVidDatasetAnyResOnlyVid, Picker
from utils import split_video_into_patches, reconstruct_video_from_patches_improved_fade

length = 5

dataloader = RandVidDatasetAnyResOnlyVid(r'G:\data_stuff_dont_delete\rand_vids_tensors_any_res', batch_size=1, vid_length=length)
dataloader.start_loading()

from torch.cuda.amp import GradScaler, autocast

# Initialize the gradient scaler for mixed precision training
scaler = GradScaler()
import time
import threading
import traceback
import sys

def train():
    repeats = 3

    patch_size = 64
    encoded_dim = 48  # Spatial (32) + Temporal (16)

    #encoder_spacial = VideoPatchEncoderWindowed(patch_size=patch_size, embed_dim=SPACIAL_DIM).to(device).train()
    #encoder_temporal = VideoPatchEncoderWindowed(patch_size=patch_size, embed_dim=TEMPORAL_DIM).to(device).train()

    #decoder = VideoPatchDecoderWindowed().to(device).train()

    #model = UnifiedModel().to(device).apply(init_weights).train()
    # model = UnifiedModel().to(device).eval()
    #model.load_state_dict(torch.load(r"models\1_17600_encoder_decoder/model.pth"))
    encoder = EnhancedVideoEncoder2().to(device).apply(init_weights).train()
    decoder = EnhancedVideoDecoder2().to(device).apply(init_weights).train()
    # colorer = ContextAwareColorCorrectionModule().to(device).apply(init_weights).train()
    edge_weighted_loss = EdgeWeightedLoss(edge_weight=10, patch_size=patch_size)
    ssim_loss = SSIMLoss()
    freq_loss = FrequencyDomainLoss()
    feat_loss = FeatureConsistencyLoss()
    #discriminator = Discriminator(10, patch_size, patch_size).to(device).apply(weights_init_d).train()
    #encoder.load_state_dict(torch.load(r"models\1_87600Z/encoder.pth"))
    #decoder.load_state_dict(torch.load(r"models\1_87600Z/decoder.pth"))

    #patchEnhancer = EnhancedDecoder().to(device).apply(init_weights2).train()


    #encoder.train()
    #motion_vector_encoder.train()
    #decoder.train()
    # patch_extractor = SimplePatchExtractorWithPadding()

    #encoder.load_state_dict(torch.load(r"models\1_4400/encoder.pth"))
    #motion_vector_encoder.load_state_dict(torch.load(r"models\1_4400/motion_vector_encoder.pth"))
    #decoder.load_state_dict(torch.load(r"models\1_4400/decoder.pth"))
    #discriminator = VideoDiscriminator().apply(init_weights).to(device)

    # Optimizer for both models (adjust learning rates as needed)
    #optimizer = optim.Adam(list(model.parameters()), lr=0.0005)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.005)
    # optimizer = optim.Adam(list(colorer.parameters()), lr=0.005)
    use_discrim = False
    #optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))
    # adversarial_loss = nn.BCELoss()
    #discriminator_optimizer = optim.Adam(list(discriminator.parameters()),
    #                                     lr=0.00002)

    picker = Picker(duration=length)
    lambda_adv = 0.5

    step_offset = 0

    # Training loop
    for epoch in range(9999999):  # Adjust number of epochs
        step = step_offset
        for i in range(len(dataloader)):  # Assuming dataloader provides videos and optional labels
            batch = dataloader.get_batch()
            if batch is not None:
                for j in range(repeats):
                    start = time.time()
                    videos, _ = picker.get_item(batch)
                    videos = videos.permute(0, 2, 3, 4, 1)
                    videos = adjust_video_size(videos.squeeze(), patch_size)
                    normalized_video = preprocessor(videos)
                    patches = video_to_patches(normalized_video, patch_size=patch_size, temporal_depth=10).to(device)
                    real_patches = video_to_patches(videos, patch_size=patch_size, temporal_depth=10).to(device)



                    #print(videos.shape)
                    #videos = adjust_video_size(videos.squeeze(), 16).unsqueeze(0)
                    #optimizer.zero_grad()
                    #normalized_video = preprocessor(videos)
                    #print(videos.shape, normalized_video.shape)
                    #patches = split_video_into_patches(normalized_video).to(device)
                    #real_patches = split_video_into_patches(videos).to(device)
                    #print(patches.shape, real_patches.shape)
                    #encoded_patches = model.encode(patches)
                    #print(f'encoded_patches: {encoded_patches.shape}')
                    #decoded_patches = model.decode(encoded_patches, videos.shape).unsqueeze(0)
                    #print(f'decoded_patches: {decoded_patches.shape}')
                    #reconstruction_loss = F.mse_loss(decoded_patches, real_patches)


                    if not use_discrim:
                        optimizer.zero_grad()
                        #print(videos.shape)
                        #print(patches.shape, real_patches.shape)
                        # with torch.no_grad():
                        encoded_patches = encoder(patches.permute(0, 4, 1, 2, 3))
                        decoded_patches = decoder(encoded_patches).permute(0, 2, 3, 4, 1)


                        # decoded_patches = colorer(decoded_patches.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)

                        weighted_loss = edge_weighted_loss(decoded_patches, real_patches)
                        color_loss = color_consistency_loss(decoded_patches, real_patches)
                        temp_loss = temporal_consistency_loss2(decoded_patches, real_patches)
                        #print(decoded_patches.shape, real_patches.shape)
                        loss = weighted_loss + color_loss * 0.5 + temp_loss * 0.5
                        loss += ssim_loss(decoded_patches, real_patches) * 0.5
                        loss += freq_loss(decoded_patches, real_patches) * 0.5
                        loss += feat_loss(decoded_patches, real_patches) * 0.5


                        loss.backward()
                        optimizer.step()
                        # scaler.scale(reconstruction_loss).backward()
                        # scaler.step(optimizer)
                        # scaler.update()


                    else:



                        ######################
                        # Train Generator
                        ######################
                        optimizer.zero_grad()

                        # Generate a batch of patches
                        encoded_patches = encoder(patches.permute(0, 4, 1, 2, 3))
                        decoded_patches = decoder(encoded_patches).permute(0, 2, 3, 4, 1)

                        # Calculate loss
                        weighted_loss = edge_weighted_loss(decoded_patches, real_patches)
                        color_loss = color_consistency_loss(decoded_patches, real_patches)
                        temp_loss = temporal_consistency_loss2(decoded_patches, real_patches)
                        #grad_loss = gradient_loss(decoded_patches, real_patches)
                        g_loss = weighted_loss + color_loss * 0.5 + temp_loss * 0.5 #+ grad_loss * 0.5

                        # Convert to the correct shape for the discriminator
                        real_patches = real_patches.permute(0, 4, 1, 2,
                                                            3)  # Shape: (num_patches, channels, temporal_depth, patch_height, patch_width)
                        decoded_patches = decoded_patches.permute(0, 4, 1, 2,
                                                            3)  # Shape: (num_patches, channels, temporal_depth, patch_height, patch_width)

                        # Labels for real and fake data for discriminator
                        valid = torch.ones((real_patches.size(0), 1), requires_grad=False).to(device)
                        fake = torch.zeros((real_patches.size(0), 1), requires_grad=False).to(device)

                        # Adversarial loss for generator
                        adversarial_g_loss = adversarial_loss(discriminator(decoded_patches), valid)

                        # Total generator loss
                        total_g_loss = g_loss + adversarial_g_loss
                        total_g_loss.backward()
                        optimizer.step()

                        ######################
                        # Train Discriminator
                        ######################
                        optimizer_D.zero_grad()

                        # Measure discriminator's ability to classify real from generated samples
                        real_loss = adversarial_loss(discriminator(real_patches), valid)
                        fake_loss = adversarial_loss(discriminator(decoded_patches.detach()), fake)

                        # Total discriminator loss
                        total_d_loss = (real_loss + fake_loss) / 2
                        total_d_loss.backward()
                        optimizer_D.step()

                        real_patches = real_patches.permute(0, 2, 3, 4,
                                                            1)
                        decoded_patches = decoded_patches.permute(0, 2, 3, 4,
                                                            1)


                    step += 1
                    dataloader.start_loading()

                    if step % 1 == 0:  # Print average loss every 100 mini-batches
                        if not use_discrim:
                            print(f'Epoch: {epoch + 1}, Step: {step}, loss: {loss.item():.4f}, seconds per iteration: {time.time()-start}')#, loss_d: {discriminator_loss.item():.4f}')
                        else:
                            print(
                                f'Epoch: {epoch + 1}, Step: {step}, g_loss: {g_loss.item():.4f}, , d_loss: {total_d_loss.item():.4f}, seconds per iteration: {time.time() - start}')

                    if step % (50*1) == 0:
                        print(real_patches.shape, decoded_patches.shape)
                        print(videos.shape)
                        #reconstructed_video = reconstruct_video_from_patches_improved_fade(decoded_patches, videos.shape)
                        #reconstructed_real_video = reconstruct_video_from_patches_improved_fade(real_patches, videos.shape)
                        #reconstructed_enhanced_video = reconstruct_video_from_patches_improved_fade(decoded_patches, videos.shape)
                        #print('what')
                        #print(patches_to_video_debug(decoded_video.detach(), videos.shape, patch_size=16, temporal_depth=10).shape)
                        reconstructed_video = patches_to_video_debug(decoded_patches.detach(), videos.shape, patch_size=patch_size, temporal_depth=10)
                        #print('whats')
                        #print(patches_to_video_debug(real_patches.detach(), videos.shape, patch_size=16, temporal_depth=10).shape)
                        reconstructed_real_video = patches_to_video_debug(real_patches.detach(), videos.shape, patch_size=patch_size, temporal_depth=10)
                        #print('whatd')

                        filename = f'test_out/test_epoch_{epoch + 1}_step_{step}'
                        save_video(reconstructed_video.detach().permute(3, 0, 1, 2), fps=10, filename=filename)
                        filename = f'test_out/test_epoch_{epoch + 1}_step_{step}_REAL'
                        save_video(reconstructed_real_video.detach().permute(3, 0, 1, 2), fps=10, filename=filename)
                        #filename = f'test_out/test_epoch_{epoch + 1}_step_{step}_enhanced'
                        #save_video(reconstructed_enhanced_video[0].detach().permute(3, 0, 1, 2), fps=10, filename=filename)

                    if step % (400*1) == 0:
                        folder = f"models/{epoch + 1}_{step}"
                        os.makedirs(folder, exist_ok=True)
                        #torch.save(patchEnhancer.state_dict(), f"{folder}/patchEnhancer.pth")
                        #torch.save(encoder_spacial.state_dict(), f"{folder}/encoder_spacial.pth")
                        #torch.save(encoder_temporal.state_dict(), f"{folder}/encoder_temporal.pth")
                        #torch.save(model.state_dict(), f"{folder}/model.pth")
                        torch.save(decoder.state_dict(), f"{folder}/decoder.pth")
                        torch.save(encoder.state_dict(), f"{folder}/encoder.pth")
                        # torch.save(colorer.state_dict(), f"{folder}/colorer.pth")
                        #torch.save(motion_vector_encoder.state_dict(), f"{folder}/motion_vector_encoder.pth")
                        #torch.save(decoder.state_dict(), f"{folder}/decoder.pth")
                        if use_discrim:
                            torch.save(discriminator.state_dict(), f"{folder}/discriminator.pth")


                    if step % 10 == 0:
                        torch.cuda.empty_cache()
                        #dataloader.start_loading()
        if (epoch + 1) % 10 == 0:
            folder = f"models/{epoch + 1}_{step}"
            os.makedirs(folder, exist_ok=True)
            #torch.save(encoder.state_dict(), f"{folder}/encoder.pth")
            #torch.save(motion_vector_encoder.state_dict(), f"{folder}/motion_vector_encoder.pth")
            #torch.save(decoder.state_dict(), f"{folder}/decoder.pth")
            #torch.save(discriminator.state_dict(), f"{folder}/discriminator_model.pth")


train()

print("among us")
