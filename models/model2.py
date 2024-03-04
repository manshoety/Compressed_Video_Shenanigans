import torch
import numpy as np
from torch.nn.utils import spectral_norm
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EnhancedSpatiotemporalAxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, heads=32, kernel_size=3, temporal_kernel_size=3, dilation_rate=2,
                 groups=32):
        super(EnhancedSpatiotemporalAxialAttention, self).__init__()
        self.groups = groups
        self.heads = heads
        assert out_planes % heads == 0, "out_planes should be divisible by the number of heads"

        self.kernel_size = kernel_size
        self.temporal_kernel_size = temporal_kernel_size
        self.dilation_rate = dilation_rate
        self.in_planes = in_planes
        self.out_planes = out_planes // heads

        self.temporal_pe = nn.Parameter(torch.randn(1, 1, out_planes // heads, 1, 1))
        self.spatial_pe = nn.Parameter(torch.randn(1, 1, out_planes // heads, 1, 1))

        self.group_conv = nn.Conv3d(in_planes, out_planes, kernel_size=1, groups=self.groups, bias=False)
        self.bn = nn.GroupNorm(num_groups=groups, num_channels=out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.temporal_conv = nn.Conv3d(out_planes, out_planes, kernel_size=(temporal_kernel_size, 1, 1),
                                       padding=((temporal_kernel_size - 1) // 2 * dilation_rate, 0, 0),
                                       dilation=(dilation_rate, 1, 1), groups=out_planes, bias=False)
        self.height_conv = nn.Conv3d(out_planes, out_planes, kernel_size=(1, kernel_size, 1),
                                     padding=(0, (kernel_size - 1) // 2, 0), groups=out_planes, bias=False)
        self.width_conv = nn.Conv3d(out_planes, out_planes, kernel_size=(1, 1, kernel_size),
                                    padding=(0, 0, (kernel_size - 1) // 2), groups=out_planes, bias=False)

        self.layer_norm = nn.LayerNorm(out_planes)
        self.multihead_attn = nn.MultiheadAttention(out_planes, self.heads, batch_first=True)

        # Temporal attention layer
        self.temporal_attention = nn.MultiheadAttention(out_planes, self.heads, batch_first=True)

        self.skip_connection = nn.Identity()

    def forward(self, x):
        b, c, t, h, w = x.shape

        temporal_pe = self.temporal_pe.repeat(1, t, 1, h, w)
        spatial_pe = self.spatial_pe.repeat(1, t, 1, h, w)
        x = x + temporal_pe + spatial_pe

        x = self.group_conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x_flattened = x.flatten(2).permute(0, 2, 1)
        x_flattened = self.layer_norm(x_flattened)

        # Apply spatial multi-head attention
        attn_output, _ = self.multihead_attn(x_flattened, x_flattened, x_flattened)
        attn_output = attn_output + x_flattened

        # Reshape for temporal attention
        x_reshaped_for_temporal_attn = attn_output.permute(0, 2, 1).view(b, -1, t, h, w).flatten(3).permute(0, 3, 1, 2)
        temporal_attn_output, _ = self.temporal_attention(x_reshaped_for_temporal_attn, x_reshaped_for_temporal_attn,
                                                          x_reshaped_for_temporal_attn)
        temporal_attn_output = temporal_attn_output + x_reshaped_for_temporal_attn
        temporal_attn_output = temporal_attn_output.permute(0, 2, 1).view(b, -1, t, h, w)

        attn_output = temporal_attn_output.permute(0, 2, 1).view(b, -1, t, h, w)

        temporal_att = self.temporal_conv(attn_output)
        height_att = self.height_conv(attn_output)
        width_att = self.width_conv(attn_output)
        combined_att = temporal_att + height_att + width_att
        combined_att = self.skip_connection(combined_att) + x

        return combined_att


class CorrectedSpacetimeCompression(nn.Module):
    # Adjusting to ensure the temporal dimension remains at 10
    def __init__(self, input_channels, output_channels):
        super(CorrectedSpacetimeCompression, self).__init__()
        self.compress = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(output_channels),
            # Temporal dimension preservation
            nn.Conv3d(output_channels, output_channels, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(output_channels)
        )

    def forward(self, x):
        compressed = self.compress(x)
        # Adjustments to ensure the temporal dimension is maintained
        return compressed


class TemporalAwarePatchTransitionLearning(nn.Module):
    def __init__(self, channels):
        super(TemporalAwarePatchTransitionLearning, self).__init__()
        self.temporal_adaptation = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=channels, bias=False),
            nn.Sigmoid()
        )
        self.spatial_transition_learn = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Learn temporal transitions
        temporal_transition = self.temporal_adaptation(x)

        # Apply learned transitions to the input
        x_temporally_adapted = x * temporal_transition

        # Learn and apply spatial transitions on temporally adapted features
        spatial_transition = self.spatial_transition_learn(x_temporally_adapted)
        transition_refined = x_temporally_adapted * spatial_transition

        return transition_refined


class TemporalAwarePatchBlendingModule(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super(TemporalAwarePatchBlendingModule, self).__init__()
        self.channels = channels
        padding = dilation * (kernel_size - 1) // 2

        self.process_conv = nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                                      groups=channels, bias=False)
        self.edge_weight_conv = nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                                          groups=channels, bias=False)
        self.temporal_gradient_conv = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0),
                                                groups=channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        processed = self.process_conv(x)
        processed = self.norm(processed)
        processed = self.relu(processed)

        edge_weight = self.edge_weight_conv(x)
        edge_weight = self.sigmoid(edge_weight)  # Normalize weights to [0, 1]

        # Compute temporal gradients to inform blending weights
        temporal_gradient = self.temporal_gradient_conv(x)
        temporal_gradient = self.sigmoid(temporal_gradient)  # Normalize gradients to [0, 1]

        # Adjust edge weights based on temporal gradient
        adjusted_edge_weight = edge_weight * (1 - temporal_gradient) + temporal_gradient

        blended = processed * adjusted_edge_weight + x * (1 - adjusted_edge_weight)

        return blended

class EnhancedVideoEncoder2(nn.Module):
    def __init__(self, input_channels=3, compressed_channels=4):
        super(EnhancedVideoEncoder2, self).__init__()
        self.spacetime_compression = CorrectedSpacetimeCompression(input_channels, compressed_channels)
        self.skip_connection = nn.Conv3d(input_channels, compressed_channels, 1)
        self.patch_blending = TemporalAwarePatchBlendingModule(compressed_channels)
        self.patch_transition = TemporalAwarePatchTransitionLearning(compressed_channels)

    def forward(self, x):
        compressed = self.spacetime_compression(x)
        skip = F.interpolate(self.skip_connection(x), size=compressed.size()[2:], mode='trilinear', align_corners=False)
        blended = self.patch_blending(compressed + skip)
        transition_refined = self.patch_transition(blended)
        return transition_refined



class EnhancedLocalFeatureEnhancement(nn.Module):
    def __init__(self, channels):
        super(EnhancedLocalFeatureEnhancement, self).__init__()
        self.enhance = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),  # Additional layer
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),  # Additional layer
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Apply local feature enhancement with residual connection
        return self.relu(x + self.enhance(x))


class EnhancedCrossPatchCoherenceLearning(nn.Module):
    def __init__(self, channels):
        super(EnhancedCrossPatchCoherenceLearning, self).__init__()
        self.coherence = nn.Sequential(
            nn.Conv3d(channels, channels // 2, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 2, channels // 2, kernel_size=3, padding=1, bias=False),  # Additional refinement layer
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 2, channels, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, original):
        # Generate a coherence map
        coherence_map = self.coherence(x)
        # Apply the coherence map for improved blending
        return original * coherence_map + x * (1 - coherence_map)


class ColorCorrectionModule(nn.Module):
    def __init__(self, channels):
        super(ColorCorrectionModule, self).__init__()
        self.color_correction = nn.Sequential(
            nn.Conv3d(channels, channels // 2, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm3d(channels // 2),  # Use instance normalization for better color consistency
            nn.ReLU(),
            nn.Conv3d(channels // 2, channels // 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(channels // 2),  # Instance normalization
            nn.ReLU(),
            nn.Conv3d(channels // 2, channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        correction = self.color_correction(x)
        corrected = correction * 2 - 1  # Output range to [-1, 1]
        return x + corrected


class PatchEdgeSmoothing(nn.Module):
    def __init__(self, channels):
        super(PatchEdgeSmoothing, self).__init__()
        self.edge_smoothing = nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.blend = nn.Sigmoid()  # Sigmoid for smooth blending

    def forward(self, x):
        smoothed_edges = self.edge_smoothing(x)
        blended_output = x * (1 - self.blend(smoothed_edges)) + smoothed_edges * self.blend(smoothed_edges)
        return blended_output


class TemporalSmoothingLayer(nn.Module):
    def __init__(self, channels):
        super(TemporalSmoothingLayer, self).__init__()
        # A convolution layer with a kernel size of 3 in the temporal dimension to smooth over time
        self.temporal_smoothing = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), groups=channels)

    def forward(self, x):
        return self.temporal_smoothing(x)


class EnhancedVideoDecoder2(nn.Module):
    def __init__(self, compressed_channels=4, output_channels=3):
        super(EnhancedVideoDecoder2, self).__init__()
        self.decode = nn.Sequential(
            nn.ConvTranspose3d(compressed_channels, compressed_channels // 2, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1)),
            nn.ReLU(),
            nn.BatchNorm3d(compressed_channels // 2),
            nn.ConvTranspose3d(compressed_channels // 2, compressed_channels // 4, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1)),
            nn.ReLU(),
            nn.BatchNorm3d(compressed_channels // 4),
            nn.ConvTranspose3d(compressed_channels // 4, output_channels, (1, 3, 3), stride=1, padding=(0, 1, 1)),  # Corrected to output_channels
            nn.Tanh()  # Tanh activation to constrain output values to [-1, 1]
        )
        self.patch_edge_smoothing = PatchEdgeSmoothing(output_channels)
        self.patch_blending = TemporalAwarePatchBlendingModule(output_channels)
        self.local_feature_enhancement = EnhancedLocalFeatureEnhancement(output_channels)
        self.cross_patch_coherence = EnhancedCrossPatchCoherenceLearning(output_channels)
        self.color_correction = ColorCorrectionModule(output_channels)
        self.temporal_smoothing = TemporalSmoothingLayer(output_channels)

    def forward(self, x):
        decoded = self.decode(x)
        edge_smoothed = self.patch_edge_smoothing(decoded)
        blended = self.patch_blending(edge_smoothed)
        enhanced = self.local_feature_enhancement(blended)
        final_output = self.cross_patch_coherence(enhanced, blended)
        color_corrected_output = self.color_correction(final_output)
        temporally_smoothed_output = self.temporal_smoothing(color_corrected_output)
        final_smoothed_output = self.patch_edge_smoothing(temporally_smoothed_output)
        return final_smoothed_output






def generate_synthetic_video(frames, height, width, channels=3):
    """
    Generates a synthetic video tensor with specified dimensions.
    """
    return torch.rand((frames, height, width, channels))


def test_video_processing(video_resolutions):
    """
    Tests video processing for a list of resolutions.
    Each resolution is a tuple: (frames, height, width).
    """
    for resolution in video_resolutions:
        print(f"Testing video resolution: {resolution}")
        video = generate_synthetic_video(*resolution).to(device)
        video = adjust_video_size(video, 64)
        original_video_shape = video.shape
        print(f"Fake video: {video.shape}")
        patches = video_to_patches(video, patch_size=64, temporal_depth=10)
        print(f"Generated {patches.shape[0]} patches of shape {patches.shape[1:]}")
        patches = patches.permute(0, 4, 1, 2, 3)
        print(f"Patches shape: {patches.shape}")

        # Assuming encoder and decoder are defined as previously
        encoded_patches = encoder(patches)
        print(f"Encoded patches shape: {encoded_patches.shape}")
        decoded_video = decoder(encoded_patches)
        print(f"Decoded patches shape: {decoded_video.shape}")

        decoded_video = decoded_video.permute(0, 2, 3, 4, 1)
        print(f"Reshaped Decoded patches: {decoded_video.shape}")

        video = patches_to_video222(decoded_video, video.shape, patch_size=64, temporal_depth=10)
        print(f"Reconstructed Video: {video.shape}")

        # Checks (simplified and based on expected behavior rather than exact output shape)
        assert video.shape == original_video_shape, "Decoded video patch size mismatch"
        print("Test passed\n")


video_resolutions = [
    (20, 128, 256),  # 20 frames, 128x256 resolution
    (30, 192, 192),  # 30 frames, 192x192 resolution
    (10, 64, 64),  # 10 frames, 64x64 resolution
    (20, 128, 256)
]
