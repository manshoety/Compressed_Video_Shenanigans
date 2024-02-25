import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class FeatureRefinementLayer(nn.Module):
    def __init__(self, in_channels):
        super(FeatureRefinementLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class SharedConvBase(nn.Module):
    def __init__(self):
        super(SharedConvBase, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, shared_base, input_height=24, input_width=24, latent_dim=32):
        super(Encoder, self).__init__()
        self.shared_base = shared_base
        # Calculate the output size after shared_base
        self.feature_size = self._get_feature_size(input_height, input_width)
        self.feature_refinement = FeatureRefinementLayer(256)
        self.to_latent = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Linear(self.feature_size, latent_dim)  # Dynamically calculated based on input size
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.to_latent.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                init.constant_(m.bias, 0)

    def _get_feature_size(self, H, W):
        # Simulate forward pass through shared_base to get output feature map size
        with torch.no_grad():
            sample_input = torch.zeros(1, 3, H, W)
            output = self.shared_base(sample_input)
            return output.nelement() // output.shape[0]  # Total elements per item in batch

    def forward(self, x):
        batch_size, num_frames, height, width, channels = x.size()
        x = x.view(batch_size * num_frames, channels, height, width)
        x = self.shared_base(x)
        x = self.feature_refinement(x)  # Refine features before flattening
        _, C, H, W = x.size()
        x = x.view(batch_size, num_frames, C * H * W)
        x = self.to_latent(x)
        return x


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super(AdaIN, self).__init__()
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, content, style):
        style = self.fc(style)
        scale, shift = style.chunk(2, dim=1)
        scale = scale.unsqueeze(2).unsqueeze(3)
        shift = shift.unsqueeze(2).unsqueeze(3)
        normalized_content = F.instance_norm(content)
        return normalized_content * scale + shift

class AttentionLayer(nn.Module):
    def __init__(self, channels):
        super(AttentionLayer, self).__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()
        query = self.query(x).view(b, -1, w * h).permute(0, 2, 1)
        key = self.key(x).view(b, -1, w * h)
        value = self.value(x).view(b, -1, w * h)

        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        return out + x


class Decoder(nn.Module):
    def __init__(self, latent_dim=32, output_channels=3, patch_size=24, overlap=4):
        super(Decoder, self).__init__()
        self.patch_size = patch_size
        self.overlap = overlap
        self.input_dim = latent_dim + 3  # Including positional encodings
        self.output_channels = output_channels
        self.funny_number = 128

        # Calculate the dimensions dynamically based on patch_size
        self.expanded_dim = self.calculate_expanded_dim()
        self.fc = nn.Linear(self.input_dim, self.expanded_dim)
        self.attention = AttentionLayer(self.funny_number // 4)

        # Assuming dynamic calculation of layers based on patch_size
        self.conv_layers = self._build_conv_layers()

    def calculate_expanded_dim(self):
        # Dynamically calculate the output size for the fully connected layer
        return self.funny_number * self.patch_size * self.patch_size

    def _build_conv_layers(self):
        # Dynamically build convolutional layers based on patch_size
        layers = nn.Sequential(
            nn.Conv2d(self.funny_number, self.funny_number // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.funny_number // 2),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(self.funny_number // 2, self.funny_number // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.funny_number // 4),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(self.funny_number // 4, self.output_channels * 4, kernel_size=3, padding=1),
            # Adjusted to ensure divisibility
            nn.PixelShuffle(upscale_factor=2),  # Sub-pixel convolution for upscaling
            nn.Conv2d(self.output_channels, self.output_channels, kernel_size=3, stride=2, padding=1), # downsample
        )
        return layers

    def forward(self, x, video_shape):
        _, num_frames, width, height, _ = video_shape
        batch_size, num_patches, _ = x.shape
        device = x.device

        # Calculate the number of patches along width and height
        num_patches_horizontally = width // (self.patch_size - self.overlap*2)
        num_patches_vertically = height // (self.patch_size - self.overlap*2)
        num_patches_per_frame = num_patches_horizontally * num_patches_vertically

        # Create positional encodings for patches
        pos_x = torch.arange(0, num_patches_horizontally, device=device).float() / (num_patches_horizontally - 1)
        pos_y = torch.arange(0, num_patches_vertically, device=device).float() / (num_patches_vertically - 1)
        pos_t = torch.arange(0, num_frames, device=device).float() / (num_frames - 1)

        # Expand positional encodings to match the batch size and the number of patches
        pos_x = pos_x.repeat(batch_size * num_frames, num_patches_vertically).reshape(-1, 1)
        pos_y = pos_y.repeat(batch_size * num_frames, num_patches_horizontally).transpose(0, 1).reshape(-1, 1)
        pos_t = pos_t.repeat(batch_size, num_patches_per_frame).reshape(-1, 1)

        # Concatenate positional encodings with the patch representations
        pos_encodings = torch.cat((pos_x, pos_y, pos_t), dim=1).repeat(1, batch_size).view(batch_size, num_patches, -1)
        x = torch.cat((x, pos_encodings), dim=2)

        # Preparing the output tensor
        output_patches = []

        # Process up to 1024 patches at a time
        max_patches_per_batch = 1024*100
        for i in range(0, num_patches, max_patches_per_batch):
            # Extract up to 1024 patches
            x_batch = x[:, i:i + max_patches_per_batch]

            # Forward pass through the decoder for the batch
            x_batch = self.fc(x_batch)
            x_batch = x_batch.view(-1, self.funny_number, self.patch_size, self.patch_size)
            x_batch = self.conv_layers(x_batch)
            x_batch = torch.sigmoid(x_batch).permute(0, 2, 3, 1)

            # Collect processed patches
            output_patches.append(x_batch)

        # Combine all processed patches
        decoded_patches = torch.cat(output_patches, dim=0)

        # Debug prints
        #print(f"Decoded patches shape: {decoded_patches.shape}")

        return decoded_patches


class UnifiedModel(nn.Module):
    def __init__(self, latent_dim=32, output_channels=3, patch_size=24):
        super(UnifiedModel, self).__init__()
        # this was used for two encoders but now I only use one so it's not really "shared"
        self.shared_base = SharedConvBase()
        self.encoder = Encoder(self.shared_base, latent_dim=latent_dim, input_height=patch_size, input_width=patch_size)
        self.decoder = Decoder(latent_dim=latent_dim, output_channels=output_channels, patch_size=patch_size)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, encoded, video_shape):
        return self.decoder(encoded, video_shape)
