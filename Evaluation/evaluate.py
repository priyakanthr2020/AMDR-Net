
import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# ============================================================
# 1. Degradation Analysis Module
# ============================================================

class DegradationAnalysisModule(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1)
        )

        self.predictor = nn.Sequential(
            nn.Flatten(),

            nn.Linear(64, 32),
            nn.ReLU(inplace=True),

            nn.Linear(32, 3)
        )

    def forward(self, x):

        feat = self.features(x)

        degradation = self.predictor(feat)

        return degradation


# ============================================================
# 2. Adaptive Feature Fusion Controller
# ============================================================

class AdaptiveFeatureFusionController(nn.Module):

    def __init__(self):
        super().__init__()

        self.controller = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),

            nn.Linear(32, 64),
            nn.Sigmoid()
        )

    def forward(self, degradation_vector):

        return self.controller(degradation_vector)


# ============================================================
# 3. Adaptive Feature Fusion
# ============================================================

class AdaptiveFeatureFusion(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, features, weights):

        weights = weights.unsqueeze(-1).unsqueeze(-1)

        return features * weights


# ============================================================
# 4. Residual Block
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1
            )
        )

    def forward(self, x):

        return x + self.block(x)


# ============================================================
# 5. Spatial Branch
# ============================================================

class SpatialBranch(nn.Module):

    def __init__(self, channels=64):

        super().__init__()

        self.body = nn.Sequential(

            ResidualBlock(channels),
            ResidualBlock(channels),
            ResidualBlock(channels),
            ResidualBlock(channels)
        )

    def forward(self, x):

        return self.body(x)


# ============================================================
# 6. FFT Feature Extractor
# ============================================================

class FFTFeatureExtractor(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, x):

        fft = torch.fft.fft2(x)

        fft = torch.fft.fftshift(fft)

        magnitude = torch.abs(fft)

        magnitude = torch.log1p(magnitude)

        return magnitude


# ============================================================
# 7. Frequency Attention
# ============================================================

class FrequencyAttention(nn.Module):

    def __init__(self, channels=64):

        super().__init__()

        self.attention = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        attention = self.attention(x)

        return x * attention


# ============================================================
# 8. Frequency Branch
# ============================================================

class FrequencyBranch(nn.Module):

    def __init__(self, channels=64):

        super().__init__()

        self.fft = FFTFeatureExtractor()

        self.attention = FrequencyAttention(
            channels
        )

    def forward(self, x):

        x = self.fft(x)

        x = self.attention(x)

        return x


# ============================================================
# 9. Dual Branch Feature Extractor
# ============================================================

class DualBranchFeatureExtractor(nn.Module):

    def __init__(self, channels=64):

        super().__init__()

        self.spatial = SpatialBranch(channels)

        self.frequency = FrequencyBranch(channels)

    def forward(self, x):

        spatial_features = self.spatial(x)

        frequency_features = self.frequency(x)

        return spatial_features, frequency_features


# ============================================================
# 10. Multi-Scale Feature Fusion
# ============================================================

class MultiScaleFeatureFusion(nn.Module):

    def __init__(self, channels=64):

        super().__init__()

        self.fusion_conv = nn.Sequential(

            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(
        self,
        spatial_features,
        frequency_features
    ):

        fused = torch.cat(
            [
                spatial_features,
                frequency_features
            ],
            dim=1
        )

        return self.fusion_conv(fused)


# ============================================================
# 11. Dense Layer
# ============================================================

class DenseLayer(nn.Module):

    def __init__(
        self,
        in_channels,
        growth_rate=32
    ):

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            growth_rate,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.relu(
            self.conv(x)
        )

        return torch.cat(
            [x, out],
            dim=1
        )


# ============================================================
# 12. Residual Dense Block
# ============================================================

class ResidualDenseBlock(nn.Module):

    def __init__(
        self,
        channels=64,
        growth_rate=32,
        num_layers=4
    ):

        super().__init__()

        layers = []

        current_channels = channels

        for _ in range(num_layers):

            layers.append(
                DenseLayer(
                    current_channels,
                    growth_rate
                )
            )

            current_channels += growth_rate

        self.layers = nn.Sequential(*layers)

        self.lff = nn.Conv2d(
            current_channels,
            channels,
            kernel_size=1
        )

    def forward(self, x):

        out = self.layers(x)

        out = self.lff(out)

        return out + x


# ============================================================
# 13. Residual Dense Reconstruction
# ============================================================

class ResidualDenseReconstruction(nn.Module):

    def __init__(self, channels=64):

        super().__init__()

        self.rdbs = nn.Sequential(

            ResidualDenseBlock(channels),
            ResidualDenseBlock(channels),
            ResidualDenseBlock(channels)
        )

    def forward(self, x):

        return self.rdbs(x)


# ============================================================
# 14. Channel Attention
# ============================================================

class ChannelAttention(nn.Module):

    def __init__(
        self,
        channels,
        reduction=16
    ):

        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(

            nn.Conv2d(
                channels,
                channels // reduction,
                1,
                bias=False
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels // reduction,
                channels,
                1,
                bias=False
            )
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.fc(
            self.avg_pool(x)
        )

        max_out = self.fc(
            self.max_pool(x)
        )

        attention = self.sigmoid(
            avg_out + max_out
        )

        return x * attention


# ============================================================
# 15. Spatial Attention
# ============================================================

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):

        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        max_out, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        attention = torch.cat(
            [avg_out, max_out],
            dim=1
        )

        attention = self.sigmoid(
            self.conv(attention)
        )

        return x * attention


# ============================================================
# 16. CBAM
# ============================================================

class CBAM(nn.Module):

    def __init__(
        self,
        channels,
        reduction=16
    ):

        super().__init__()

        self.channel_attention = ChannelAttention(
            channels,
            reduction
        )

        self.spatial_attention = SpatialAttention()

    def forward(self, x):

        x = self.channel_attention(x)

        x = self.spatial_attention(x)

        return x


# ============================================================
# 17. AMDR-Net
# ============================================================

class AMDRNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.head = nn.Sequential(

            nn.Conv2d(
                1,
                64,
                3,
                padding=1
            ),

            nn.ReLU(inplace=True)
        )

        self.dam = DegradationAnalysisModule()

        self.controller = (
            AdaptiveFeatureFusionController()
        )

        self.adaptive = AdaptiveFeatureFusion()

        self.extractor = (
            DualBranchFeatureExtractor()
        )

        self.fusion = MultiScaleFeatureFusion()

        self.reconstruction = (
            ResidualDenseReconstruction()
        )

        self.cbam = CBAM(64)

        self.upsample = nn.Sequential(

            nn.Conv2d(
                64,
                256,
                3,
                padding=1
            ),

            nn.PixelShuffle(2),

            nn.ReLU(inplace=True)
        )

        self.tail = nn.Conv2d(
            64,
            1,
            3,
            padding=1
        )

    def forward(self, x):

        features = self.head(x)

        degradation = self.dam(x)

        weights = self.controller(
            degradation
        )

        features = self.adaptive(
            features,
            weights
        )

        spatial, frequency = (
            self.extractor(features)
        )

        fused = self.fusion(
            spatial,
            frequency
        )

        fused = self.reconstruction(
            fused
        )

        fused = self.cbam(fused)

        fused = self.upsample(fused)

        output = self.tail(fused)

        return output


# ============================================================
# 18. Inference Function
# ============================================================

def run_inference(
    input_dir,
    output_dir,
    model_path,
    device
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Loading model
    # --------------------------------------------------------

    print()
    print("Loading AMDR-Net model...")

    model = AMDRNet().to(device)

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint
    )

    model.eval()

    print("Model loaded successfully.")


    # --------------------------------------------------------
    # Getting input files
    # --------------------------------------------------------

    files = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith(".npy")
    ])

    if len(files) == 0:

        raise RuntimeError(
            "No .npy files found in input directory."
        )

    print("=" * 70)
    print("AMDR-Net Evaluation")
    print("=" * 70)

    print("Input directory :", input_dir)
    print("Output directory:", output_dir)
    print("Model           :", model_path)
    print("Device          :", device)
    print("Input images    :", len(files))


    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    total_start = time.perf_counter()

    inference_times = []

    with torch.no_grad():

        for filename in files:

            input_path = os.path.join(
                input_dir,
                filename
            )

            # .npy output
            output_path = os.path.join(
                output_dir,
                filename
            )

            # .png output
            png_filename = (
                os.path.splitext(filename)[0]
                + ".png"
            )

            png_output_path = os.path.join(
                output_dir,
                png_filename
            )


            # ------------------------------------------------
            # Load input image
            # ------------------------------------------------

            image = np.load(
                input_path
            )

            image = np.clip(
                image,
                0,
                1
            )


            # ------------------------------------------------
            # Convert to tensor
            # ------------------------------------------------

            tensor = torch.tensor(
                image,
                dtype=torch.float32,
                device=device
            )

            tensor = (
                tensor
                .unsqueeze(0)
                .unsqueeze(0)
            )


            # ------------------------------------------------
            # GPU timing
            # ------------------------------------------------

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()


            # ------------------------------------------------
            # AMDR-Net inference
            # ------------------------------------------------

            restored = model(
                tensor
            )


            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = (
                time.perf_counter() - start
            )

            inference_times.append(
                elapsed
            )


            # ------------------------------------------------
            # Convert output to NumPy
            # ------------------------------------------------

            restored = (
                restored
                .squeeze()
                .cpu()
                .numpy()
            )

            restored = np.clip(
                restored,
                0,
                1
            )


            # =================================================
            # Save .npy output
            # =================================================

            np.save(
                output_path,
                restored
            )


            # =================================================
            # Save .png output
            # =================================================

            restored_uint8 = (
                restored * 255.0
            ).round().astype(
                np.uint8
            )

            Image.fromarray(
                restored_uint8,
                mode="L"
            ).save(
                png_output_path
            )


    # ========================================================
    # Total execution time
    # ========================================================

    total_time = (
        time.perf_counter()
        - total_start
    )


    # ========================================================
    # Statistics
    # ========================================================

    average_time = np.mean(
        inference_times
    )

    minimum_time = np.min(
        inference_times
    )

    maximum_time = np.max(
        inference_times
    )

    throughput = (
        len(files) / total_time
    )


    # ========================================================
    # Verify outputs
    # ========================================================

    npy_outputs = sorted([
        f for f in os.listdir(output_dir)
        if f.endswith(".npy")
    ])

    png_outputs = sorted([
        f for f in os.listdir(output_dir)
        if f.endswith(".png")
    ])


    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 70)
    print("AMDR-Net Evaluation Summary")
    print("=" * 70)

    print(
        f"Images processed       : {len(files)}"
    )

    print(
        f".npy outputs generated : {len(npy_outputs)}"
    )

    print(
        f".png outputs generated : {len(png_outputs)}"
    )

    print(
        f"Total inference time   : "
        f"{total_time:.4f} seconds"
    )

    print(
        f"Average inference     : "
        f"{average_time * 1000:.2f} ms/image"
    )

    print(
        f"Minimum inference     : "
        f"{minimum_time * 1000:.2f} ms/image"
    )

    print(
        f"Maximum inference     : "
        f"{maximum_time * 1000:.2f} ms/image"
    )

    print(
        f"Throughput            : "
        f"{throughput:.2f} images/second"
    )

    print()
    print("Output directory:")
    print(output_dir)

    print()
    print("First 5 .npy outputs:")
    print(npy_outputs[:5])

    print()
    print("First 5 .png outputs:")
    print(png_outputs[:5])


    # ========================================================
    # Final verification
    # ========================================================

    if (
        len(npy_outputs) == len(files)
        and
        len(png_outputs) == len(files)
    ):

        print()
        print(
            "SUCCESS: Evaluation completed successfully."
        )

        print(
            f"All {len(files)} images have both "
            ".npy and .png outputs."
        )

    else:

        print()
        print(
            "WARNING: Output count does not "
            "match input count."
        )

        print(
            f"Input images : {len(files)}"
        )

        print(
            f".npy outputs : {len(npy_outputs)}"
        )

        print(
            f".png outputs : {len(png_outputs)}"
        )


# ============================================================
# 19. Command-Line Interface
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "AMDR-Net standalone image restoration "
            "evaluation script. Generates both "
            ".npy and .png restored outputs."
        )
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing input .npy images."
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help=(
            "Directory where restored .npy and "
            ".png images will be saved."
        )
    )

    parser.add_argument(
        "--model",
        default="AMDRNet_Final.pth",
        help="Path to trained AMDR-Net weights."
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    run_inference(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model,
        device=device
    )


if __name__ == "__main__":

    main()
