"""
feature_extractor.py — ResNet-18 frame feature extraction.

Replicates the exact preprocessing pipeline used during training:
    1. PIL .convert("RGB")
    2. Resize to 224 × 224
    3. ToTensor  (scales [0, 255] → [0.0, 1.0])
    4. ImageNet normalisation  mean=[0.485, 0.456, 0.406]
                               std =[0.229, 0.224, 0.225]
    5. ResNet-18 (fc = Identity) → 512-dim feature vector
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


class FeatureExtractor:
    """
    Wraps a pretrained ResNet-18 (fc layer replaced with Identity) to convert
    a list of RGB frames into a (T, 512) feature tensor.

    Frames are processed in batches of BATCH_SIZE for GPU/memory efficiency.
    """

    BATCH_SIZE: int = 32

    def __init__(self, device: str = "cpu"):
        """
        Load pretrained ResNet-18 with fc = nn.Identity().

        Args:
            device: torch device string, e.g. 'cpu', 'cuda', 'mps'.
        """
        self.device = torch.device(device)

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        backbone.eval()
        self.backbone = backbone.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def extract(self, frames: list) -> torch.Tensor:
        """
        Extract a 512-dim ResNet feature for each frame.

        Args:
            frames: list of H × W × 3 numpy arrays in RGB order,
                    or PIL Image objects.

        Returns:
            (T, 512) float32 tensor on CPU.

        Raises:
            ValueError: if frames is empty.
            TypeError:  if a frame is not a numpy array or PIL Image.
        """
        if not frames:
            raise ValueError(
                "frames list is empty — nothing to extract features from."
            )

        num_batches = (len(frames) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        all_features: list[torch.Tensor] = []

        for batch_index, batch_start in enumerate(
            range(0, len(frames), self.BATCH_SIZE), start=1
        ):
            if num_batches > 1:
                print(f"  Extracting features: batch {batch_index}/{num_batches}")

            batch_frames = frames[batch_start : batch_start + self.BATCH_SIZE]
            tensors: list[torch.Tensor] = []

            for frame in batch_frames:
                if isinstance(frame, np.ndarray):
                    image = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
                elif isinstance(frame, Image.Image):
                    image = frame.convert("RGB")
                else:
                    raise TypeError(
                        f"Unsupported frame type: {type(frame)}. "
                        "Expected numpy array or PIL Image."
                    )
                tensors.append(self.transform(image))

            batch_tensor = torch.stack(tensors).to(self.device)   # (B, 3, 224, 224)
            features     = self.backbone(batch_tensor)             # (B, 512)
            all_features.append(features.cpu())

        return torch.cat(all_features, dim=0)   # (T, 512)
