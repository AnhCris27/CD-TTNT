import sys
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, Dinov2Config, Dinov2Model

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

class DinoVideoEncoder(nn.Module):
    """
    Video Encoder using Meta's DINOv2 backbone to replace Video Swin Transformer.
    
    This encoder processes video frames [B, C, T, H, W] or [B, T, C, H, W]
    using DINOv2 (ViT) to extract spatial-semantic features per frame,
    followed by temporal aggregation and projection to a 1024-dim feature vector
    (matching the DuST multimodal fusion head interface).
    """
    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        num_classes: int = 4,
        out_features: int = 1024,
        temporal_pooling: str = "mean",  # 'mean', 'max', or 'attention'
        freeze_backbone: bool = True,
        use_fallback_if_offline: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.temporal_pooling = temporal_pooling
        self.out_features = out_features

        # 1. Load DINOv2 Backbone
        try:
            self.backbone = AutoModel.from_pretrained(model_name)
        except Exception as e:
            if use_fallback_if_offline:
                print(f"[Warning] Could not load '{model_name}' online: {e}. Using local Dinov2Config.")
                config = Dinov2Config(
                    hidden_size=384,
                    num_attention_heads=6,
                    num_hidden_layers=6,
                    patch_size=14,
                    image_size=224
                )
                self.backbone = Dinov2Model(config)
            else:
                raise e

        self.hidden_size = self.backbone.config.hidden_size

        # Freeze backbone weights if specified (standard DINOv2 transfer learning)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 2. Temporal Aggregation Module
        if self.temporal_pooling == "attention":
            self.temporal_attn = nn.Sequential(
                nn.Linear(self.hidden_size, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )

        # 3. Projection Head (Projects DINOv2 hidden dim to target fusion dim 1024)
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, out_features),
            nn.LayerNorm(out_features),
            nn.GELU()
        )

        # 4. Optional Classification Head (for standalone video training - Stage 1)
        self.classifier = nn.Linear(out_features, num_classes) if num_classes > 0 else nn.Identity()

    def extract_frame_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract features for individual frames using DINOv2.
        frames: [Batch*T, C, H, W]
        returns: [Batch*T, hidden_size]
        """
        outputs = self.backbone(frames)
        # In DINOv2, index 0 is the [CLS] token representation
        cls_token = outputs.last_hidden_state[:, 0, :]
        return cls_token

    def get_vid_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract video representation matching the DuST 'get_vid_feature' API.
        x: [B, C, T, H, W] or [B, T, C, H, W]
        returns: [B, out_features] (e.g. 1024 dimensions)
        """
        # Ensure shape is [B, T, C, H, W]
        if x.ndim == 5:
            if x.shape[1] == 3:  # [B, C, T, H, W]
                x = x.permute(0, 2, 1, 3, 4)
        else:
            raise ValueError(f"Expected 5D video tensor [B, C, T, H, W] or [B, T, C, H, W], got shape {x.shape}")

        b, t, c, h, w = x.shape

        # Flatten batch and time dimensions to pass through DINOv2
        x_reshaped = x.reshape(b * t, c, h, w)
        frame_feats = self.extract_frame_features(x_reshaped)  # [B*T, hidden_size]
        frame_feats = frame_feats.view(b, t, self.hidden_size) # [B, T, hidden_size]

        # Temporal Pooling
        if self.temporal_pooling == "mean":
            vid_feat = frame_feats.mean(dim=1)  # [B, hidden_size]
        elif self.temporal_pooling == "max":
            vid_feat, _ = frame_feats.max(dim=1)  # [B, hidden_size]
        elif self.temporal_pooling == "attention":
            weights = F.softmax(self.temporal_attn(frame_feats), dim=1) # [B, T, 1]
            vid_feat = (frame_feats * weights).sum(dim=1) # [B, hidden_size]
        else:
            vid_feat = frame_feats.mean(dim=1)

        # Project to target feature dimension (e.g. 1024 for DuST fusion)
        vid_feat_projected = self.proj(vid_feat) # [B, out_features]
        return vid_feat_projected

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for standalone video classification.
        x: video tensor
        returns: class logits [B, num_classes]
        """
        vid_feat = self.get_vid_feature(x)
        logits = self.classifier(vid_feat)
        return logits


if __name__ == "__main__":
    print("[Test] Khởi chạy kiểm thử độc lập DinoVideoEncoder...")
    encoder = DinoVideoEncoder(model_name="facebook/dinov2-small", out_features=1024)
    dummy_video = torch.randn(2, 3, 4, 224, 224)
    feat = encoder.get_vid_feature(dummy_video)
    logits = encoder(dummy_video)
    print(f" -> Feature shape: {feat.shape} (Expected: [2, 1024])")
    print(f" -> Logits shape : {logits.shape} (Expected: [2, 4])")
    print("[Test OK] DinoVideoEncoder hoạt động hoàn hảo!")

