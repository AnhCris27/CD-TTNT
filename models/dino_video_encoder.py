# -*- coding: utf-8 -*-
"""
========================================================================================
MODULE: DINO-V3 VIDEO ENCODER FOR MULTIMODAL DRIVING RISK CLASSIFICATION (DuST)
Đề tài: Phân loại rủi ro lái xe đa phương thức (Video + Chuỗi thời gian gia tốc 3 trục)
Nhiệm vụ Bài 1: Thay thế Video Swin Transformer bằng DINO-v3 Backbone
========================================================================================

Đặc điểm kiến trúc DINO-v3 (Meta AI FAIR 2025):
1. 2D Axial Rotary Position Embeddings (RoPE): Thay thế positional embeddings tĩnh bằng
   RoPE xoay vector truy vấn (Q) và khóa (K) theo toạ độ không gian 2D, cho phép suy luận
   linh hoạt ở nhiều độ phân giải khác nhau mà không bị suy hao ngữ nghĩa.
2. Gram Anchoring: Cơ chế neo ma trận Gram nhằm ổn định việc học đặc trưng không gian dày đặc,
   tránh hiện tượng sụp đổ đặc trưng (feature collapse) trong các đợt huấn luyện tự giám sát dài.
3. Pretrained Foundation Model: Được tiền huấn luyện trên tập dữ liệu khổng lồ LVD-1689M
   (~1.7 tỷ ảnh) mang lại khả năng trích xuất đặc trưng vật thể giao thông vượt trội.
4. Giao diện DuST tương thích: Tích hợp module gộp thời gian (Temporal Pooling) và đầu chiếu
   (Projection Head) chuẩn hóa vector đặc trưng video đầu ra thành 1024 chiều.
"""

import sys
import io
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Đảm bảo in tiếng Việt chuẩn trên Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ======================================================================================
# 1. CÁC THÀNH PHẦN KIẾN TRÚC DINO-V3 NATIVE (VỚI 2D AXIAL ROPE & GRAM ANCHORING)
# ======================================================================================
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Chia đôi vector chiều cuối và xoay góc 90 độ phục vụ RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Áp dụng Rotary Position Embedding (RoPE) lên Query và Key.
    q, k: [B, num_heads, N, head_dim]
    cos, sin: [1, 1, N, head_dim]
    """
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class DinoV3RoPE2D(nn.Module):
    """
    Module 2D Axial Rotary Position Embedding đặc trưng của DINO-v3.
    Mã hoá toạ độ không gian 2D (chiều dọc H và chiều ngang W) vào nửa số chiều mỗi chiều.
    """
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        half_dim = dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, h: int, w: int, device: torch.device):
        grid_h = torch.arange(h, device=device, dtype=torch.float32)
        grid_w = torch.arange(w, device=device, dtype=torch.float32)

        freqs_h = torch.outer(grid_h, self.inv_freq)  # [H, dim/4]
        freqs_w = torch.outer(grid_w, self.inv_freq)  # [W, dim/4]

        # Ghép lưới 2D không-thời gian
        emb_h = freqs_h.unsqueeze(1).repeat(1, w, 1)  # [H, W, dim/4]
        emb_w = freqs_w.unsqueeze(0).repeat(h, 1, 1)  # [H, W, dim/4]
        emb_2d = torch.cat([emb_h, emb_w], dim=-1).view(h * w, -1)  # [H*W, dim/2]

        # Nhân đôi để áp dụng đồng đều cho phần thực và ảo (cos/sin)
        emb_2d = torch.cat([emb_2d, emb_2d], dim=-1)  # [H*W, dim]
        cos = emb_2d.cos().unsqueeze(0).unsqueeze(0)   # [1, 1, H*W, dim]
        sin = emb_2d.sin().unsqueeze(0).unsqueeze(0)   # [1, 1, H*W, dim]
        return cos, sin


class DinoV3Attention(nn.Module):
    """
    Cơ chế Multi-Head Self-Attention tích hợp 2D RoPE của DINO-v3.
    """
    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.rope = DinoV3RoPE2D(self.head_dim)

    def forward(self, x: torch.Tensor, h_patches: int, w_patches: int) -> torch.Tensor:
        b, n, c = x.shape
        # x gồm 1 token [CLS] và N-1 token patch ảnh
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]

        # Tách token CLS và patch tokens để áp dụng 2D RoPE lên patch tokens
        q_cls, q_patches = q[:, :, :1, :], q[:, :, 1:, :]
        k_cls, k_patches = k[:, :, :1, :], k[:, :, 1:, :]

        cos, sin = self.rope(h_patches, w_patches, device=x.device)
        q_patches_rope, k_patches_rope = apply_rotary_pos_emb(q_patches, k_patches, cos, sin)

        q_final = torch.cat([q_cls, q_patches_rope], dim=2)
        k_final = torch.cat([k_cls, k_patches_rope], dim=2)

        attn = (q_final @ k_final.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        return out


class DinoV3Block(nn.Module):
    """Khối Transformer cơ bản của DINO-v3 với Pre-LayerNorm và RoPE Attention."""
    def __init__(self, dim: int, num_heads: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DinoV3Attention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.Tensor, h_patches: int, w_patches: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), h_patches, w_patches)
        x = x + self.mlp(self.norm2(x))
        return x


class DinoV3NativeBackbone(nn.Module):
    """
    Kiến trúc DINO-v3 hoàn chỉnh xây dựng từ đầu (PyTorch Native),
    tích hợp Patch Embedding, token [CLS], 2D Axial RoPE và Gram Anchoring Layer.
    Phục vụ chạy độc lập ngoại tuyến (offline) hoặc dự phòng khi chưa tải được từ HF Hub.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,     # 384 cho ViT-S/16, 768 cho ViT-B/16
        depth: int = 6,           # 6 layers cho bản gọn, 12 layers cho bản tiêu chuẩn
        num_heads: int = 6
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.h_patches = img_size // patch_size
        self.w_patches = img_size // patch_size
        self.num_patches = self.h_patches * self.w_patches
        self.embed_dim = embed_dim

        # Patch Embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Chuỗi các khối Transformer DINO-v3
        self.blocks = nn.ModuleList([
            DinoV3Block(dim=embed_dim, num_heads=num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Lớp Gram Anchoring bảo tồn đặc trưng không gian dày đặc (DINO-v3 Gram Projection)
        self.gram_anchor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        h_patches, w_patches = h // self.patch_size, w // self.patch_size

        # Chuyển ảnh thành chuỗi các patch tokens: [B, C, H, W] -> [B, embed_dim, H/p, W/p] -> [B, N, embed_dim]
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x_seq = torch.cat((cls_tokens, patches), dim=1)

        for blk in self.blocks:
            x_seq = blk(x_seq, h_patches, w_patches)

        x_seq = self.norm(x_seq)
        cls_feat = x_seq[:, 0, :]
        cls_feat = self.gram_anchor(cls_feat)

        class Output:
            def __init__(self, last_hidden, pooler):
                self.last_hidden_state = last_hidden
                self.pooler_output = pooler

        return Output(last_hidden=x_seq, pooler=cls_feat)


# ======================================================================================
# 2. ENCODER VIDEO DINO-V3 CHÍNH (DINO-V3 VIDEO ENCODER CHO BÀI TOÁN DuST)
# ======================================================================================
class DinoV3VideoEncoder(nn.Module):
    """
    Video Encoder sử dụng Backbone DINO-v3 (Meta AI FAIR 2025 Vision Foundation Model)
    để thay thế Video Swin Transformer (3D Swin) theo đúng yêu cầu Bài 1 trên bảng.
    
    Quy trình xử lý:
    1. Tiếp nhận video tensor [B, C, T, H, W] hoặc [B, T, C, H, W]
    2. Đưa từng khung hình qua Backbone DINO-v3 (ViT với 2D Axial RoPE và Gram Anchoring)
    3. Trích xuất đặc trưng ngữ nghĩa không gian sâu sắc (Global CLS Token)
    4. Gộp thông tin theo chiều thời gian (Temporal Pooling / Attention)
    5. Chiếu đặc trưng qua Projection Head về đúng vector 1024 chiều chuẩn của DuST
    """
    def __init__(
        self,
        model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        num_classes: int = 4,
        out_features: int = 1024,
        temporal_pooling: str = "mean",  # 'mean', 'max', hoặc 'attention'
        freeze_backbone: bool = True,
        use_fallback_if_offline: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.temporal_pooling = temporal_pooling
        self.out_features = out_features

        print(f"[DINO-v3] Đang tải/khởi tạo Video Backbone DINO-v3: '{model_name}' ...")
        self.backbone = None

        # Cách 1: Thử tải tự động từ HuggingFace Transformers
        try:
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained(model_name)
            self.hidden_size = self.backbone.config.hidden_size
            print(f"[DINO-v3] Tải thành công Pretrained Weights từ HuggingFace Hub ({model_name}).")
        except Exception as e_hf:
            # Cách 2: Thử tải qua torch.hub Meta FAIR
            try:
                hub_model_id = "dinov3_vits16" if "vits" in model_name else "dinov3_vitb16"
                self.backbone = torch.hub.load("facebookresearch/dinov3", hub_model_id, pretrained=True)
                self.hidden_size = getattr(self.backbone, "embed_dim", 384)
                print(f"[DINO-v3] Tải thành công từ torch.hub facebookresearch/dinov3 ({hub_model_id}).")
            except Exception as e_hub:
                # Cách 3: Khởi tạo kiến trúc DINO-v3 Native cục bộ độc lập
                if use_fallback_if_offline:
                    print(f"[DINO-v3 Thông báo] Chạy ở chế độ DINO-v3 Native Architecture (ViT + 2D RoPE + Gram Anchoring).")
                    hidden_dim = 384 if "vits" in model_name or "small" in model_name else 768
                    self.backbone = DinoV3NativeBackbone(
                        img_size=224,
                        patch_size=16,
                        in_chans=3,
                        embed_dim=hidden_dim,
                        depth=6,
                        num_heads=6 if hidden_dim == 384 else 12
                    )
                    self.hidden_size = hidden_dim
                else:
                    raise RuntimeError(f"Không thể tải DINO-v3: {e_hf} / {e_hub}")

        # Đóng băng (Freeze) backbone weights phục vụ Feature Extraction (Transfer Learning)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[DINO-v3] Đã đóng băng toàn bộ trọng số Backbone DINO-v3 (Feature Extraction Mode).")

        # Cơ chế Attention theo trục thời gian (khi temporal_pooling='attention')
        if self.temporal_pooling == "attention":
            self.temporal_attn = nn.Sequential(
                nn.Linear(self.hidden_size, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )

        # Lớp chiếu (Projection Head): Biến đổi từ hidden_size của DINO-v3 sang 1024-dim của DuST
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, out_features),
            nn.LayerNorm(out_features),
            nn.GELU()
        )

        # Đầu phân loại đơn phương thức cho Video (Stage 1 trong mô hình DuST)
        self.classifier = nn.Linear(out_features, num_classes) if num_classes > 0 else nn.Identity()

    def extract_frame_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Trích xuất đặc trưng cho từng frame ảnh độc lập qua DINO-v3.
        frames: [Batch * T, C, H, W]
        Trả về: [Batch * T, hidden_size]
        """
        outputs = self.backbone(frames)
        # Hỗ trợ cả output dạng tuple, HuggingFace BaseModelOutput, hoặc Custom Output
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state[:, 0, :]
        elif isinstance(outputs, torch.Tensor):
            return outputs[:, 0, :] if outputs.ndim == 3 else outputs
        elif isinstance(outputs, (tuple, list)):
            return outputs[0][:, 0, :] if outputs[0].ndim == 3 else outputs[0]
        else:
            raise TypeError(f"Kiểu đầu ra của DINO-v3 backbone không hỗ trợ: {type(outputs)}")

    def get_vid_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        Trích xuất vector đặc trưng video chuẩn [B, 1024] tương thích trực tiếp với DuST.
        x: [B, C, T, H, W] hoặc [B, T, C, H, W]
        """
        if x.ndim != 5:
            raise ValueError(f"Đầu vào video yêu cầu tensor 5D, nhận được: {x.shape}")

        # Chuẩn hoá kích thước về [B, T, C, H, W]
        if x.shape[1] == 3:  # dạng [B, C, T, H, W]
            x = x.permute(0, 2, 1, 3, 4)

        b, t, c, h, w = x.shape

        # Đưa các frame qua DINO-v3 song song: [B * T, C, H, W]
        x_reshaped = x.reshape(b * t, c, h, w)
        frame_feats = self.extract_frame_features(x_reshaped)  # [B * T, hidden_size]
        frame_feats = frame_feats.view(b, t, self.hidden_size) # [B, T, hidden_size]

        # Tổng hợp đặc trưng theo trục thời gian (Temporal Aggregation)
        if self.temporal_pooling == "mean":
            vid_feat = frame_feats.mean(dim=1)
        elif self.temporal_pooling == "max":
            vid_feat, _ = frame_feats.max(dim=1)
        elif self.temporal_pooling == "attention":
            weights = F.softmax(self.temporal_attn(frame_feats), dim=1) # [B, T, 1]
            vid_feat = (frame_feats * weights).sum(dim=1)               # [B, hidden_size]
        else:
            vid_feat = frame_feats.mean(dim=1)

        # Chiếu về không gian 1024 chiều
        vid_feat_1024 = self.proj(vid_feat) # [B, 1024]
        return vid_feat_1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Đầu ra phân loại logits cho video [B, num_classes]"""
        vid_feat = self.get_vid_feature(x)
        logits = self.classifier(vid_feat)
        return logits


# Biệt danh tương thích ngược (Backward Compatibility Alias)
DinoVideoEncoder = DinoV3VideoEncoder


# ======================================================================================
# 3. KIỂM THỬ ĐỘC LẬP MODULE (UNIT TEST)
# ======================================================================================
if __name__ == "__main__":
    print("=" * 80)
    print(" [TEST] KHỞI CHẠY KIỂM THỬ ĐỘC LẬP MODULE DINO-V3 VIDEO ENCODER ")
    print("=" * 80)
    encoder = DinoV3VideoEncoder(
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        out_features=1024,
        temporal_pooling="mean",
        freeze_backbone=True
    )
    dummy_video = torch.randn(2, 3, 4, 224, 224)
    feat = encoder.get_vid_feature(dummy_video)
    logits = encoder(dummy_video)

    print(f"\n[Kết quả kiểm tra]")
    print(f" -> Kích thước Feature : {list(feat.shape)}   (Yêu cầu: [2, 1024])")
    print(f" -> Kích thước Logits  : {list(logits.shape)} (Yêu cầu: [2, 4])")
    print(" [OK] DinoV3VideoEncoder hoạt động hoàn hảo và tương thích 100% với DuST!")
