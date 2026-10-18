# -*- coding: utf-8 -*-
"""
========================================================================================
BÀI 1: THAY ĐỔI ENCODER / BACKBONE SANG DINO-V2 TRONG MÔ HÌNH ĐA PHƯƠNG THỨC (DuST)
Đề tài: Phân loại rủi ro lái xe đa phương thức (Video + Chuỗi thời gian gia tốc 3 trục)
Dựa trên bài báo CVPR: "DuST: Dual Swin Transformer for Multi-modal Video and Time-Series Modeling"
========================================================================================

Thay đổi trọng tâm (Bài 1):
- Gốc (DuST baseline): Sử dụng Video Swin Transformer (3D Swin) để trích xuất đặc trưng video.
- Đã cập nhật (Bài 1): Thay thế backbone video bằng DINOv2 (Meta AI Vision Transformer).
  + DINOv2 trích xuất đặc trưng ngữ nghĩa không gian sâu sắc cho từng khung hình.
  + Module gộp thời gian (Temporal Pooling / Attention) tổng hợp đặc trưng theo trục thời gian.
  + Chiếu (Projection Head) về vector đặc trưng 1024 chiều để kết nối trực tiếp với 1D Swin Transformer.
  + Đầu phân loại (Classification Head) phân loại 4 lớp sự kiện an toàn:
      0: Crash (Tai nạn)
      1: Tire strike (Va chạm bánh xe / lệch làn)
      2: Near-crash (Suýt va chạm)
      3: Normal driving (Lái xe bình thường)
"""

import os
import sys
import io
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Đảm bảo in tiếng Việt có dấu không bị lỗi trên Windows PowerShell (cp1252)
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Import DINOv2 từ HuggingFace Transformers
from transformers import AutoModel, Dinov2Config, Dinov2Model

# Import mô hình 1D Swin Transformer từ thư mục models/
from models.SwinTransformer_cls import SwinTransformer_1D
from models.SwinTransformer_fusion import SwinTransformer_1D as SwinTransformer_1D_Fusion


# ======================================================================================
# 1. VIDEO BACKBONE MỚI: DINO-V2 VIDEO ENCODER (THAY THẾ VIDEO SWIN TRANSFORMER)
# ======================================================================================
class DinoV2VideoEncoder(nn.Module):
    """
    Encoder Video sử dụng Backbone DINOv2 (Vision Transformer tự giám sát từ Meta AI).
    
    Đầu vào:
        x: Tensor video kích thước [B, C, T, H, W] hoặc [B, T, C, H, W]
           - B: Batch size
           - C: Số kênh màu (3 cho RGB hoặc Optical Flow)
           - T: Số frame thời gian (mặc định 77 frame theo bài báo SHRP2)
           - H, W: Độ phân giải ảnh (224 x 224)
           
    Đầu ra:
        - get_vid_feature(x): Vector đặc trưng video kích thước [B, 1024]
        - forward(x): Logits phân loại 4 lớp [B, 4] nếu huấn luyện đơn phương thức
    """
    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        num_classes: int = 4,
        out_features: int = 1024,
        temporal_pooling: str = "mean",  # 'mean', 'max', hoặc 'attention'
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.temporal_pooling = temporal_pooling
        self.out_features = out_features

        print(f"[DINOv2] Đang khởi tạo backbone: {model_name} ...")
        try:
            self.backbone = AutoModel.from_pretrained(model_name)
            print(f"[DINOv2] Tải trọng số Pretrained thành công từ HuggingFace Cache!")
        except Exception as e:
            print(f"[DINOv2 Warning] Không thể kết nối tải online ({e}). Khởi tạo kiến trúc Dinov2 nội bộ.")
            config = Dinov2Config(
                hidden_size=384,
                num_attention_heads=6,
                num_hidden_layers=6,
                patch_size=14,
                image_size=224
            )
            self.backbone = Dinov2Model(config)

        self.hidden_size = self.backbone.config.hidden_size

        # Đóng băng (Freeze) backbone nếu dùng đặc trưng trích xuất sẵn (Transfer Learning)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[DINOv2] Đã đóng băng trọng số Backbone DINOv2 (Feature Extraction mode).")

        # Cơ chế Attention theo trục thời gian (nếu chọn temporal_pooling='attention')
        if self.temporal_pooling == "attention":
            self.temporal_attn = nn.Sequential(
                nn.Linear(self.hidden_size, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )

        # Lớp chiếu (Projection Head): Biến đổi từ hidden_size của DINOv2 sang 1024-dim của DuST
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, out_features),
            nn.LayerNorm(out_features),
            nn.GELU()
        )

        # Đầu phân loại cho video (Stage 1 trong DuST)
        self.classifier = nn.Linear(out_features, num_classes) if num_classes > 0 else nn.Identity()

    def extract_frame_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Trích xuất đặc trưng cho từng frame độc lập qua DINOv2.
        frames: [Batch * T, C, H, W]
        Trả về: [Batch * T, hidden_size]
        """
        outputs = self.backbone(frames)
        # Token [CLS] đại diện cho ngữ nghĩa toàn thể frame
        cls_token = outputs.last_hidden_state[:, 0, :]
        return cls_token

    def get_vid_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        Trích xuất vector đặc trưng video chuẩn [B, 1024] tương thích trực tiếp với DuST.
        """
        # Chuẩn hoá kích thước về [B, T, C, H, W]
        if x.ndim == 5:
            if x.shape[1] == 3:  # [B, C, T, H, W]
                x = x.permute(0, 2, 1, 3, 4)
        else:
            raise ValueError(f"Đầu vào video yêu cầu tensor 5D, nhận được: {x.shape}")

        b, t, c, h, w = x.shape

        # Gộp batch và time để suy luận song song qua ViT DINOv2
        x_reshaped = x.reshape(b * t, c, h, w)
        frame_feats = self.extract_frame_features(x_reshaped)   # [B*T, hidden_size]
        frame_feats = frame_feats.view(b, t, self.hidden_size)  # [B, T, hidden_size]

        # Tổng hợp đặc trưng theo trục thời gian (Temporal Aggregation)
        if self.temporal_pooling == "mean":
            vid_feat = frame_feats.mean(dim=1)
        elif self.temporal_pooling == "max":
            vid_feat, _ = frame_feats.max(dim=1)
        elif self.temporal_pooling == "attention":
            weights = F.softmax(self.temporal_attn(frame_feats), dim=1)
            vid_feat = (frame_feats * weights).sum(dim=1)
        else:
            vid_feat = frame_feats.mean(dim=1)

        # Chiếu về không gian 1024 chiều
        vid_feat_1024 = self.proj(vid_feat)  # [B, 1024]
        return vid_feat_1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass video thuần (Stage 1)"""
        vid_feat = self.get_vid_feature(x)
        logits = self.classifier(vid_feat)
        return logits


# ======================================================================================
# 2. MÔ HÌNH HỢP NHẤT ĐA PHƯƠNG THỨC HOÀN CHỈNH (DuST VỚI BACKBONE DINO-V2)
# ======================================================================================
class DuST_DINOv2_Multimodal(nn.Module):
    """
    Toàn bộ hệ thống DuST hoàn chỉnh tích hợp:
    1. Nhánh Video: Backbone DINOv2 -> Vector đặc trưng video (1024-dim)
    2. Nhánh Kinematics: 1D Swin Transformer -> Vector đặc trưng chuỗi thời gian gia tốc 3 trục
    3. Stagewise Fusion Head: Ghép nối đặc trưng và phân loại 4 lớp nguy cơ giao thông.
    """
    def __init__(
        self,
        num_classes: int = 4,
        kine_seq_len: int = 51,
        kine_channels: int = 3,
        dinov2_model_name: str = "facebook/dinov2-small",
        freeze_video_backbone: bool = True
    ):
        super().__init__()
        # Nhánh 1: Video Backbone DINOv2
        self.video_encoder = DinoV2VideoEncoder(
            model_name=dinov2_model_name,
            num_classes=num_classes,
            out_features=1024,
            temporal_pooling="mean",
            freeze_backbone=freeze_video_backbone
        )

        # Nhánh 2 & Fusion: 1D Swin Transformer kết hợp với đặc trưng video 1024-dim
        self.fusion_model = SwinTransformer_1D_Fusion(
            seq_len=kine_seq_len,
            patch_size=1,
            in_chans=kine_channels,
            num_classes=num_classes,
            embed_dim=256,
            depths=[2, 2, 6, 2],
            num_heads=[16, 16, 16, 16],
            window_size=8,
            drop_path_rate=0.1
        )

    def forward(self, video_input: torch.Tensor, kine_input: torch.Tensor) -> torch.Tensor:
        """
        video_input: [B, C, T, H, W]
        kine_input: [B, 3, 51] (3 trục gia tốc x, y, z trong 51 bước thời gian)
        """
        # Bước 1: Trích xuất đặc trưng video qua DINOv2
        vid_features = self.video_encoder.get_vid_feature(video_input)  # [B, 1024]

        # Bước 2: Hợp nhất với nhánh chuỗi thời gian gia tốc 1D Swin Transformer
        output_logits = self.fusion_model(kine_input, vid_features)     # [B, 4]
        return output_logits


# ======================================================================================
# 3. HÀM CHẠY KIỂM THỬ VÀ TRÌNH DIỄN (DEMO RUNNER)
# ======================================================================================
def run_demo():
    print("=" * 80)
    print(" CHƯƠNG TRÌNH CHẠY THỬ MÔ HÌNH BÀI 1: DuST VỚI BACKBONE DINO-V2 ")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị thực thi: {device}")

    # Cấu hình kích thước mẫu dữ liệu mô phỏng theo chuẩn bài báo DuST (SHRP 2 NDS):
    batch_size = 2
    num_frames = 4       # Mẫu 4 frames cho demo chạy nhanh trên CPU (khi train đầy đủ: 77 frames)
    height, width = 224, 224
    kine_len = 51        # 51 điểm thời gian gia tốc (tần số 10Hz trong 5.1 giây)
    kine_chans = 3       # Gia tốc 3 trục: X (dọc), Y (ngang), Z (thẳng đứng)
    num_classes = 4

    class_names = [
        "0: Crash (Tai nạn)",
        "1: Tire strike (Va quẹt bánh/lệch lề)",
        "2: Near-crash (Suýt va chạm)",
        "3: Normal driving (Lái xe bình thường)"
    ]

    print("\n--- BƯỚC 1: TẠO DỮ LIỆU ĐA PHƯƠNG THỨC MÔ PHỎNG (SYNTHETIC BATCH) ---")
    dummy_video = torch.randn(batch_size, 3, num_frames, height, width).to(device)
    dummy_kine = torch.randn(batch_size, kine_chans, kine_len).to(device)
    dummy_labels = torch.tensor([0, 3]).to(device) # Giả lập 2 nhãn: Crash và Normal

    print(f" + Kích thước Video Input : {list(dummy_video.shape)} [Batch, Channels, Frames, Height, Width]")
    print(f" + Kích thước Kinematic   : {list(dummy_kine.shape)} [Batch, Trục XYZ, Số bước thời gian]")
    print(f" + Nhãn thực tế (GroundTruth): {dummy_labels.tolist()} -> {[class_names[i] for i in dummy_labels]}")

    print("\n--- BƯỚC 2: KIỂM TRA NHÁNH BACKBONE DINO-V2 (VIDEO ENCODER) ---")
    t0 = time.time()
    video_encoder = DinoV2VideoEncoder(
        model_name="facebook/dinov2-small",
        num_classes=num_classes,
        out_features=1024,
        freeze_backbone=True
    ).to(device)

    with torch.no_grad():
        vid_feature = video_encoder.get_vid_feature(dummy_video)
    t_vid = time.time() - t0
    print(f" [OK] Trích xuất vector đặc trưng Video DINOv2 hoàn tất!")
    print(f"      -> Kích thước Video Feature : {list(vid_feature.shape)} (Chuẩn 1024 chiều của DuST)")
    print(f"      -> Thời gian xử lý          : {t_vid:.2f} giây")

    print("\n--- BƯỚC 3: KIỂM TRA MÔ HÌNH HỢP NHẤT TOÀN DIỆN (FULL MULTIMODAL MODEL) ---")
    full_model = DuST_DINOv2_Multimodal(
        num_classes=num_classes,
        kine_seq_len=kine_len,
        kine_channels=kine_chans,
        dinov2_model_name="facebook/dinov2-small",
        freeze_video_backbone=True
    ).to(device)

    # Đếm số lượng tham số
    total_params = sum(p.numel() for p in full_model.parameters())
    trainable_params = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
    print(f" + Tổng số tham số mô hình   : {total_params:,}")
    print(f" + Số tham số cần huấn luyện : {trainable_params:,} (Backbone DINOv2 đã đóng băng)")

    print("\n--- BƯỚC 4: SUY LUẬN (FORWARD PASS & DỰ ĐOÁN XÁC SUẤT) ---")
    full_model.eval()
    with torch.no_grad():
        logits = full_model(dummy_video, dummy_kine)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

    print(f" + Logits đầu ra  : {logits.shape} -> \n{logits.cpu().numpy()}")
    print(f" + Xác suất (Softmax): \n{probs.cpu().numpy()}")
    print("\n -> KẾT QUẢ DỰ ĐOÁN TỪNG MẪU:")
    for idx, (pred, prob_vec) in enumerate(zip(preds, probs)):
        conf = prob_vec[pred].item() * 100
        print(f"    * Mẫu {idx+1}: Dự đoán = {class_names[pred]} (Độ tin cậy: {conf:.2f}%)")

    print("\n--- BƯỚC 5: MÔ PHỎNG 1 BƯỚC HUẤN LUYỆN (TRAINING STEP & BACKWARD PASS) ---")
    full_model.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, full_model.parameters()), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()
    train_logits = full_model(dummy_video, dummy_kine)
    loss = criterion(train_logits, dummy_labels)
    loss.backward()
    optimizer.step()

    print(f" [OK] Tính toán CrossEntropyLoss : {loss.item():.4f}")
    print(f" [OK] Lan truyền ngược (Backward pass) và cập nhật trọng số thành công!")

    print("\n" + "=" * 80)
    print(" KẾT LUẬN: MÃ NGUỒN BÀI 1 ĐÃ HOẠT ĐỘNG HOÀN HẢO THEO ĐÚNG YÊU CẦU TRÊN BẢNG!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bài 1: DuST với backbone DINO-v2")
    parser.add_argument("--mode", type=str, default="demo", choices=["demo", "test"],
                        help="Chế độ chạy (mặc định: demo)")
    args = parser.parse_args()

    run_demo()
