# -*- coding: utf-8 -*-
"""
========================================================================================
BÀI 1: THAY ĐỔI ENCODER / BACKBONE SANG DINO-V3 TRONG MÔ HÌNH ĐA PHƯƠNG THỨC (DuST)
Đề tài: Phân loại rủi ro lái xe đa phương thức (Video + Chuỗi thời gian gia tốc 3 trục)
Dựa trên bài báo CVPR: "DuST: Dual Swin Transformer for Multi-modal Video and Time-Series Modeling"
========================================================================================

Thay đổi trọng tâm (Bài 1 trên bảng):
- Kiến trúc gốc (DuST baseline): Sử dụng Video Swin Transformer (3D Swin) để trích xuất đặc trưng video.
- Kiến trúc mới cập nhật (Bài 1): Thay thế backbone video bằng DINO-v3 (Meta AI FAIR 2025 Vision Foundation Model).
  + DINO-v3 tích hợp 2D Axial Rotary Position Embeddings (RoPE) và cơ chế Gram Anchoring.
  + Module gộp thời gian (Temporal Pooling / Attention) tổng hợp đặc trưng chuỗi khung hình.
  + Lớp chiếu đặc trưng (Projection Head) đưa về vector 1024 chiều chuẩn hóa để kết nối trực tiếp
    với 1D Swin Transformer (chuỗi thời gian gia tốc 3 trục XYZ).
  + Đầu phân loại (Classification Head) phân loại 4 lớp sự kiện an toàn giao thông:
      0: Crash (Tai nạn giao thông)
      1: Tire strike (Va chạm bánh xe / chệch làn)
      2: Near-crash (Suýt va chạm)
      3: Normal driving (Lái xe bình thường, an toàn)
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

# Import DINO-v3 Video Encoder từ thư mục models/
from models.dino_video_encoder import DinoV3VideoEncoder, DinoVideoEncoder

# Import mô hình 1D Swin Transformer từ thư mục models/
from models.SwinTransformer_cls import SwinTransformer_1D
from models.SwinTransformer_fusion import SwinTransformer_1D as SwinTransformer_1D_Fusion


# ======================================================================================
# 1. MÔ HÌNH HỢP NHẤT ĐA PHƯƠNG THỨC HOÀN CHỈNH (DuST VỚI BACKBONE DINO-V3)
# ======================================================================================
class DuST_DINOv3_Multimodal(nn.Module):
    """
    Toàn bộ hệ thống DuST hoàn chỉnh tích hợp:
    1. Nhánh Video: Backbone DINO-v3 -> Vector đặc trưng video (1024-dim)
    2. Nhánh Kinematics: 1D Swin Transformer -> Vector đặc trưng chuỗi thời gian gia tốc 3 trục
    3. Stagewise Fusion Head: Ghép nối đặc trưng và phân loại 4 lớp nguy cơ giao thông.
    """
    def __init__(
        self,
        num_classes: int = 4,
        kine_seq_len: int = 51,
        kine_channels: int = 3,
        dinov3_model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        temporal_pooling: str = "mean",
        freeze_video_backbone: bool = True
    ):
        super().__init__()
        # Nhánh 1: Video Backbone DINO-v3 (Thay thế Video Swin Transformer cũ)
        self.video_encoder = DinoV3VideoEncoder(
            model_name=dinov3_model_name,
            num_classes=num_classes,
            out_features=1024,
            temporal_pooling=temporal_pooling,
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
        Lan truyền xuôi (Forward pass) mô hình đa phương thức DuST:
        video_input: [B, C, T, H, W]
        kine_input:  [B, 3, 51] (3 trục gia tốc x, y, z trong 51 bước thời gian)
        """
        # Bước 1: Trích xuất đặc trưng video qua DINO-v3
        vid_features = self.video_encoder.get_vid_feature(video_input)  # [B, 1024]

        # Bước 2: Hợp nhất với nhánh chuỗi thời gian gia tốc 1D Swin Transformer
        output_logits = self.fusion_model(kine_input, vid_features)     # [B, 4]
        return output_logits


# ======================================================================================
# 2. HÀM CHẠY KIỂM THỬ VÀ TRÌNH DIỄN TOÀN DIỆN (DEMO RUNNER)
# ======================================================================================
def run_demo():
    print("=" * 80)
    print(" CHƯƠNG TRÌNH CHẠY THỬ MÔ HÌNH BÀI 1: DuST VỚI BACKBONE DINO-V3 ")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị thực thi: {device}")

    # Cấu hình kích thước mẫu dữ liệu mô phỏng theo chuẩn bài báo DuST (SHRP 2 NDS):
    batch_size = 2
    num_frames = 4       # Mẫu 4 frames cho demo chạy nhanh (khi train đầy đủ: 77 frames)
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
    dummy_labels = torch.tensor([0, 3]).to(device)  # Giả lập 2 nhãn: Crash và Normal

    print(f" + Kích thước Video Input : {list(dummy_video.shape)} [Batch, Channels, Frames, Height, Width]")
    print(f" + Kích thước Kinematic   : {list(dummy_kine.shape)} [Batch, Trục XYZ, Số bước thời gian]")
    print(f" + Nhãn thực tế (GroundTruth): {dummy_labels.tolist()} -> {[class_names[i] for i in dummy_labels]}")

    print("\n--- BƯỚC 2: KIỂM TRA NHÁNH BACKBONE DINO-V3 (VIDEO ENCODER) ---")
    t0 = time.time()
    video_encoder = DinoV3VideoEncoder(
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        num_classes=num_classes,
        out_features=1024,
        temporal_pooling="mean",
        freeze_backbone=True
    ).to(device)

    with torch.no_grad():
        vid_feature = video_encoder.get_vid_feature(dummy_video)
    t_vid = time.time() - t0
    print(f" [OK] Trích xuất vector đặc trưng Video DINO-v3 hoàn tất!")
    print(f"      -> Kích thước Video Feature : {list(vid_feature.shape)} (Chuẩn 1024 chiều của DuST)")
    print(f"      -> Thời gian xử lý          : {t_vid:.2f} giây")

    print("\n--- BƯỚC 3: KIỂM TRA MÔ HÌNH HỢP NHẤT TOÀN DIỆN (FULL MULTIMODAL DuST + DINO-V3) ---")
    full_model = DuST_DINOv3_Multimodal(
        num_classes=num_classes,
        kine_seq_len=kine_len,
        kine_channels=kine_chans,
        dinov3_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        freeze_video_backbone=True
    ).to(device)

    # Đếm số lượng tham số
    total_params = sum(p.numel() for p in full_model.parameters())
    trainable_params = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
    print(f" + Tổng số tham số mô hình   : {total_params:,}")
    print(f" + Số tham số cần huấn luyện : {trainable_params:,} (Backbone DINO-v3 đã đóng băng)")

    print("\n--- BƯỚC 4: SUY LUẬN (FORWARD PASS & DỰ ĐOÁN XÁC SUẤT) ---")
    full_model.eval()
    with torch.no_grad():
        logits = full_model(dummy_video, dummy_kine)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

    print(f" + Logits đầu ra     : {list(logits.shape)} -> \n{logits.cpu().numpy()}")
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
    print(" KẾT LUẬN: MÃ NGUỒN BÀI 1 ĐÃ CẬP NHẬT HOÀN TOÀN SANG DINO-V3 THEO ĐÚNG YÊU CẦU TRÊN BẢNG!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bài 1: DuST với backbone DINO-v3")
    parser.add_argument("--mode", type=str, default="demo", choices=["demo", "test"],
                        help="Chế độ chạy (mặc định: demo)")
    args = parser.parse_args()

    run_demo()
