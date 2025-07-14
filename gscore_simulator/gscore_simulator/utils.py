# gscore_simulator/utils.py (수정 버전)

import torch
import numpy as np
from torchmetrics.image import PeakSignalNoiseRatio, LearnedPerceptualImagePatchSimilarity

def calculate_psnr(img1, img2, device="cpu"):
    """두 이미지 간의 PSNR을 지정된 장치에서 계산합니다."""
    # 메트릭 계산 객체를 지정된 장치로 보냄
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    
       
    # img1 (rendered_image) 처리
    if isinstance(img1, np.ndarray):
        img1_t = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float()
    elif isinstance(img1, torch.Tensor):
        if img1.ndim == 3:
            img1_t = img1.float()
        elif img1.ndim == 4:
            img1_t = img1.float()
        else:
            raise ValueError("img1 shape must be (H, W, C) or (1, C, H, W)")
    else:
        raise TypeError("img1 must be np.ndarray or torch.Tensor")

    # img2 처리
    if isinstance(img2, np.ndarray):
        img2_t = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float()
    elif isinstance(img2, torch.Tensor):
        img2_t = img2.float()
    else:
        raise TypeError("img2 must be np.ndarray or torch.Tensor")
    
    # 디바이스로 옮기기 (중요!)
    img1_t = img1_t.to(device)
    img2_t = img2_t.to(device)
    
    return psnr_metric(img1_t, img2_t).item()

def calculate_lpips(img1, img2, device="cpu", net_type='alex'):
    """두 이미지 간의 LPIPS를 지정된 장치에서 계산합니다."""
    print("img1 range:", img1.min().item(), img1.max().item())
    print("img2 range:", img2.min().item(), img2.max().item())

    
    # 메트릭 계산 객체를 지정된 장치로 보냄
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type=net_type, normalize=True).to(device)
    
       
    # img1 (rendered_image) 처리
    if isinstance(img1, np.ndarray):
        img1_t = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float()
    elif isinstance(img1, torch.Tensor):
        if img1.ndim == 3:
            img1_t = img1.permute(2, 0, 1).unsqueeze(0).float()
            img1_t = img1.unsqueeze(0).float()
        elif img1.ndim == 4:
            img1_t = img1.float()
        else:
            raise ValueError("img1 shape must be (H, W, C) or (1, C, H, W)")
    else:
        raise TypeError("img1 must be np.ndarray or torch.Tensor")

    # img2 처리
    if isinstance(img2, np.ndarray):
        img2_t = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float()
    elif isinstance(img2, torch.Tensor):
        if img2.dim()== 3:
            img2_t = img2.unsqueeze(0).float()
        else:
            img2_t = img2.float()
    else:
        raise TypeError("img2 must be np.ndarray or torch.Tensor")
    
    # 디바이스로 옮기기 (중요!)
    img1_t = img1_t.to(device)
    img2_t = img2_t.to(device)
    
    return lpips_metric(img1_t, img2_t).item()

# ... (get_view_matrix, get_projection_matrix 함수는 변경 없음) ...
def get_view_matrix(camera):
    return camera.world_view_transform


def get_projection_matrix(camera):
    return camera.full_proj_transform

