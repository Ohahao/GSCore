# gscore_simulator/rasterizer.py (No JIT, PyTorch 벡터화 최종 버전)

import torch
import numpy as np
from tqdm import tqdm
from gscore_simulator.structures import RenderMetrics
import math

# Numba 관련 import는 모두 제거합니다.

def rasterize_tiles(
    sorted_gaussians_by_tile: dict,
    camera,
    config: dict,
    device # GPU 장치를 명시적으로 받음
):
    """
    VRU (Volume Rendering Unit) 시뮬레이션 함수 (PyTorch 벡터화 버전).
    모든 연산을 PyTorch 텐서로 GPU 상에서 직접 처리하여 성능을 극대화합니다.
    """
    
    H, W = camera.image_height, camera.image_width
    tile_size = config['tile_size']
    subtile_res = config['subtile_res']
    
    # 최종 이미지를 저장할 버퍼를 GPU 메모리에 생성
    image_buffer = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
    alpha_buffer = torch.zeros((H, W), dtype=torch.float32, device=device)
    
    metrics = RenderMetrics()
    # 성능 지표 계산은 복잡성을 줄이기 위해 일단 생략
    
    print("Running Vectorized VRU (PyTorch on GPU): Rasterizing tiles...")
    
    # 타일 루프는 순차적으로 실행
    for tile_id, chunks in tqdm(sorted_gaussians_by_tile.items(), desc="Rasterizing Tiles"):
        if not chunks:
            continue
            
        metrics.tiles_rendered += 1
        
        # --- 청크 및 가우시안 루프 (순차적 알파 블렌딩을 위해 유지) ---
        for chunk in chunks:
            for gaus in chunk:
                # gaus 객체의 데이터가 NumPy일 수 있으므로 PyTorch GPU 텐서로 변환
                gaus_mean = gaus.mean.to(device) if isinstance(gaus.mean, torch.Tensor) else torch.tensor(gaus.mean, dtype=torch.float32, device=device)
                gaus_cov = gaus.cov.to(device) if isinstance(gaus.cov, torch.Tensor) else torch.tensor(gaus.cov, dtype=torch.float32, device=device)
                gaus_color = gaus.color_precomp.to(device) if isinstance(gaus.color_precomp, torch.Tensor) else torch.tensor(gaus.color_precomp, dtype=torch.float32, device=device)
                obb_corners = gaus.obb_corners.to(device) if isinstance(gaus.obb_corners, torch.Tensor) else torch.tensor(gaus.obb_corners, dtype=torch.float32, device=device)

                # 1. 가우시안이 영향을 미치는 픽셀 영역(Bounding Box) 정의
                min_bound_orig = torch.floor(torch.min(obb_corners, dim=0).values).int()
                max_bound_orig = torch.ceil(torch.max(obb_corners, dim=0).values).int()
                
                # 화면 경계를 벗어나지 않도록 클리핑
                min_bound = torch.maximum(torch.tensor([0,0], device=device), min_bound_orig)
                max_bound = torch.minimum(torch.tensor([W,H], device=device), max_bound_orig)

                if max_bound[0] <= min_bound[0] or max_bound[1] <= min_bound[1]:
                    continue
                
                # 2. 해당 영역의 픽셀 좌표 그리드 생성
                px_range = torch.arange(min_bound[0], max_bound[0], device=device)
                py_range = torch.arange(min_bound[1], max_bound[1], device=device)
                # PyTorch의 meshgrid는 기본적으로 'ij' 인덱싱을 사용하므로 py_grid가 먼저 옴
                py_grid, px_grid = torch.meshgrid(py_range, px_range, indexing='ij')

                # 3. 벡터화된 알파 값 계산
                d_grid = torch.stack((px_grid, py_grid), dim=-1) - gaus_mean
                
                try:
                    cov_inv = torch.linalg.inv(gaus_cov)
                except torch.linalg.LinAlgError:
                    continue
                
                power = -0.5 * torch.einsum('hwi,ij,hwj->hw', d_grid, cov_inv, d_grid)
                alpha_map = gaus.opacity * torch.exp(power)
                
                # 4. 벡터화된 Subtile Skipping
                subtile_size = tile_size / subtile_res
                tile_tx = (min_bound[0] // tile_size) * tile_size
                tile_ty = (min_bound[1] // tile_size) * tile_size
                
                subtile_idx_x = ((px_grid - tile_tx) // subtile_size).long()
                subtile_idx_y = ((py_grid - tile_ty) // subtile_size).long()
                subtile_indices = (subtile_idx_y * subtile_res + subtile_idx_x)
                
                bitmap = gaus.subtile_bitmaps.get(tile_id, 0)
                
                skip_mask = ((bitmap >> subtile_indices) & 1) == 0
                alpha_map[skip_mask] = 0

                # 5. 벡터화된 알파 블렌딩
                alpha_slice = alpha_buffer[min_bound[1]:max_bound[1], min_bound[0]:max_bound[0]]
                
                alpha_map[alpha_slice > 0.99] = 0

                valid_alpha_mask = alpha_map > 1e-4
                if not torch.any(valid_alpha_mask):
                    continue

                T = alpha_map * (1.0 - alpha_slice)
                T_valid = T[valid_alpha_mask]
                
                img_slice = image_buffer[min_bound[1]:max_bound[1], min_bound[0]:max_bound[0]]
                # PyTorch에서는 unsqueeze(-1)을 사용하여 차원을 맞춥니다.
                img_slice[valid_alpha_mask] += T_valid.unsqueeze(-1) * gaus_color
                
                alpha_slice += T

    # 최종 결과를 CPU로 가져와 NumPy 배열로 변환하여 반환
    rendered_image = image_buffer.cpu().numpy()
    return rendered_image, metrics