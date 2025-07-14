# gscore_simulator/rasterizer.py (GPU 가속 최종 버전)

import numpy as np
import torch
from tqdm import tqdm
from gscore_simulator.structures import RenderMetrics
import numba
from numba import cuda
import math
from gscore_simulator.structures import Gaussian2D

# ======================================================================
# === GPU에서 실행될 CUDA 커널 함수 ===
# ======================================================================

@cuda.jit
def rasterize_tile_kernel(
    # 입출력 버퍼 (GPU 메모리)
    image_buffer,         # (H, W, 3) 전체 이미지 버퍼
    alpha_buffer,         # (H, W) 전체 알파 버퍼
    
    # 현재 타일 정보
    tx, ty, tile_size,
    
    # 이 타일에 속한 가우시안들의 데이터 (GPU 메모리)
    gaus_means,           # (G, 2)
    gaus_inv_covs,        # (G, 2, 2)
    gaus_opacities,       # (G,)
    gaus_colors,          # (G, 3)
    gaus_bitmaps,         # (G,)
    subtile_res
):
    """
    GPU 스레드들이 타일 하나의 픽셀들을 병렬로 처리하는 CUDA 커널.
    각 스레드는 하나의 픽셀을 담당합니다.
    """
    # 1. 현재 스레드가 담당할 픽셀의 타일 내 좌표(local) 및 전체 이미지 좌표(global) 계산
    px_local, py_local = cuda.grid(2)

    tile_h, tile_w = image_buffer.shape[0], image_buffer.shape[1]
    
    px_global = tx * tile_size + px_local
    py_global = ty * tile_size + py_local

    # 스레드가 타일 영역을 벗어나면 아무것도 하지 않고 즉시 종료
    if px_local >= tile_size or py_local >= tile_size or px_global >= tile_w or py_global >= tile_h:
        return

    # 2. Volume Rendering을 위한 변수 초기화
    T = 1.0  # Transmittance
    pixel_color_r, pixel_color_g, pixel_color_b = 0.0, 0.0, 0.0
    num_gaussians = len(gaus_means)
    subtile_size = tile_size // subtile_res

    # 3. 이 픽셀에 대해 모든 가우시안을 순서대로(front-to-back) 처리
    for g in range(num_gaussians):
        # 3.1. Subtile Skipping
        stx = px_local // subtile_size
        sty = py_local // subtile_size
        subtile_idx = int(sty * subtile_res + stx)
        if not ((gaus_bitmaps[g] >> subtile_idx) & 1):
            continue

        # 3.2. 알파(Alpha) 값 계산 (Point-in-Ellipse)
        d_x = float(px_global) - gaus_means[g, 0]
        d_y = float(py_global) - gaus_means[g, 1]
        
        inv_cov = gaus_inv_covs[g]
        power = -0.5 * (inv_cov[0, 0]*d_x*d_x + inv_cov[1, 1]*d_y*d_y + 2*inv_cov[0, 1]*d_x*d_y)
        
        if power < -15.0 or power > 0:
            continue
        
        # cuda.exp는 numba.cuda에서 제공하는 GPU용 지수 함수
        alpha = gaus_opacities[g] * cuda.exp(power)
        
        # 3.3. 알파 블렌딩
        color_contribution = T * alpha
        pixel_color_r += color_contribution * gaus_colors[g, 0]
        pixel_color_g += color_contribution * gaus_colors[g, 1]
        pixel_color_b += color_contribution * gaus_colors[g, 2]

        T = T * (1.0 - alpha)

        # 3.4. 조기 종료
        if T < 1e-4:
            break
            
    # 4. 최종 계산된 색상과 알파 값을 전역 버퍼(GPU 메모리)에 기록
    image_buffer[py_global, px_global, 0] = pixel_color_r
    image_buffer[py_global, px_global, 1] = pixel_color_g
    image_buffer[py_global, px_global, 2] = pixel_color_b
    alpha_buffer[py_global, px_global] = 1.0 - T


def rasterize_tiles(
    sorted_gaussians_by_tile: dict,  # keys are integer tile IDs
    camera,
    config: dict,
    device  # GPU 장치를 명시적으로 받음
):
    """
    VRU 시뮬레이션 함수 (GPU 가속 최종 버전).
    각 타일에 대해 CUDA 커널을 실행하여 전체 이미지를 렌더링합니다.
    sorted_gaussians_by_tile의 키는 (tx, ty) 픽셀 시작 좌표 튜플,
    값은 Gaussian2D 객체의 청크 리스트입니다.
    """
    H, W       = camera.image_height, camera.image_width
    tile_size  = config['tile_size']
    subtile_res= config['subtile_res']

    # GPU에 결과 버퍼 생성
    image_buffer = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
    alpha_buffer = torch.zeros((H, W),    dtype=torch.float32, device=device)

    metrics = RenderMetrics()

    print("Running GPU-accelerated VRU: Launching CUDA kernels for each tile...")
    #print(f" sorted gaussians: {sorted_gaussians_by_tile}")
    for tile_idx, tiles_data in tqdm(sorted_gaussians_by_tile.items(), desc="Rasterizing Tiles"):
        chunks = tiles_data["chunks"]
        txty = tiles_data["txty"]
        tx, ty = txty
        if not chunks:
            continue

        # 1) 각 chunk에서 Gaussian2D 객체만 골라내기
        gaussians_in_tile = [gaus for chunk in chunks for gaus in chunk if isinstance(gaus, Gaussian2D)]
        #print(f"    Finishing get gaussians in chunk! {len(gaussians_in_tile)} gaussians")

        # 안전 장치: 처리할 가우시안이 없는 타일은 건너뜁니다.
        if not gaussians_in_tile:
            print(f"DEBUG: Tile {tile_idx} has no gaussians to rasterize. Skipping.")
            continue

        print(f"DEBUG: Processing Tile {tile_idx} with {len(gaussians_in_tile)} gaussians.")

        mean_list = []
        opacity_list = []
        color_list = []
        cov_list = []
        bitmap_list = []

        # 2. `gaussians_in_tile` 리스트를 단 한 번만 순회합니다.
        for g in gaussians_in_tile:
            # 각 가우시안의 속성을 해당하는 리스트에 추가합니다.
            mean_list.append(g.mean)
            opacity_list.append(g.opacity)
            color_list.append(g.color_precomp)
            cov_list.append(g.cov)
            bitmap_list.append(g.tiles[tile_idx]["bitmap"])

        # 3. 루프가 끝난 후, 모아진 리스트들을 한 번에 GPU 텐서로 변환합니다.
        # 이 방식이 여러 번 루프를 도는 것보다 훨씬 효율적입니다.

        gaus_means     = torch.stack(mean_list, dim=0).to(device)
        gaus_opacities = torch.tensor(opacity_list, device=device)
        gaus_colors    = torch.stack(color_list, dim=0).to(device)
        gaus_covs      = torch.stack(cov_list, dim=0).to(device)
        gaus_bitmaps   = torch.stack(bitmap_list, dim=0).to(device)

        #print(f"DEBUG:gaus_covs shape: {gaus_covs.shape}")
        # --- 요청하신 전체 디버깅 출력 코드 ---
        print("-------------------------------------------")
        print(f"DEBUG: Processing Tile {tile_idx} with {len(gaussians_in_tile)} gaussians.")
        print(f"DEBUG: gaus_means     device: {gaus_means.device}, shape: {gaus_means.shape}")
        print(f"DEBUG: gaus_opacities device: {gaus_opacities.device}, shape: {gaus_opacities.shape}")
        print(f"DEBUG: gaus_colors    device: {gaus_colors.device}, shape: {gaus_colors.shape}")
        print(f"DEBUG: gaus_covs      device: {gaus_covs.device}, shape: {gaus_covs.shape}")
        print(f"DEBUG: gaus_bitmaps   device: {gaus_bitmaps.device}, shape: {gaus_bitmaps.shape}")
        print("-------------------------------------------")
        # --- 여기까지 디버깅 코드 ---
 
        try:
            gaus_inv_covs = torch.linalg.inv(gaus_covs)
        except RuntimeError:
            continue

        # CUDA kernel launch parameters
        threads_per_block = (16, 16)
        blocks_x = (tile_size + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_y = (tile_size + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_x, blocks_y)

        gaus_means_numba     = cuda.as_cuda_array(gaus_means)
        gaus_inv_covs_numba  = cuda.as_cuda_array(gaus_inv_covs)
        gaus_opacities_numba = cuda.as_cuda_array(gaus_opacities)
        gaus_colors_numba    = cuda.as_cuda_array(gaus_colors)
        gaus_bitmaps_numba   = cuda.as_cuda_array(gaus_bitmaps)

        # image_buffer와 alpha_buffer도 PyTorch 텐서라면 변환해야 합니다.
        image_buffer_numba = cuda.as_cuda_array(image_buffer)
        alpha_buffer_numba = cuda.as_cuda_array(alpha_buffer)

        # 커널 호출
        rasterize_tile_kernel[blocks_per_grid, threads_per_block](
            image_buffer_numba, alpha_buffer_numba,
            tx, ty, tile_size,
            gaus_means_numba, gaus_inv_covs_numba, gaus_opacities_numba,
            gaus_colors_numba, gaus_bitmaps_numba, subtile_res
        )

    rendered_image = image_buffer.cpu().numpy()
    return rendered_image, metrics