# gscore_simulator/culling.py (Full GPU Acceleration 최종 버전)

import numpy as np
import torch
import time
from tqdm import tqdm
import sys
import numba
from numba import cuda

# 이 파일에서 직접 사용하는 구조체 및 유틸리티
from gaussian_splatting.utils.graphics_utils import fov2focal
from gaussian_splatting.utils.sh_utils import eval_sh

# 로컬 경로에 있는 다른 프로젝트 파일 (경로는 사용자 환경에 맞게 유지)
sys.path.append("/home/hyoh/shape-aware-GS-processor/")
from test_gpu import analytic_2x2_eigendecomp, shape_analysis 

# ======================================================================
# === GPU Device 함수: CUDA 커널 내부에서 호출될 헬퍼 함수들 ===
# ======================================================================

@cuda.jit(device=True)
def _check_sat_intersection_cuda(box1_corners, box2_corners, box1_axes):
    """
    SAT 교차 판정을 GPU에서 수행하는 device 함수 (NumPy 없음).
    box1은 OBB, box2는 AABB(타일)로 가정합니다.
    """
    # 축 1: OBB의 첫 번째 축
    axis_x, axis_y = box1_axes[0, 0], box1_axes[0, 1]
    min1, max1 = 1e9, -1e9
    for i in range(4):
        proj = box1_corners[i, 0] * axis_x + box1_corners[i, 1] * axis_y
        min1 = min(min1, proj)
        max1 = max(max1, proj)
    min2, max2 = 1e9, -1e9
    for i in range(4):
        proj = box2_corners[i, 0] * axis_x + box2_corners[i, 1] * axis_y
        min2 = min(min2, proj)
        max2 = max(max2, proj)
    if max1 < min2 or max2 < min1:
        return False

    # 축 2: OBB의 두 번째 축
    axis_x, axis_y = box1_axes[1, 0], box1_axes[1, 1]
    min1, max1 = 1e9, -1e9
    for i in range(4):
        proj = box1_corners[i, 0] * axis_x + box1_corners[i, 1] * axis_y
        min1 = min(min1, proj)
        max1 = max(max1, proj)
    min2, max2 = 1e9, -1e9
    for i in range(4):
        proj = box2_corners[i, 0] * axis_x + box2_corners[i, 1] * axis_y
        min2 = min(min2, proj)
        max2 = max(max2, proj)
    if max1 < min2 or max2 < min1:
        return False

    # 축 3: AABB(타일)의 x축 (1, 0)
    min1, max1 = box1_corners[0, 0], box1_corners[0, 0]
    for i in range(1, 4):
        min1 = min(min1, box1_corners[i, 0])
        max1 = max(max1, box1_corners[i, 0])
    # box2(타일)는 이미 축에 정렬되어 있으므로 min/max는 꼭짓점 좌표 그 자체임
    min2, max2 = box2_corners[0, 0], box2_corners[1, 0]
    if max1 < min2 or max2 < min1:
        return False

    # 축 4: AABB(타일)의 y축 (0, 1)
    min1, max1 = box1_corners[0, 1], box1_corners[0, 1]
    for i in range(1, 4):
        min1 = min(min1, box1_corners[i, 1])
        max1 = max(max1, box1_corners[i, 1])
    min2, max2 = box2_corners[0, 1], box2_corners[2, 1]
    if max1 < min2 or max2 < min1:
        return False
        
    return True

@cuda.jit(device=True)
def is_pixel_in_gaussian_cuda(px, py, gaus_center, gaus_cov2d):
    """픽셀 중심점이 가우시안 타원 내부에 있는지 검사하는 device 함수"""
    a, b, d = gaus_cov2d[0, 0], gaus_cov2d[0, 1], gaus_cov2d[1, 1]
    det = a * d - b * b
    if det < 1e-9:
        return False
    inv_det = 1.0 / det
    inv_a, inv_b, inv_d = d * inv_det, -b * inv_det, a * inv_det
    dx, dy = px - gaus_center[0], py - gaus_center[1]
    dist = inv_a * dx * dx + 2 * inv_b * dx * dy + inv_d * dy * dy
    return dist <= 9.0

# ======================================================================
# === 메인 CUDA 커널: G_List 생성을 위한 교차 테스트 및 비트맵 계산 ===
# ======================================================================

@cuda.jit
def generate_intersections_kernel(
    results_buffer, result_counter,
    use_obb, 
    tx_min_all, tx_max_all, ty_min_all, ty_max_all,
    p_view_all, screen_xy_all, cov2d_all, 
    obb_corners_all, obb_axes_all,
    w, h, tile_size, subtile_res
):
    g_id = cuda.grid(1)
    if g_id >= len(use_obb):
        return

    gaus_center = screen_xy_all[g_id]
    gaus_cov2d = cov2d_all[g_id]
    
    tx_start, tx_end = tx_min_all[g_id], tx_max_all[g_id]
    ty_start, ty_end = ty_min_all[g_id], ty_max_all[g_id]

    for tx in range(tx_start, tx_end + 1):
        for ty in range(ty_start, ty_end + 1):
            
            should_process_tile = True
            if use_obb[g_id]:
                obb_corners_i = obb_corners_all[g_id]
                obb_axes_i = obb_axes_all[g_id]
                
                tile_min_x, tile_min_y = float(tx * tile_size), float(ty * tile_size)
                tile_corners_i = cuda.local.array((4, 2), dtype=numba.float32)
                tile_corners_i[0,0], tile_corners_i[0,1] = tile_min_x, tile_min_y
                tile_corners_i[1,0], tile_corners_i[1,1] = tile_min_x + tile_size, tile_min_y
                tile_corners_i[2,0], tile_corners_i[2,1] = tile_min_x + tile_size, tile_min_y + tile_size
                tile_corners_i[3,0], tile_corners_i[3,1] = tile_min_x, tile_min_y + tile_size
                
                should_process_tile = _check_sat_intersection_cuda(obb_corners_i, tile_corners_i, obb_axes_i)

            if not should_process_tile:
                continue

            bitmap = 0
            subtile_size = tile_size / subtile_res
            for j in range(subtile_res * subtile_res):
                stx, sty = j % subtile_res, j // subtile_res
                px = float(tx * tile_size + (stx + 0.5) * subtile_size)
                py = float(ty * tile_size + (sty + 0.5) * subtile_size)
                
                if is_pixel_in_gaussian_cuda(px, py, gaus_center, gaus_cov2d):
                    bitmap |= (1 << j)

            if bitmap > 0:
                depth = p_view_all[g_id, 2]
                write_idx = cuda.atomic.add(result_counter, 0, 1)
                
                if write_idx < len(results_buffer):
                    results_buffer[write_idx, 0] = g_id
                    results_buffer[write_idx, 1] = tx
                    results_buffer[write_idx, 2] = ty
                    results_buffer[write_idx, 3] = depth
                    results_buffer[write_idx, 4] = bitmap

# ======================================================================
# === 기존 CPU 헬퍼 함수들 (PyTorch 기반) ===
# ======================================================================
# 이 함수들은 메인 `cull_and_convert` 함수에서 GPU 텐서를 준비하는 데 사용됩니다.
def _compute_cov3d_vectorized(scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    norm_rot = rotations / torch.norm(rotations, dim=1, keepdim=True)
    w, x, y, z = norm_rot[:, 0], norm_rot[:, 1], norm_rot[:, 2], norm_rot[:, 3]
    R = torch.stack([
        torch.stack([1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w], dim=-1),
        torch.stack([2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w], dim=-1),
        torch.stack([2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2], dim=-1)
    ], dim=1)
    S = torch.zeros((scales.shape[0], 3, 3), dtype=scales.dtype, device=scales.device)
    S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = scales[:, 0], scales[:, 1], scales[:, 2]
    M = torch.bmm(R, S)
    return torch.bmm(M, M.transpose(1, 2))

def ndc2Pix(p_proj, W, H):
    x_ndc, y_ndc = p_proj[:, 0], p_proj[:, 1]
    x_pix = ((x_ndc + 1) * W - 1) * 0.5
    y_pix = ((y_ndc + 1) * H - 1) * 0.5
    return x_pix, y_pix

def compute_intrinsics(camera):
    width, height = camera.image_width, camera.image_height 
    fx, fy = fov2focal(camera.FoVx, width), fov2focal(camera.FoVy, height)
    intrinsics = {"fx": fx, "fy": fy, "cx": width / 2, "cy": height / 2, "image_size": (width, height)}
    return intrinsics, width, height

def _get_obb_corners(center, axes, radii):
    radii_2d = radii.unsqueeze(1).expand(-1, 2)
    extents = axes * radii_2d.unsqueeze(-1)
    c = center.unsqueeze(1) 
    corners = torch.cat([
        c - extents[:, 0:1] - extents[:, 1:2], c + extents[:, 0:1] - extents[:, 1:2],
        c + extents[:, 0:1] + extents[:, 1:2], c - extents[:, 0:1] + extents[:, 1:2],
    ], dim=1)
    return corners

# ======================================================================
# === 메인 함수 (Full GPU 파이프라인) ===
# ======================================================================
def cull_and_convert(
    gaussian_model, 
    camera, 
    config: dict
):
    with torch.no_grad():
        device = config['device']
        
        # --- 1, 2단계: 모든 데이터는 PyTorch GPU 텐서로 계산하고 유지 ---
        xyz = gaussian_model.get_xyz.to(device)
        opacities, scales, rotations, shs = gaussian_model.get_opacity.to(device), gaussian_model.get_scaling.to(device), gaussian_model.get_rotation.to(device), gaussian_model.get_features.to(device)
        view_matrix, proj_matrix, cam_center = camera.world_view_transform.to(device), camera.full_proj_transform.to(device), camera.camera_center.to(device)
        
        print("[DEBUG] Total Gaussians loaded:", xyz.shape[0])
        print("Running Vectorized Pre-computation & Culling on GPU...")
        intr, w, h = compute_intrinsics(camera)
        
        N = xyz.shape[0]
        xyz_h = torch.cat([xyz, torch.ones(N,1,device=device)], dim=1)
        p_hom = xyz_h @ proj_matrix.T
        p_w = 1.0 / (p_hom[:,3] + 1e-7)
        p_proj = p_hom[:,:3] * p_w.unsqueeze(1)
        
        p_view_h = xyz_h @ view_matrix
        p_view = p_view_h[:, :3] / (p_view_h[:, 3:] + 1e-7)

        valid_mask = p_view[:, 2] > 0.2
        
        # 유효한 가우시안들의 원본 인덱스를 저장해두면 나중에 추적하기 용이함
        original_indices = torch.where(valid_mask)[0]
        num_culled = len(original_indices)
        if num_culled == 0: return {}, []

        xyz, opacities, scales, rotations, shs, p_view, p_proj = \
            xyz[valid_mask], opacities[valid_mask], scales[valid_mask], \
            rotations[valid_mask], shs[valid_mask], p_view[valid_mask], p_proj[valid_mask]
        
        print(f"Vectorized Culling Complete. {num_culled} Gaussians remain.")

        print("Running Vectorized Covariance Calculation on GPU...")
        cov3D = _compute_cov3d_vectorized(scales, rotations)
        fx, fy = fov2focal(camera.FoVx, w), fov2focal(camera.FoVy, h)
        J = torch.zeros((num_culled,3,3), device=device)
        inv_z = 1.0 / p_view[:,2]
        J[:,0,0], J[:,0,2] = fx * inv_z, -fx * p_view[:,0] * inv_z**2
        J[:,1,1], J[:,1,2] = fy * inv_z, -fy * p_view[:,1] * inv_z**2
        
        W = view_matrix[:3, :3]
        T = W @ J
        cov2d = T.transpose(1, 2) @ cov3D @ T
        cov2d = cov2d[:, :2, :2]
        cov2d[:, 0, 0] += 0.3
        cov2d[:, 1, 1] += 0.3
        
        dir_pp = xyz - cam_center.repeat(num_culled, 1)
        colors_precomp_t = torch.clamp_min(eval_sh(gaussian_model.active_sh_degree, shs.transpose(1, 2).view(-1, 3, (gaussian_model.max_sh_degree+1)**2), dir_pp / dir_pp.norm(dim=1,keepdim=True)) + 0.5, 0.0)

        # --- 3단계: GPU 커널 실행 ---
        print("\nRunning Intersection Test and Bitmap Generation on GPU...")
        
        tile_size = config['tile_size']
        subtile_res = config['subtile_res']
        
        screen_x, screen_y = ndc2Pix(p_proj, w, h)
        screen_xy = torch.stack([screen_x, screen_y], dim=1)
        
        radii, R1, R2, _ = shape_analysis(cov2d)
        _, _, e_min = analytic_2x2_eigendecomp(cov2d)
        e_max = torch.stack([-e_min[:,1], e_min[:,0]], dim=1)
        obb_axes = torch.stack([e_max, e_min], dim=2)
        obb_corners = _get_obb_corners(screen_xy, obb_axes, radii)
        
        mins, maxs = obb_corners.min(dim=1).values, obb_corners.max(dim=1).values
        tx_min = torch.div(mins[:,0], tile_size, rounding_mode='floor').int()
        tx_max = torch.div(maxs[:,0], tile_size, rounding_mode='floor').int()
        ty_min = torch.div(mins[:,1], tile_size, rounding_mode='floor').int()
        ty_max = torch.div(maxs[:,1], tile_size, rounding_mode='floor').int()
        
        use_obb = ((tx_max - tx_min + 1) * (ty_max - ty_min + 1) > 1) & ((R2 / (R1 + 1e-9)) > 2)
        use_obb_int = use_obb.to(torch.uint8)

        
        # --- GPU 커널 실행 준비 ---
        max_intersections = num_culled * 20 # 가우시안당 평균 20개 타일 교차 가정, 충분히 크게 설정
        results_buffer = torch.empty((max_intersections, 5), dtype=torch.float32, device=device)
        result_counter = torch.zeros(1, dtype=torch.int32, device=device)

        threads_per_block = 128
        blocks_per_grid = (num_culled + (threads_per_block - 1)) // threads_per_block

        print(f"Launching CUDA kernel with {blocks_per_grid} blocks and {threads_per_block} threads per block...")
        start_time_gpu = time.time()
        
        generate_intersections_kernel[blocks_per_grid, threads_per_block](
            results_buffer, result_counter,
            use_obb_int, 
            tx_min, tx_max, ty_min, ty_max,
            p_view, screen_xy, cov2d,
            obb_corners, obb_axes,
            w, h, tile_size, subtile_res
        )
        cuda.synchronize()
        end_time_gpu = time.time()
        num_results = result_counter.item()
        print(f"GPU kernel finished. Found {num_results} intersections in {end_time_gpu - start_time_gpu:.4f} seconds.")

        # --- 최종 결과 정리 ---
        # 1. GPU에서 계산된 유효한 결과만 가져옴
        valid_results = results_buffer[:num_results]

        # 2. 다음 단계(sorting)를 위해 G_list를 CPU에서 재구성
        print("Aggregating results on CPU...")
        G_list = [[] for _ in range(num_culled)]
        # 결과를 CPU로 한번에 가져와서 처리
        valid_results_cpu = valid_results.cpu().numpy()
        for i in range(num_results):
            g_id_culled, tx, ty, depth, bitmap = valid_results_cpu[i]
            # g_id는 culled된 리스트 내에서의 인덱스이므로 그대로 사용
            g_id_culled = int(g_id_culled)
            G_list[g_id_culled].append(((int(tx), int(ty)), int(bitmap), depth))

        # 3. sorting.py가 사용할 수 있도록 culled_gaussians 딕셔너리를 NumPy 배열로 생성
        culled_gaussians = {
            "mean": screen_xy.cpu().numpy(),
            "cov": cov2d.cpu().numpy(),
            "opacity": opacities[:, 0].cpu().numpy(),
            "colors_precomp": colors_precomp_t.cpu().numpy(),
            "obb_corners": obb_corners.cpu().numpy(),
            "obb_axes": obb_axes.cpu().numpy()
        }

        return culled_gaussians, G_list