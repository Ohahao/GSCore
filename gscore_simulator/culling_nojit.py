# gscore_simulator/culling.py

import numpy as np
import torch
from tqdm import tqdm
import sys, os
import math
import torch.nn.functional as F

from collections import defaultdict
#from gscore_simulator.structures import Gaussian2D
#from gscore_simulator.utils import get_view_matrix, get_projection_matrix
from gaussian_splatting.utils.graphics_utils import fov2focal
from gaussian_splatting.utils.sh_utils import eval_sh

sys.path.append("/home/hyoh/shape-aware-GS-processor/")

from test_gpu import analytic_2x2_eigendecomp
from preprocessing import in_frustum, compute_cov3d, project_3d_to_2d

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
torch.set_grad_enabled(False) 

def cull_and_convert(
    gaussian_model, 
    camera, 
    config: dict
):
    """
    CCU (Culling and Conversion Unit) 시뮬레이션 함수 (벡터화 버전).
    모든 가우시안에 대한 수학 연산을 numpy 배열 연산으로 한 번에 처리하여 성능을 극대화합니다.
    """
    with torch.no_grad():
        device = config['device']
        xyz       = gaussian_model.get_xyz.to(device)       # (N,3)
        opacities = gaussian_model.get_opacity.to(device)   # (N,1)
        scales    = gaussian_model.get_scaling.to(device)   # (N,3)
        rotations = gaussian_model.get_rotation.to(device)  # (N,4)
        shs       = gaussian_model.get_features.to(device)  # (N,F)

        view_matrix = camera.world_view_transform.to(device)  # (4,4) or (3,4)
        proj_matrix = camera.full_proj_transform.to(device)  # (4,4)
        cam_center = camera.camera_center.to(device)
    
        print("[DEBUG] Total Gaussians loaded:", xyz.shape[0])
        print("Running Vectorized Pre-computation & Culling...")
        
        original_indices = torch.arange(xyz.shape[0], device=device)
        intrinsics, w, h = compute_intrinsics(camera)  # 정수
        modifier = 1
        rotations_normalized_t = F.normalize(rotations)
        cov3D = compute_cov3d(scales, modifier, rotations_normalized_t)
        
        # 2) frustum culling 
        N = xyz.shape[0]
        ones = torch.ones((N, 1), device=xyz.device)
        xyz_h = torch.cat([xyz, ones], dim=-1)  # (N, 4)

    
        # View transform → camera space
        if view_matrix.shape == (3, 4):
            R = view_matrix[:, :3]   # (3, 3)
            t = view_matrix[:, 3]    # (3,)
            p_view = xyz @ R.T + t  # (N, 3)
        else:  # (4, 4)
            p_view_h = xyz_h @ view_matrix.T  # (N, 4)
            p_view = p_view_h[:, :3] / (p_view_h[:, 3:] + 1e-7)  # (N, 3)

        # z < 0.2 frustum culling
        depths = p_view[:, 2]
        in_front_mask = depths > 0.2
        valid_mask = in_front_mask
      
        total = N
        dropped = int((~valid_mask).sum().item())
        if dropped:
            print(f"[preprocess] Dropped {dropped}/{total} gaussians with z ≤ {0.2}")

        # world view -> pixel space
        p = xyz_h[valid_mask] @ proj_matrix.T
        ndc    = p[:, :3] / p[:, 3:4] 
        # 3) frustum culling with ndc
        in_ndc_mask = (
            (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0) &
            (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0) &
            (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0)  # OpenGL 기준, DirectX면 0~1로 변경
        )
        total = int((valid_mask).sum().item())
        dropped = int((~in_ndc_mask).sum().item())
        print(f"[preprocess] Dropped {dropped}/{total} gaussians with ndc frustum culling")

        final_mask = torch.zeros_like(valid_mask)
        final_mask[valid_mask] = in_ndc_mask

        '''
        [DEBUG]
        # 4) 개수 출력
        num_total = ndc.shape[0]
        num_outside = (~in_ndc_mask).sum().item()
        num_inside = in_ndc_mask.sum().item()

        print(f"[DEBUG] NDC total points: {num_total}")
        print(f"[DEBUG] NDC inside [-1,1]: {num_inside}")
        print(f"[DEBUG] NDC outside [-1,1]: {num_outside}")
        '''
        # 5) 필터링된 NDC만 남김
        ndc = ndc[in_ndc_mask]
        p = p[in_ndc_mask]
                      
        screen_u = ((ndc[:,0] + 1.0) * w - 1.0) * 0.5
        screen_v = ((ndc[:,1] + 1.0) * h - 1.0) * 0.5
   
        
        '''
        # 1) frustum culling 
        N = xyz.shape[0]
        xyz_h = torch.cat([xyz, torch.ones(N,1,device=device)], dim=1)  # (N,4)
        # Projection -> clip space
        p_hom = xyz_h @ proj_matrix.T                                 # (N,4)
        p_w   = 1.0 / (p_hom[:,3] + 1e-7)
        p_proj = p_hom[:,:3] * p_w.unsqueeze(1)                        # (N,3)
    
        # View transform → camera space
        if view_matrix.shape == (3, 4):
            R = view_matrix[:, :3]   # (3, 3)
            t = view_matrix[:, 3]    # (3,)
            p_view = xyz @ R.T + t  # (N, 3)
        else:  # (4, 4)
            p_view_h = xyz_h @ view_matrix  # (N, 4)
            p_view = p_view_h[:, :3] / (p_view_h[:, 3:] + 1e-7)  # (N, 3)
    
        
        # z < 0.2 culling
        in_front_mask = p_view[:, 2] > 0.2
        valid_mask = in_front_mask 
        
        #Frustum Culling: NDC 공간에서 경계 벗어나는 가우시안 필터링
        #p_clip_h = xyz_h @ proj_matrix.T
        #p_clip = p_clip_h[:, :3] / (p_clip_h[:, 3, np.newaxis] + 1e-7)
        #in_frustum_mask = (np.abs(p_clip) <= 1.0).all(axis=1)
        '''

        # 유효한 가우시안 데이터만 남김
        xyz, opacities, scales, rotations, shs, p_view, valid_original_ids, cov3D = \
            xyz[final_mask], opacities[final_mask], scales[final_mask], \
            rotations[final_mask], shs[final_mask], p_view[final_mask], \
            original_indices[final_mask], cov3D[final_mask]
        
        screen_coords = torch.stack([screen_u, screen_v], dim=-1)

        print("[DEBUG] screen_u stats:", screen_u.min().item(), screen_u.max().item(), screen_u.shape)
        print("[DEBUG] screen_v stats:", screen_v.min().item(), screen_v.max().item(), screen_v.shape)

        num_culled = xyz.shape[0]
        if num_culled == 0:
            return []
            
        print(f"Vectorized Culling Complete. {num_culled} Gaussians remain.")
    
        # 2) feature computation
        # 2-1) projecting 3D mean, cov
        print("Running Vectorized Covariance Calculation...")
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]
        # 카메라 공간 좌표(t_x, t_y, t_z) -> 픽셀 좌표
        x = p_view[:, 0]
        y = p_view[:, 1]
        z = p_view[:, 2]
        center2d_t = torch.stack([
            fx * x / z + cx,
            fy * y / z + cy
        ], dim=1)
        cov2d_t = project_3d_to_2d(cov3D, p_view, camera, intrinsics, device, valid_original_ids)


        '''
        cov3D = _compute_cov3d_vectorized(scales, rotations) # (N, 3, 3) 
        fx = fov2focal(camera.FoVx, camera.image_width)
        fy = fov2focal(camera.FoVy, camera.image_height)
        J = torch.zeros((num_culled,3,3), device=device)
        inv_z = 1.0 / p_view[:,2]
        J[:,0,0] = fx * inv_z
        J[:,0,2] = -fx * p_view[:,0] * inv_z**2
        J[:,1,1] = fy * inv_z
        J[:,1,2] = -fy * p_view[:,1] * inv_z**2
    
    
        # 2-2) compute 2D cov
        W = view_matrix[:3, :3]
        T = W @ J
        cov2d = T.transpose(1, 2) @ cov3D @ T
        cov2d = cov2d[:, :2, :2]
        cov2d[:, 0, 0] += 0.3
        cov2d[:, 1, 1] += 0.3
        cov2d_t = cov2d.to(device)
        '''

        # 2-3) precompute color
        shs_view = shs.transpose(1, 2).view(-1, 3, (gaussian_model.max_sh_degree+1)**2)
        dir_pp = xyz - cam_center.repeat(shs.shape[0],1)                   
        dir_pp_norm = dir_pp / dir_pp.norm(dim=1,keepdim=True) 
        dir_pp_norm = dir_pp_norm.to(device)
        sh2rgb_t = eval_sh(gaussian_model.active_sh_degree, shs_view, dir_pp_norm)  # (N,3) Tensor
        colors_precomp_t = torch.clamp_min(sh2rgb_t + 0.5, 0.0)            # (N,3)
        colors_precomp_t = torch.clamp(colors_precomp_t, 0.0, 1.0)
        
        
        # 2-4) culled gaussians list(culling 결과와 함께 G_List에 저장)
        means_valid       = center2d_t     
        covs_valid        = cov2d_t         
        opacities_valid   = opacities
        scales_valid      = scales
        rotations_valid   = rotations

        num_gaussians = xyz.shape[0]
        GID = torch.arange(num_gaussians, device=xyz.device) 
        
        ############################################
        ############################################

        # 3) AABB/OBB intersection test
        print("Running Final Logic (AABB/OBB intersection test, Bitmap generation)...")
        
        # 3-1) OOBB 계산
        tile_size   = config["tile_size"]
        subtile_res = config["subtile_res"]

        # subtile 크기 계산
        subtile_w = math.ceil(tile_size / subtile_res)
        subtile_h = math.ceil(tile_size / subtile_res)
        subtile_size = subtile_w

        # 각 타일의 시작 좌표(tx, ty) 계산
        xs = torch.arange(0, w, tile_size, device=device)
        ys = torch.arange(0, h, tile_size, device=device)
        xs = xs.clamp(max=w - tile_size)
        ys = ys.clamp(max=h - tile_size)
        
        tx, ty = torch.meshgrid(xs, ys, indexing="xy")
        tx = tx.flatten()
        ty = ty.flatten()
        
        tile_origins = torch.stack([tx, ty], dim=1)  # (T, 2)
        # 3-1) OBB 계산
        tile_corners   = _get_tile_corners(tx, ty, tile_size)

        # 공분산으로부터 최소/최대 고유벡터와 반지름 구하기
        _, _, e_min          = analytic_2x2_eigendecomp(cov2d_t)
        R1, R2    = shape_analysis(cov2d_t)

        e_max                = torch.stack([-e_min[:, 1], e_min[:, 0]], dim=1)
        eigenvectors         = torch.stack([e_max, e_min], dim=2)

        obb_axes             = eigenvectors
        obb_axes             = torch.nn.functional.normalize(obb_axes, dim=1) 
        obb_radii            = torch.stack([R1, R2], dim=1)  # (N, 2)
        obb_corners          = _get_obb_corners(screen_coords, obb_axes, obb_radii)
        
        # 3-2) AABB intersection test
        mins = obb_corners.min(dim=1).values  # (N,2)
        maxs = obb_corners.max(dim=1).values  # (N,2)
        tx_min = torch.div(mins[:,0], tile_size, rounding_mode='trunc')  #겹치는 tile x 시작 좌표 (N,)
        tx_max = torch.div(maxs[:,0], tile_size, rounding_mode='trunc')  #겹치는 tile x 끝 좌표 (N,)
        ty_min = torch.div(mins[:,1], tile_size, rounding_mode='trunc')  #겹치는 tile y 시작 좌표 (N,)
        ty_max = torch.div(maxs[:,1], tile_size, rounding_mode='trunc')  #겹치는 tile y 끝 좌표  (N,)
        #각 가우시안에 대해 aabb 만들고 겹치는 타일 시작 좌표까지 구함
        
        # 3-3) decide OBB usage: multiple tiles & aspect ratio
        multi_tile = (tx_max - tx_min + 1) * (ty_max - ty_min + 1) > 1
        aspect = (R1 / (R2 + 1e-9)) > 2
        use_obb   = multi_tile & aspect  # (N,)
        GID_obb   = GID[use_obb]
        GID_aabb  = GID[~use_obb]
        print(f"[DEBUG] # of multi_tile gaussians: {(multi_tile).sum().item()}")
        print(f"[DEBUG] # of high_aspect gaussians (R2/R1 > 2): {(aspect).sum().item()}")
        print(f"[DEBUG] # of gaussians using OBB: {(use_obb).sum().item()}")
        print(f"\n number of OBB test: {GID_obb.numel()}, number of AABB test: {GID_aabb.numel()}")

        # ======================================================================
        # <<< 로직 수정 시작: 최종 반환 값 생성 >>>
        # ======================================================================

        #aabb gaussian intersection values
        obb_corners_aabb_test = obb_corners[GID_aabb]
        gs_idx_aabb_test, tile_idx_aabb_test, bitmap_aabb_test, txty_aabb_test = calculate_bitmap_aabb(obb_corners_aabb_test, tile_origins, tile_size, subtile_size)
        #print(f"\n [DEBUG] aabb test gaussians: {gs_idx_aabb_test.shape}, number of intersected tiles(aabb): {tile_idx_aabb_test.shape}, bitmap(aabb): {bitmap_aabb_test.shape}")
        all_zero_mask = (bitmap_aabb_test.sum(dim=1) == 0)  # 각 row의 합이 0인지 검사
        num_all_zero = all_zero_mask.sum().item()
        print(f"[DEBUG] Number of Gaussian-Tile pairs with all-zero bitmap(aabb): {num_all_zero} / {bitmap_aabb_test.shape[0]}")


        #obb gaussian intersection values
        obb_corners_obb_test = obb_corners[GID_obb]
        screen_coords_obb_test = screen_coords[GID_obb]
        obb_axes_obb_test = obb_axes[GID_obb]
        radii_obb_test = obb_radii[GID_obb]
        
        ##############################
        ########### DEBUG ############
        # tile origin 범위 (global 좌표)
        tile_min = tile_origins.min(dim=0).values
        tile_max = (tile_origins + tile_size).max(dim=0).values

        # Gaussian 중심 좌표 범위
        center_min = screen_coords_obb_test.min(dim=0).values
        center_max = screen_coords_obb_test.max(dim=0).values

        print(f"[DEBUG] Tile origin range: x [{tile_min[0]:.1f}, {tile_max[0]:.1f}], y [{tile_min[1]:.1f}, {tile_max[1]:.1f}]")
        print(f"[DEBUG] Gaussian center range: x [{center_min[0]:.1f}, {center_max[0]:.1f}], y [{center_min[1]:.1f}, {center_max[1]:.1f}]")
        print("screen_coords_obb_test stats:")
        print("   shape:", screen_coords_obb_test.shape)
        print("   min:", screen_coords_obb_test.min(dim=0).values)
        print("   max:", screen_coords_obb_test.max(dim=0).values)


        radii_max = radii_obb_test.max().values  # 가장 큰 축
        radii_min = radii_obb_test.min().values  # 가장 작은 축

        if radii_obb_test.ndim == 1:
            # shape: [N]
            r_min = radii_obb_test.min().item()
            r_max = radii_obb_test.max().item()
            r_mean = radii_obb_test.mean().item()
            print(f"[DEBUG] radius: min={r_min:.3f}, max={r_max:.3f}, mean={r_mean:.3f}")
        elif radii_obb_test.ndim == 2:
            # shape: [N, 2]
            radii_max = radii_obb_test.max(dim=1).values
            radii_min = radii_obb_test.min(dim=1).values
            print(f"[DEBUG] max radius: min={radii_max.min().item():.3f}, max={radii_max.max().item():.3f}, mean={radii_max.mean().item():.3f}")
            print(f"[DEBUG] min radius: min={radii_min.min().item():.3f}, max={radii_min.max().item():.3f}, mean={radii_min.mean().item():.3f}")
        else:
            print("[ERROR] Unexpected radii shape:", radii_obb_test.shape)

        obb_min = obb_corners_obb_test.view(-1, 2).min(dim=0).values
        obb_max = obb_corners_obb_test.view(-1, 2).max(dim=0).values
        print(f"[DEBUG] OBB corner global range: x [{obb_min[0]:.1f}, {obb_max[0]:.1f}], y [{obb_min[1]:.1f}, {obb_max[1]:.1f}]")

        ##############################
        ########### DEBUG ############
    
        obb_mask = check_sat_intersection(obb_corners_obb_test, obb_axes_obb_test, tile_corners, 8)      #(gaussians, tiles)
        print("[DEBUG] SAT mask true count:", obb_mask.sum().item(), "/", obb_mask.numel())
    
        gs_idx_obb_test, tile_idx_obb_test, bitmap_obb_test, txty_obb_test = calculate_bitmap_obb(obb_mask, screen_coords_obb_test, 
                                                                               obb_axes_obb_test, radii_obb_test, tile_origins,       
                                                                               tile_size, subtile_size)
       
        print(f"\n [DEBUG] obb test gaussian: {gs_idx_obb_test.shape}, intersected tiles(obb): {tile_idx_obb_test.shape}, bitmap(obb): {bitmap_obb_test.shape}")
        idx = torch.where((bitmap_obb_test.sum(dim=1) == 0))[0][:2]
        '''
        for i in idx:
            print(f"\n=== DEBUG Gaussian {i.item()} ===")
            print("  center:", screen_coords_obb_test[i].cpu().numpy())
            print("  radii:", radii_obb_test[i].cpu().numpy())
            print("  obb_axes:", obb_axes_obb_test[i].cpu().numpy())
            print("  obb_corners:", obb_corners_obb_test[i].cpu().numpy())
        '''
        # bitmap에서 모든 요소가 0인 row 수 계산
        all_zero_mask = (bitmap_obb_test.sum(dim=1) == 0)  # (P,)
        num_all_zero = all_zero_mask.sum().item()

        print(f"[DEBUG] Number of Gaussian-Tile pairs with all-zero bitmap: {num_all_zero} / {bitmap_obb_test.shape[0]}")
        # 5. GSU 입력을 위한 최종 데이터 구조 생성
        print("Reformatting data for GSU...")
        # 5-1. 가우시안 ID를 기준으로 교차 타일 목록을 재구성합니다.
        depths = p_view[:, 2]
        depths_aabb = depths[GID_aabb][gs_idx_aabb_test]
        depths_obb  = depths[GID_obb][gs_idx_obb_test] 

        # 5) concatenate results
        gauss_idx_mapped_aabb = GID_aabb[gs_idx_aabb_test]
        gauss_idx_mapped_obb  = GID_obb[gs_idx_obb_test]
        gauss_idx = torch.cat([gauss_idx_mapped_aabb, gauss_idx_mapped_obb], dim=0)
        tile_idx  = torch.cat([tile_idx_aabb_test,  tile_idx_obb_test ], dim=0)
        bitmap    = torch.cat([bitmap_aabb_test,   bitmap_obb_test    ], dim=0)  # (P_total, S)
        depth     = torch.cat([depths_aabb,       depths_obb         ], dim=0)  # (P_total,)
        txty      = torch.cat([txty_aabb_test,       txty_obb_test         ], dim=0) 

        print(f"\n[DEBUG] Combined → total pairs: {gauss_idx.shape[0]}, bitmap shape: {bitmap.shape}, depth shape: {depth.shape}")
        G_List = {  
                    "mean":        means_valid,
                    "cov":         covs_valid,
                    "opacity":     opacities_valid,
                    "scales":      scales_valid,
                    "rotations":   rotations_valid,
                    "colors_precomp":  colors_precomp_t,
                    "obb_corners":    obb_corners,    
                    "obb_axes":       obb_axes,      
                    "gauss_idx": gauss_idx,
                    "tile_idx":  tile_idx,
                    "bitmap":    bitmap,
                    "depth":     depth,
                    "txty" :     txty
                }

        return G_List
    
    
# --- Vectorized Helper Functions ---


def _compute_cov3d_vectorized(scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """3D 공분산 행렬을 모든 가우시안에 대해 일괄적으로 계산합니다.
    Args:
        scales: (N, 3) - x, y, z scaling
        rotations: (N, 4) - w, x, y, z quaternion
    Returns:
        cov3d: (N, 3, 3) - 3D covariance matrix per Gaussian
    """
    # 정규화된 쿼터니언 (w, x, y, z)
    norm_rot = rotations / torch.norm(rotations, dim=1, keepdim=True)
    w, x, y, z = norm_rot[:, 0], norm_rot[:, 1], norm_rot[:, 2], norm_rot[:, 3]

    # 회전 행렬 R: (N, 3, 3)
    R = torch.stack([
        torch.stack([1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w], dim=-1),
        torch.stack([2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w], dim=-1),
        torch.stack([2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2], dim=-1)
    ], dim=1)  # (N, 3, 3)

    # 스케일링 행렬 S: (N, 3, 3)
    S = torch.zeros((scales.shape[0], 3, 3), dtype=scales.dtype, device=scales.device)
    S[:, 0, 0] = scales[:, 0]
    S[:, 1, 1] = scales[:, 1]
    S[:, 2, 2] = scales[:, 2]

    # M = R @ S
    M = torch.bmm(R, S)

    # Σ = M @ M^T
    cov3d = torch.bmm(M, M.transpose(1, 2))  # (N, 3, 3)

    return cov3d
    

# --- Helper Functions ---
def shape_analysis(cov2d_t):
    C_x = cov2d_t[:, 0, 0]
    C_y = cov2d_t[:, 0, 1]
    C_z = cov2d_t[:, 1, 1]
    
    #compute eigenvalue
    trace = (C_x + C_z)
    det = (C_x * C_z - C_y * C_y)
    det_inv = torch.zeros_like(det)
    nonzero = det != 0
    det_inv[nonzero] = 1.0 / det[nonzero]

    conic = torch.stack([
        C_z * det_inv,    # A = C_z / det
        -C_y * det_inv,   # B = -C_y / det
        C_x * det_inv     # C = C_x / det
    ], dim=1)           # shape (M, 3)
    
    mid  = 0.5 * (C_x + C_z)                   # shape (M,)
    disc = mid * mid - det                     # shape (M,)
    disc_clamped = torch.clamp(disc, min=0.1)  # shape (M,)
    sqrt_disc    = torch.sqrt(disc_clamped)    # shape (M,)
    
    lam1 = mid + sqrt_disc                     # largest eigenvalue (σ²)
    lam2 = mid - sqrt_disc                     # smallest eigenvalue (σ²)
    lam2 = torch.clamp(lam2, min=0)

    R1 = torch.ceil(0.45 * torch.sqrt(lam1))
    R2 = torch.ceil(0.45 * torch.sqrt(lam2))
    R1 = torch.clamp(R1, min=0.5, max=64)  #tile size 2배로 클램핑
    R2 = torch.clamp(R2, min=0.5, max=64)

    #N_total = cov2d_t.shape[0]
    #print(f"Spiky인 Gaussian 개수: {num_spiky}, 전체 gaussian 개수: {N_total}")
    
    return R1, R2

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    q: (N, 4) numpy array, where q = [w, x, y, z]
    returns: (N, 3, 3) rotation matrices
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    B = q.shape[0]
    R = np.zeros((B, 3, 3), dtype=np.float32)

    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x*y - z*w)
    R[:, 0, 2] = 2 * (x*z + y*w)

    R[:, 1, 0] = 2 * (x*y + z*w)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y*z - x*w)

    R[:, 2, 0] = 2 * (x*z - y*w)
    R[:, 2, 1] = 2 * (y*z + x*w)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

    return R


def ndc2Pix(p_proj, W, H):

    x_ndc = p_proj[:, 0]
    y_ndc = p_proj[:, 1]

    x_pix = (x_ndc + 1) * 0.5 * W
    y_pix = (1 - (y_ndc + 1) * 0.5) * H 

    return x_pix, y_pix


def compute_intrinsics(camera):

    width, height = camera.image_width, camera.image_height 
    fx = fov2focal(camera.FoVx, width)
    fy = fov2focal(camera.FoVy, height)
    cx = width / 2
    cy = height / 2
  
    intrinsics = {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "image_size": (width, height)
        }
        
    #for k, v in intrinsics.items():
    #  print(f"{k}: {v}")
            
    return intrinsics, width, height


def _quat_to_rotmat(q):
    """쿼터니언을 회전 행렬로 변환"""
    # q = w, x, y, z
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def _get_obb_corners(center, axes, radii):
    """OBB의 4개 꼭짓점 좌표 계산"""
    #radii_2d = radii.unsqueeze(1).expand(-1, 2) -> radii가 scalar인 경우 사용
    extents = axes * radii.unsqueeze(-1)  #(N, 2, 2)
    c = center.unsqueeze(1) #(N, 1, 2)

    corners = torch.cat([
        c - extents[:, 0:1] - extents[:, 1:2],  # (-r1·axis1 - r2·axis2)
        c + extents[:, 0:1] - extents[:, 1:2],  # (+r1·axis1 - r2·axis2)
        c + extents[:, 0:1] + extents[:, 1:2],  # (+r1·axis1 + r2·axis2)
        c - extents[:, 0:1] + extents[:, 1:2],  # (-r1·axis1 + r2·axis2)
    ], dim=1)  # => (N,4,2)

    return corners
    
def _get_tile_corners(tx, ty, tile_size, device=None):
    """타일의 4개 꼭짓점 좌표를 torch.Tensor로 반환"""

    min_x, min_y = tx, ty
    max_x, max_y = min_x + tile_size, min_y + tile_size
    # 3) (T,4,2) 형태로 쌓기
    #    각 행이 [ (x0,y0), (x1,y0), (x1,y1), (x0,y1) ]
    corners = torch.stack([
        torch.stack([min_x, min_y], dim=1),
        torch.stack([max_x, min_y], dim=1),
        torch.stack([max_x, max_y], dim=1),
        torch.stack([min_x, max_y], dim=1),
    ], dim=1)

    return corners  # (T, 4, 2)

def _get_subtile_corners(tx, ty, stx, sty, tile_size, subtile_res, device=None):
    """서브타일의 4개 꼭짓점 좌표를 torch.Tensor로 반환"""

    subtile_size = tile_size / subtile_res
    min_x = tx * tile_size + stx * subtile_size
    min_y = ty * tile_size + sty * subtile_size
    max_x, max_y = min_x + subtile_size, min_y + subtile_size

    corners = torch.tensor([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ], dtype=torch.float32, device=device)

    return corners  # (4, 2)


def check_sat_intersection(
    box1_corners: torch.Tensor,  # (M, 4, 2)
    box1_axes:    torch.Tensor,  # (M, 2, 2)
    box2_corners: torch.Tensor,   # (T, 4, 2)
    chunk_size: int = 512
    ) -> torch.Tensor:

    """
    for문 없이 SAT 교차 판정 (M×T → (M,T) boolean mask)
    """
    M = box1_corners.shape[0]
    T = box2_corners.shape[0]

    device = box1_corners.device
    dtype  = box1_corners.dtype

    box1_corners = box1_corners.to(device=device, dtype=dtype)
    box1_axes    = box1_axes.to(device=device, dtype=dtype)
    box2_corners = box2_corners.to(device=device, dtype=dtype)

    global_axes = torch.tensor([[1,0],[0,1]], device=device, dtype=dtype)
    intersects = torch.zeros(M, T, dtype=torch.bool, device=device)

    n_chunks = math.ceil(T / chunk_size)
    print(f"    chunk size(한번에 처리하는 타일 수): {chunk_size}")
    for chunk_id in tqdm(range(n_chunks), desc="SAT chunks"):
        start = chunk_id * chunk_size
        end   = min(start + chunk_size, T)
        sub_T = end - start

        # (M,sub_T,4,2)
        b1_c = box1_corners.unsqueeze(1).expand(-1, sub_T, -1, -1)
        b2_c = box2_corners[start:end].unsqueeze(0).expand(M, -1, -1, -1)

        # axes (M,sub_T,2,2)
        gauss_ax = box1_axes.unsqueeze(1).expand(-1, sub_T, -1, -1)
        tile_ax  = global_axes.unsqueeze(0).unsqueeze(0).expand(M, sub_T, -1, -1)

        axes   = torch.cat([gauss_ax, tile_ax], dim=2)       # (M,sub_T,4,2)
        axes_t = axes.permute(0,1,3,2)                       # (M,sub_T,2,4)

        proj1 = torch.matmul(b1_c, axes_t)
        proj2 = torch.matmul(b2_c, axes_t)

        p1_min = proj1.min(dim=2).values; p1_max = proj1.max(dim=2).values
        p2_min = proj2.min(dim=2).values; p2_max = proj2.max(dim=2).values

        margin = 0.0
        sep = (p1_max + margin < p2_min) | (p2_max + margin < p1_min)        # (M,sub_T,4)
        intersects[:, start:end] = ~sep.any(dim=2)         # (M,sub_T)
        '''
        if chunk_id == 20:
            # 예를 들어 0번째 Gaussian과 0번째 tile
            print("== DEBUG: SAT check on 0-th Gaussian and 0-th Tile ==")
            print("proj1 (gaussian):", proj1[0, 0])
            print("proj2 (tile):", proj2[0, 0])
            print("proj1 min-max:", p1_min[0, 0], p1_max[0, 0])
            print("proj2 min-max:", p2_min[0, 0], p2_max[0, 0])
            print("separated axes:", sep[0, 0])
            print("→ final intersect?:", intersects[0, 0])
        '''
    num_unique_intersecting_tiles = intersects.any(dim=0).sum().item()
    print(f"Number of unique intersecting tiles: {num_unique_intersecting_tiles}")
    return intersects

 
    
def vectorize_candidate_generation(num_culled, tx_min, tx_max, ty_min, ty_max):
    """
    주어진 for 루프를 벡터화하여 가우시안-타일 후보 목록을 생성합니다.
    """
    device = tx_min.device

    # 처리할 가우시안이 없는 경우 즉시 빈 텐서 반환
    if num_culled == 0:
        return torch.empty((0, 3), dtype=torch.long, device=device)

    # 1. 각 가우시안이 차지하는 타일의 최대 너비와 높이 계산
    tile_widths = tx_max - tx_min + 1
    tile_heights = ty_max - ty_min + 1
    
    max_width = torch.max(tile_widths)
    max_height = torch.max(tile_heights)

    # 2. 기준이 될 로컬 오프셋 그리드 생성 (0,0), (0,1), ..., (h,w)
    # 크기: (max_height, max_width)
    dx = torch.arange(max_width, device=device)
    dy = torch.arange(max_height, device=device)
    # 'ij' 인덱싱: 첫 번째 인덱스가 y(행), 두 번째가 x(열)이 되도록 함
    grid_y, grid_x = torch.meshgrid(dy, dx, indexing='ij')

    # 3. 브로드캐스팅으로 모든 가우시안에 대한 전체 타일 후보 생성
    # tx_min[:, None, None]는 (N,) -> (N, 1, 1)로 차원을 확장하여
    # (max_height, max_width) 크기의 그리드와 연산이 가능하게 함
    # 결과 텐서의 크기: (num_culled, max_height, max_width)
    candidate_tx = tx_min[:, None, None] + grid_x
    candidate_ty = ty_min[:, None, None] + grid_y
    
    # 4. 유효성 검사를 위한 마스크 생성
    # 생성된 후보가 실제 경계(tx_max, ty_max) 내에 있는지 확인
    mask_x = candidate_tx <= tx_max[:, None, None]
    mask_y = candidate_ty <= ty_max[:, None, None]
    valid_mask = mask_x & mask_y

    # 5. 유효한 후보들만 선택하여 최종 결과 조합
    # 가우시안 인덱스 g도 브로드캐스팅하여 마스크와 같은 크기로 만듦
    g_indices = torch.arange(num_culled, device=device)[:, None, None].expand_as(valid_mask)
    
    # valid_mask가 True인 위치의 값들만 1D 텐서로 추출
    valid_g = g_indices[valid_mask]
    valid_tx = candidate_tx[valid_mask]
    valid_ty = candidate_ty[valid_mask]
    
    # 추출된 1D 텐서들을 합쳐 (M, 3) 형태의 최종 결과 생성
    # M은 유효한 (가우시안, 타일) 쌍의 총 개수
    candidates = torch.stack([valid_g, valid_tx, valid_ty], dim=1)
    
    return candidates


def calculate_bitmap_obb(
    intersects:    torch.Tensor,  # (M, T)
    gauss_centers: torch.Tensor,  # (M, 2)
    gauss_axes:    torch.Tensor,  # (M, 2, 2)
    gauss_radii:   torch.Tensor,  # (M, 2)
    tile_origins:  torch.Tensor,  # (T, 2)
    tile_size:     int,
    subtile_size:  int,
    batch_size:    int = 5000,
):
    """
    SAT 교차된 (gauss, tile) 쌍에 대해,
    픽셀 단위가 아니라 subtile 단위로 0/1 bitmap 을 계산합니다.
    메모리 폭발을 방지하기 위해 (gauss, tile) 쌍을 batch 처리합니다.
    """
    device = gauss_centers.device
    dtype = gauss_centers.dtype

    # precompute: subtile 위치 오프셋
    n_sub_x = math.ceil(tile_size / subtile_size)
    n_sub_y = math.ceil(tile_size / subtile_size)
    S = n_sub_x * n_sub_y

    #타일 내부에서 subtile 펼치기
    sx = torch.arange(0, n_sub_x * subtile_size, subtile_size, device=device).clamp(max=tile_size - subtile_size)
    sy = torch.arange(0, n_sub_y * subtile_size, subtile_size, device=device).clamp(max=tile_size - subtile_size)
    bx, by = torch.meshgrid(sx, sy, indexing='xy')
    block_offsets = torch.stack([bx.flatten(), by.flatten()], dim=1).to(dtype=dtype)   # (S,2)

    rect_axes = torch.tensor([[1, 0], [0, 1]], device=device, dtype=dtype)

    all_gauss_idx = []
    all_tile_idx = []
    all_bitmap = []
    all_tx_ty = []

    # SAT 결과로부터 실제 교차 쌍 (gauss_idx, tile_idx)
    gauss_idx_full, tile_idx = torch.nonzero(intersects, as_tuple=True)  # (P,), (P,)
    total_pairs = gauss_idx_full.shape[0]

    for start in tqdm(range(0, total_pairs, batch_size), desc="Calculating OBB bitmap", ncols=100):
        end = min(start + batch_size, total_pairs)

        gauss_idx = gauss_idx_full[start:end]
        tile_idx_b = tile_idx[start:end]
        P = gauss_idx.shape[0]

        # --- 1단계: OBB의 AABB 계산 ---
        # 각 가우시안의 OBB 코너를 먼저 계산합니다.
        centers = gauss_centers[gauss_idx]
        axes    = gauss_axes[gauss_idx]
        radii   = gauss_radii[gauss_idx]
        obb_corners = _get_obb_corners(centers, axes, radii) # (P, 4, 2)

        # OBB를 감싸는 AABB의 최소/최대 좌표를 찾습니다. (배치 내 가우시안별로 한 번만 계산)
        obb_mins = obb_corners.min(dim=1).values  # (P, 2)
        obb_maxs = obb_corners.max(dim=1).values  # (P, 2)

        # --- 2단계: 서브타일 AABB와 간단한 겹침 검사 ---
        # 각 서브타일의 시작 좌표 계산
        tile_xy = tile_origins[tile_idx_b]
        subtile_origins = tile_xy[:, None, :] + block_offsets[None, :, :] # (P, S, 2)

        # 각 서브타일의 AABB 최소/최대 좌표 계산
        subtile_mins_x = subtile_origins[..., 0] # (P, S)
        subtile_mins_y = subtile_origins[..., 1] # (P, S)
        subtile_maxs_x = subtile_mins_x + subtile_size
        subtile_maxs_y = subtile_mins_y + subtile_size

        # 브로드캐스팅을 이용한 AABB 겹침 검사 (SAT보다 훨씬 빠름)
        # 가우시안의 AABB와 서브타일의 AABB가 겹치는지 확인
        overlap_x = (subtile_mins_x < obb_maxs[:, None, 0]) & (subtile_maxs_x > obb_mins[:, None, 0])
        overlap_y = (subtile_mins_y < obb_maxs[:, None, 1]) & (subtile_maxs_y > obb_mins[:, None, 1])
        
        # 두 축에서 모두 겹치면 최종적으로 겹치는 것으로 판단
        bitmap = (overlap_x & overlap_y).to(torch.uint8)

        # (이후 로직은 동일)
        all_gauss_idx.append(gauss_idx.cpu())
        all_tile_idx.append(tile_idx_b.cpu())
        all_bitmap.append(bitmap.cpu())
        all_tx_ty.append(tile_xy.cpu())

    gauss_idx = torch.cat(all_gauss_idx, dim=0).to(device)
    tile_idx  = torch.cat(all_tile_idx, dim=0).to(device)
    bitmap    = torch.cat(all_bitmap, dim=0).to(device)
    tx_ty     = torch.cat(all_tx_ty, dim=0).to(device)

    return gauss_idx, tile_idx, bitmap, tx_ty





def calculate_bitmap_aabb(
    obb_corners:   torch.Tensor,
    tile_origins:  torch.Tensor,
    tile_size:     int,
    subtile_size:  int
):
    with torch.no_grad():
        device = obb_corners.device
        dtype  = obb_corners.dtype

        mins = obb_corners.min(dim=1).values
        maxs = obb_corners.max(dim=1).values

        tx = torch.div(tile_origins[:,0], tile_size, rounding_mode='trunc')
        ty = torch.div(tile_origins[:,1], tile_size, rounding_mode='trunc')

        n_sub = math.ceil(tile_size / subtile_size)
        sx = torch.arange(0, n_sub*subtile_size, subtile_size, device=device).clamp(max=tile_size-subtile_size)
        sy = torch.arange(0, n_sub*subtile_size, subtile_size, device=device).clamp(max=tile_size-subtile_size)
        by, bx = torch.meshgrid(sy, sx, indexing='ij')
        block_offsets = torch.stack([bx.flatten(), by.flatten()], dim=1)
        S = block_offsets.shape[0]

        all_gauss_idx = []
        all_tile_idx = []
        all_bitmap = []
        all_tx_ty = []

        B = 2000
        total = mins.shape[0]   #AABB gaussian 개수
        
        for i in tqdm(range(0, total, B), desc="Calculating bitmap AABB", ncols=100):
            mins_b = mins[i:i+B]
            maxs_b = maxs[i:i+B]

            tx_min = torch.div(mins_b[:,0], tile_size, rounding_mode='trunc')
            tx_max = torch.div(maxs_b[:,0], tile_size, rounding_mode='trunc')
            ty_min = torch.div(mins_b[:,1], tile_size, rounding_mode='trunc')
            ty_max = torch.div(maxs_b[:,1], tile_size, rounding_mode='trunc')

            mask_tile = (
                (tx[None,:] >= tx_min[:,None]) & (tx[None,:] <= tx_max[:,None]) &
                (ty[None,:] >= ty_min[:,None]) & (ty[None,:] <= ty_max[:,None])
            )

            gauss_idx_b, tile_idx_b = torch.nonzero(mask_tile, as_tuple=True)
            if gauss_idx_b.numel() == 0:
                continue
            
            gauss_idx_full = i + gauss_idx_b
            tile_origin_b = tile_origins[tile_idx_b]

            origins = tile_origin_b[:, None, :] + block_offsets[None, :, :]
            x0 = origins[..., 0]
            y0 = origins[..., 1]
            x1 = x0 + subtile_size
            y1 = y0 + subtile_size

            gx0 = mins[gauss_idx_full, 0][:, None]
            gy0 = mins[gauss_idx_full, 1][:, None]
            gx1 = maxs[gauss_idx_full, 0][:, None]
            gy1 = maxs[gauss_idx_full, 1][:, None]

            overlap_x = (x0 < gx1) & (x1 > gx0)
            overlap_y = (y0 < gy1) & (y1 > gy0)
            bitmap = (overlap_x & overlap_y).to(torch.uint8)

            all_gauss_idx.append(gauss_idx_full.cpu())
            all_tile_idx.append(tile_idx_b.cpu())
            all_bitmap.append(bitmap.cpu())
            all_tx_ty.append(tile_origin_b.cpu())
            # 중간 텐서 제거
            del origins, x0, y0, x1, y1, gx0, gy0, gx1, gy1, overlap_x, overlap_y, bitmap
            torch.cuda.empty_cache()
            


        gauss_idx = torch.cat(all_gauss_idx, dim=0).to(device)
        tile_idx  = torch.cat(all_tile_idx, dim=0).to(device)
        bitmap    = torch.cat(all_bitmap, dim=0).to(device)
        tx_ty     = torch.cat(all_tx_ty, dim=0).to(device)

    return gauss_idx, tile_idx, bitmap, tx_ty
