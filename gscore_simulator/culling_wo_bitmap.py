# gscore_simulator/culling.py

import numpy as np
import torch
from tqdm import tqdm
import sys, os
import math
import torch.nn.functional as F
from typing import Tuple 
from statistics import mean



from collections import defaultdict
#from gscore_simulator.structures import Gaussian2D
#from gscore_simulator.utils import get_view_matrix, get_projection_matrix
from gaussian_splatting.utils.graphics_utils import fov2focal
from gaussian_splatting.utils.sh_utils import eval_sh
from gscore_simulator.structures import RenderMetrics


sys.path.append("/home/hyoh/shape-aware-GS-processor/")

from test_gpu import analytic_2x2_eigendecomp
from preprocessing import in_frustum, compute_cov3d

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
        metrics = RenderMetrics()

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
            p_view = xyz @ R.T + t   # (N, 3)
        else:  # (4, 4)
            p_view_h = xyz_h @ view_matrix  # (N, 4)
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
        ndc      = p[:, :3] / p[:, 3:4] 
        '''
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
        
        # 5) 필터링된 NDC만 남김
        ndc = ndc[in_ndc_mask]
        p = p[in_ndc_mask]
                               
        screen_u = (ndc[:,0] + 1.0) * 0.5 * w
        screen_v = (ndc[:,1] + 1.0) * 0.5 * h
        '''
        final_mask = valid_mask
        # 유효한 가우시안 데이터만 남김(final mask -> valid mask로 변경(디버깅))
        xyz, opacities, scales, rotations, shs, p_view, valid_original_ids, cov3D, depths = \
            xyz[final_mask], opacities[final_mask], scales[final_mask], \
            rotations[final_mask], shs[final_mask], p_view[final_mask], \
            original_indices[final_mask], cov3D[final_mask], depths[final_mask]
        '''
        # --- 디버깅 코드 추가 ---
        print("\n--- p_view (Camera Space Coords) Stats ---")
        if p_view.numel() > 0:
            print(f"p_view X: min={p_view[:,0].min():.2f}, max={p_view[:,0].max():.2f}, mean={p_view[:,0].mean():.2f}")
            print(f"p_view Y: min={p_view[:,1].min():.2f}, max={p_view[:,1].max():.2f}, mean={p_view[:,1].mean():.2f}")
            print(f"p_view Z (Depth): min={p_view[:,2].min():.2f}, max={p_view[:,2].max():.2f}, mean={p_view[:,2].mean():.2f}")
        print("------------------------------------------\n")
        # --- 디버깅 코드 끝 ---
        '''
        #screen_coords = torch.stack([screen_u, screen_v], dim=-1)

        #print("[DEBUG] screen_u stats:", screen_u.min().item(), screen_u.max().item(), screen_u.shape)
        #print("[DEBUG] screen_v stats:", screen_v.min().item(), screen_v.max().item(), screen_v.shape)

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
        cov2d_t = cov2d_t[:, :2, :2]
    
        # 앨리어싱(aliasing)을 방지하고 수치적 안정성을 위해
        # 대각선에 작은 값을 더해주는 저역 통통과 필터입니다.
        cov2d_t[:, 0, 0] += 0.3
        cov2d_t[:, 1, 1] += 0.3

        # 2-3) precompute color
        shs_view = shs.transpose(1, 2).view(-1, 3, (gaussian_model.max_sh_degree+1)**2)
        dir_pp = xyz - cam_center.repeat(shs.shape[0],1)               
        dir_pp_norm = dir_pp / dir_pp.norm(dim=1,keepdim=True) 
        dir_pp_norm = dir_pp_norm.to(device)
        sh2rgb_t = eval_sh(gaussian_model.active_sh_degree, shs_view, dir_pp_norm)  # (N,3) Tensor
        colors_precomp_t = torch.clamp_min(sh2rgb_t + 0.5, 0.0)             # (N,3)
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
        print("Running Final Logic (AABB/OBB intersection test)...")
        
        # 3-1) OOBB 계산
        tile_size   = config["tile_size"]

        # 각 타일의 시작 좌표(tx, ty) 계산
        num_tiles_x = math.ceil(w / tile_size)
        num_tiles_y = math.ceil(h / tile_size)

        # 2. 계산된 타일 개수만큼 0부터 시작하는 인덱스 텐서를 만듭니다.
        ix = torch.arange(num_tiles_x, device=device)
        iy = torch.arange(num_tiles_y, device=device)

        # 3. 인덱스에 타일 크기를 곱해 각 타일의 시작 좌표(tx, ty)를 생성합니다.
        xs = ix * tile_size
        ys = iy * tile_size

        # 4. (선택) 마지막 타일이 이미지 경계를 넘지 않도록 시작점을 보정합니다.
        #    이 clamp 로직은 그대로 사용하셔도 좋습니다. 마지막 타일의 시작점을
        #    '이미지 끝 - 타일 크기' 위치로 당겨와서 타일이 이미지 밖에 그려지는 것을 막습니다.
        xs = xs.clamp(max=w - tile_size)
        ys = ys.clamp(max=h - tile_size)

        
        tx_all, ty_all = torch.meshgrid(xs, ys, indexing="xy")
        tx_all = tx_all.flatten()
        ty_all = ty_all.flatten()
        
        tile_origins = torch.stack([tx_all, ty_all], dim=1)  # (T, 2)

        # 3-1) OBB 계산
        tile_corners   = _get_tile_corners(tx_all, ty_all, tile_size)

        # 공분산으로부터 최소/최대 고유벡터와 반지름 구하기
        _, _, e_min          = analytic_2x2_eigendecomp(cov2d_t)    
        R1, R2               = shape_analysis(cov2d_t)

        e_max                = torch.stack([-e_min[:, 1], e_min[:, 0]], dim=1)
        eigenvectors         = torch.stack([e_max, e_min], dim=2)

        obb_axes             = eigenvectors
        obb_axes             = torch.nn.functional.normalize(obb_axes, dim=1) 
        obb_radii            = torch.stack([R1, R2], dim=1)  # (N, 2)
        obb_corners          = _get_obb_corners(center2d_t, obb_axes, obb_radii)
        
        
        # 3-2) AABB intersection test
        mins = obb_corners.min(dim=1).values  # (N,2)
        maxs = obb_corners.max(dim=1).values  # (N,2)
        tx_min = torch.div(mins[:,0], tile_size, rounding_mode='trunc')  #겹치는 tile x 시작 좌표 (N,)
        #tx_max = torch.div(maxs[:,0], tile_size, rounding_mode='trunc')  #겹치는 tile x 끝 좌표 (N,)
        ty_min = torch.div(mins[:,1], tile_size, rounding_mode='trunc')  #겹치는 tile y 시작 좌표 (N,)
        #ty_max = torch.div(maxs[:,1], tile_size, rounding_mode='trunc')  #겹치는 tile y 끝 좌표  (N,)
        tx_max = torch.div((maxs[:,0] - 1).clamp(min=0), tile_size, rounding_mode='trunc')
        ty_max = torch.div((maxs[:,1] - 1).clamp(min=0), tile_size, rounding_mode='trunc')
        tx_min = tx_min.clamp(0, num_tiles_x - 1)
        tx_max = tx_max.clamp(0, num_tiles_x - 1)
        ty_min = ty_min.clamp(0, num_tiles_y - 1)
        ty_max = ty_max.clamp(0, num_tiles_y - 1)
        #각 가우시안에 대해 aabb 만들고 겹치는 타일 시작 좌표까지 구함
        '''
        # ==================== 디버깅 코드: 단일 가우시안 추적 ====================
        target_gaussian_idx = 0 # 확인할 가우시안 인덱스 (0번째 가우시안)
        
        print(f"\n--- Tracing Gaussian ID: {target_gaussian_idx} ---")
        
        # 1. 이 가우시안의 2D 중심 좌표
        center_coord = screen_coords[target_gaussian_idx]
        print(f"Center (screen_coords): {center_coord.cpu().numpy()}")

        # 2. 이 가우시안의 OBB 꼭짓점들
        obb_c = obb_corners[target_gaussian_idx]
        print(f"OBB Corners:\n{obb_c.cpu().numpy()}")

        # 3. OBB를 감싸는 AABB의 최소/최대 좌표
        min_coord = mins[target_gaussian_idx]
        max_coord = maxs[target_gaussian_idx]
        print(f"AABB of OBB: min={min_coord.cpu().numpy()}, max={max_coord.cpu().numpy()}")

        # 4. AABB가 걸치는 타일 인덱스 범위
        tile_min_x = tx_min[target_gaussian_idx]
        tile_max_x = tx_max[target_gaussian_idx]
        tile_min_y = ty_min[target_gaussian_idx]
        tile_max_y = ty_max[target_gaussian_idx]
        print(f"Overlapping Tile Index Range: x=[{tile_min_x.item()}, {tile_max_x.item()}], y=[{tile_min_y.item()}, {tile_max_y.item()}]")
        
        print("---------------------------------------\n")
        # ==================== 디버깅 코드 끝 ======================
        '''
        # 3-3) decide OBB usage: multiple tiles & aspect ratio
        multi_tile = (tx_max > tx_min) | (ty_max > ty_min)
        aspect = (R1 / (R2 + 1e-9)) > 2
        use_obb   = multi_tile & aspect  # (N,)
        GID_obb   = GID[use_obb]
        GID_aabb  = GID[~use_obb]
        #print(f"[DEBUG] # of multi_tile gaussians: {(multi_tile).sum().item()}")
        #print(f"[DEBUG] # of high_aspect gaussians (R2/R1 > 2): {(aspect).sum().item()}")
        #print(f"[DEBUG] # of gaussians using OBB: {(use_obb).sum().item()}")
        print(f"\nnumber of OBB test: {GID_obb.numel()}, number of AABB test: {GID_aabb.numel()}")

        # ======================================================================
        # <<< 로직 수정: 비트맵 계산 제외, 교차 쌍만 계산 >>>
        # ======================================================================

        # AABB 테스트를 위한 가우시안-타일 교차 쌍 계산
        gaussians_aabb = GID_aabb
        gs_idx_aabb_test_list = []
        tile_idx_aabb_test_list = []

        # AABB 가우시안을 작은 청크로 나누어 처리
        aabb_chunk_size = 8192  # 메모리 상황에 맞게 조절 가능한 청크 크기
        if gaussians_aabb.numel() > 0:
            # 비교 기준이 될 타일 인덱스는 한 번만 생성
            tile_indices_x = torch.arange(num_tiles_x, device=device).repeat(num_tiles_y)
            tile_indices_y = torch.arange(num_tiles_y, device=device).repeat_interleave(num_tiles_x)

            for i in tqdm(range(0, gaussians_aabb.numel(), aabb_chunk_size), desc="AABB Test Chunks"):
                start = i
                end = min(i + aabb_chunk_size, gaussians_aabb.numel())
                
                # 현재 청크에 해당하는 가우시안들의 타일 범위
                tx_min_chunk = tx_min[~use_obb][start:end]
                tx_max_chunk = tx_max[~use_obb][start:end]
                ty_min_chunk = ty_min[~use_obb][start:end]
                ty_max_chunk = ty_max[~use_obb][start:end]

                # 작은 청크에 대해서만 mask_tile 계산 (메모리 사용량 감소)
                mask_tile_chunk = (
                    (tile_indices_x[None, :] >= tx_min_chunk[:, None]) & (tile_indices_x[None, :] <= tx_max_chunk[:, None]) &
                    (tile_indices_y[None, :] >= ty_min_chunk[:, None]) & (tile_indices_y[None, :] <= ty_max_chunk[:, None])
                )
                
                local_gs_idx, tile_idx_chunk = torch.nonzero(mask_tile_chunk, as_tuple=True)

                if local_gs_idx.numel() > 0:
                    # local 인덱스는 현재 청크 내에서의 인덱스이므로, 청크의 시작(start)을 더해줌
                    global_gs_idx = local_gs_idx + start 
                    gs_idx_aabb_test_list.append(global_gs_idx)
                    tile_idx_aabb_test_list.append(tile_idx_chunk)

        # 모든 청크의 결과를 하나로 합침
        if gs_idx_aabb_test_list:
            gs_idx_aabb_test = torch.cat(gs_idx_aabb_test_list)
            tile_idx_aabb_test = torch.cat(tile_idx_aabb_test_list)
        else:
            gs_idx_aabb_test = torch.empty(0, dtype=torch.long, device=device)
            tile_idx_aabb_test = torch.empty(0, dtype=torch.long, device=device)

        # --- 수정 제안 끝 ---
        # OBB 테스트를 위한 가우시안-타일 교차 쌍 계산
        gaussians_obb = GID_obb
        if gaussians_obb.numel() > 0:
            obb_corners_obb_test = obb_corners[GID_obb]
            obb_axes_obb_test = obb_axes[GID_obb]
            gs_idx_obb_test, tile_idx_obb_test = check_sat_intersection(obb_corners_obb_test, obb_axes_obb_test, tile_corners) # (gaussians_obb, tiles)
        else:
            gs_idx_obb_test = torch.empty(0, dtype=torch.long, device=device)
            tile_idx_obb_test = torch.empty(0, dtype=torch.long, device=device)

        print(f"\n[DEBUG] AABB test: Found {gs_idx_aabb_test.shape[0]} intersecting pairs.")
        print(f"[DEBUG] OBB test: Found {gs_idx_obb_test.shape[0]} intersecting pairs.")
        
        # 5. GSU 입력을 위한 최종 데이터 구조 생성(pair 분해:  tile → [gaussian, gaussian, ...] )
        print("Reformatting data for GSU...")
        # 1) 겹친다고 판정된 가우시안 / 타일 종류
          # 각 텐서에서 256 이상인 값의 개수를 계산하고 출력합니다.
         # 1. 상대 인덱스를 사용해 실제 전역 가우시안 ID를 가져옵니다.

        if gs_idx_aabb_test.numel() > 0:
            global_gs_idx_aabb = GID_aabb[gs_idx_aabb_test]
        else:
            global_gs_idx_aabb = torch.empty(0, dtype=torch.long, device=device)

        if gs_idx_obb_test.numel() > 0:
            global_gs_idx_obb = GID_obb[gs_idx_obb_test]
        else:
            global_gs_idx_obb = torch.empty(0, dtype=torch.long, device=device)

        all_gauss_idxs = torch.cat([global_gs_idx_aabb, global_gs_idx_obb], dim=0)
        all_tile_idxs = torch.cat([tile_idx_aabb_test, tile_idx_obb_test], dim=0)

        # (2) 타일 → 가우시안 매핑 만들기
        G_list = {}
        unique_tiles     = all_tile_idxs.unique()
        print(f"기존 타일 개수: {tile_origins.shape[0]}")
        print(f"intersected tiles 종류 개수: {unique_tiles.shape}")

        for t in tqdm(unique_tiles.tolist(), desc="Make G_list"):
            gauss_ids = all_gauss_idxs[all_tile_idxs== t].tolist()

            # 2) 이 타일의 시작 좌표 [tx, ty]
            txty = tile_origins[int(t)].tolist()

            # 3) 각 gaussian에 대해서 idx와 depth 추출
            entries = []
            for g in gauss_ids:
                entries.append({
                    "gaussian_idx": g,
                    "depth": float(depths[g])   # depths는 (N,) Tensor
                })

            # 4) 딕셔너리에 저장
            G_list[t] = {
                "txty":       txty,
                "gaussians":  entries
            }

        '''
        # --- 디버깅 코드: G_list 매핑 상세 검증 ---
        print("\n--- G_list Mapping Verification (Detailed) ---")

        if unique_tiles.numel() > 0:
            # 검증할 샘플 타일 선택 (첫 번째 타일)
            test_tile_idx = unique_tiles[0].item()

            # 1. 기대값: 원본 데이터에서 ID 목록을 집합(set)으로 변환
            expected_ids = set(all_gauss_idxs[all_tile_idxs == test_tile_idx].tolist())

            if test_tile_idx in G_list:
                # 2. 실제 결과: G_list에서 ID 목록을 집합(set)으로 변환
                actual_gauss_entries = G_list[test_tile_idx]["gaussians"]
                actual_ids = {entry["gaussian_idx"] for entry in actual_gauss_entries}

                # 3. 두 집합을 비교하여 일치/불일치 및 차이점 출력
                print(f"Verifying mapping for Tile Index: {test_tile_idx}")
                print(f" - Expected count from source: {len(expected_ids)}")
                print(f" - Actual count in G_list: {len(actual_ids)}")

                if expected_ids == actual_ids:
                    print(" -> Verification SUCCESS: All IDs match perfectly.")
                else:
                    print(" -> Verification FAILED: Mismatch in IDs found!")
                    # 어떤 ID가 사라지거나 추가되었는지 확인
                    missing_ids = expected_ids - actual_ids
                    extra_ids = actual_ids - expected_ids
                    if missing_ids:
                        print(f"   - IDs Missing from G_list ({len(missing_ids)}): {missing_ids}")
                    if extra_ids:
                        print(f"   - Unexpected IDs in G_list ({len(extra_ids)}): {extra_ids}")
            else:
                print(f"Error: Test tile index {test_tile_idx} not found in G_list.")
        else:
            print("No unique tiles found to verify.")
            
        print("------------------------------------------\n")
        # --- 디버깅 코드 끝 ---
        '''
        #[METRICS]
        target_list = []
        target_idx_list = []
        count_list = []

        for target in [(32, 16), (48, 64), (208, 368)]:
            # origin이 target인 첫 번째 타일 t를 찾음
            matching = [t for t, data in G_list.items() if data["txty"] == list(target)]
            if matching:
                t_idx = matching[0]
                count = len(G_list[t_idx]["gaussians"])
            else:
                print("No target!")
                count = 0
                t_idx = None
            count_list.append(count)
            target_list.append(target)
            target_idx_list.append(t_idx)

        counts = [len(data["gaussians"]) for data in G_list.values()]
        metrics.gaussians_per_tile = count_list
        metrics.mac_per_tile = [x * 4 for x in count_list]
        metrics.avg_gaussians_per_tile = mean(counts) if counts else 0
        metrics.max_gaussians_per_tile = max(counts)  if counts else 0
        metrics.tile_coords = target_list
        metrics.tile_idxs = target_idx_list

        #Metrics 계산 확인
        print(f"gaussian per tile: {metrics.gaussians_per_tile}")
        print(f"average gaussian: {metrics.avg_gaussians_per_tile}")
        print(f"max gaussians per tile: {metrics.max_gaussians_per_tile}")
        print(f"at {metrics.tile_coords}, {metrics.tile_idxs}")
         
        culled_gaussians = {  
                    "mean":           means_valid,
                    "cov":            covs_valid,
                    "opacity":        opacities_valid,
                    "scales":         scales_valid,
                    "rotations":      rotations_valid,
                    "colors_precomp": colors_precomp_t   
                }

        return culled_gaussians, G_list, metrics
    
    

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
        C_z * det_inv,   # A = C_z / det
        -C_y * det_inv,  # B = -C_y / det
        C_x * det_inv    # C = C_x / det
    ], dim=1)           # shape (M, 3)
    
    mid  = 0.5 * (C_x + C_z)                 # shape (M,)
    disc = mid * mid - det                   # shape (M,)
    disc_clamped = torch.clamp(disc, min=0.1)  # shape (M,)
    sqrt_disc    = torch.sqrt(disc_clamped)    # shape (M,)
    
    lam1 = mid + sqrt_disc                   # largest eigenvalue (σ²)
    lam2 = mid - sqrt_disc                   # smallest eigenvalue (σ²)
    lam2 = torch.clamp(lam2, min=0)

    R1 = torch.ceil(3*torch.sqrt(lam1))
    R2 = torch.ceil(3*torch.sqrt(lam2))
    R1 = torch.clamp(R1, min=0.5, max=128)  #tile size 2배로 클램핑 128 최적
    R2 = torch.clamp(R2, min=0.5, max=128)
    
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
        
    return intrinsics, width, height


def project_3d_to_2d(cov3d, cam_space_points, camera, intrinsics, device, global_ids):
        
    # 디버깅할 가우시안 ID 설정
    #TARGET_GAUSSIAN_IDX = 15
    #is_target_in_batch = cov3d.shape[0] > TARGET_GAUSSIAN_IDX


    viewmatrix = camera.world_view_transform.to(device)
    fx = intrinsics['fx']
    fy = intrinsics['fy']

    t = cam_space_points

    
    
    t_x, t_y, t_z = t[:, 0], t[:, 1], t[:, 2]

    J = torch.zeros((t.shape[0], 3, 3), device=device, dtype=t.dtype)
    J[:, 0, 0] = fx / t_z
    J[:, 0, 2] = -fx * t_x / (t_z * t_z)
    J[:, 1, 1] = fy / t_z
    J[:, 1, 2] = -fy * t_y / (t_z * t_z)

    Vrk = torch.zeros((cov3d.shape[0], 3, 3), device=device, dtype=cov3d.dtype)
    Vrk[:, 0, 0] = cov3d[:, 0]
    Vrk[:, 0, 1] = cov3d[:, 1]; Vrk[:, 1, 0] = cov3d[:, 1]
    Vrk[:, 0, 2] = cov3d[:, 2]; Vrk[:, 2, 0] = cov3d[:, 2]
    Vrk[:, 1, 1] = cov3d[:, 3]
    Vrk[:, 1, 2] = cov3d[:, 4]; Vrk[:, 2, 1] = cov3d[:, 4]
    Vrk[:, 2, 2] = cov3d[:, 5]

    W = viewmatrix[:3, :3]

    #cov_cam = W @ Vrk @ W.transpose(-2, -1) 
    #cov2d = J @ cov_cam @ J.transpose(-2, -1)
    T = W @ J
    cov2d = T.transpose(1, 2) @ Vrk @ T

    #T = J @ W
    #cov2d = T @ Vrk @ T.transpose(1, 2)
    cov2d = cov2d[:, :2, :2]
    cov2d[:, 0, 0] += 0.3
    cov2d[:, 1, 1] += 0.3

    return cov2d
    
    


def _get_obb_corners(center, axes, radii):
    """OBB의 4개 꼭짓점 좌표 계산"""
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

#for low-resolution
'''
def check_sat_intersection(
    box1_corners: torch.Tensor,  # (M, 4, 2)
    box1_axes:    torch.Tensor,  # (M, 2, 2)
    box2_corners: torch.Tensor,   # (T, 4, 2)
    chunk_size: int = 8
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
    print(f"      chunk size(한번에 처리하는 타일 수): {chunk_size}")
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

        axes   = torch.cat([gauss_ax, tile_ax], dim=2)      # (M,sub_T,4,2)
        axes_t = axes.permute(0,1,3,2)                      # (M,sub_T,2,4)

        proj1 = torch.matmul(b1_c, axes_t)
        proj2 = torch.matmul(b2_c, axes_t)

        p1_min = proj1.min(dim=2).values; p1_max = proj1.max(dim=2).values
        p2_min = proj2.min(dim=2).values; p2_max = proj2.max(dim=2).values

        margin = 0.0
        sep = (p1_max < p2_min) | (p2_max < p1_min)      # (M,sub_T,4)
        intersects[:, start:end] = ~sep.any(dim=2)       # (M,sub_T)

         # ===== DEBUG PRINTS =====
        # 여기서 idx 0,0: box1 첫번째, box2 첫번째에 대한 예시를 출력
        
        i, j = 0, 0
        if ~intersects[i, start+j].item():
            print(f"\n--- chunk {chunk_id} (tiles {start}~{end-1}) debug ---")
            print("b1_c[0,0] corners:\n", b1_c[i,j])           # shape (4,2)
            print("b2_c[0,0] corners:\n", b2_c[i,j])           # shape (4,2)
            print("axes[0,0] (4 axes):\n", axes[i,j])          # shape (4,2)
            print("proj1[0,0] shape:", proj1[i,j].shape)       # (4corners,4axes)
            print("proj1[0,0]:\n", proj1[i,j])
            print("p1_min[0,0]:", p1_min[i,j].tolist(), 
                "p1_max[0,0]:", p1_max[i,j].tolist())
            print("proj2[0,0]:\n", proj2[i,j])
            print("p2_min[0,0]:", p2_min[i,j].tolist(), 
                "p2_max[0,0]:", p2_max[i,j].tolist())
            print("sep[0,0] per axis:", sep[i,j].tolist())
            print("intersects[0, start+j]:", intersects[i, start+j].item())
            print("-----------------------------------\n")
        
    num_unique_intersecting_tiles = intersects.any(dim=0).sum().item()
    print(f"Number of unique intersecting tiles / total tiles: {num_unique_intersecting_tiles} / {T}")
    #M_, T_ = torch.nonzero(intersects, as_tuple=True)

    #print(f"numbter of intersected points: {M_.shape}, {T_.shape}")

    return intersects
'''
#for full-resolution
def check_sat_intersection(
    box1_corners: torch.Tensor,  # (M, 4, 2)
    box1_axes:    torch.Tensor,  # (M, 2, 2)
    box2_corners: torch.Tensor,   # (T, 4, 2)
    tile_chunk_size: int = 8,
    gauss_chunk_size: int = 4096
) -> Tuple[torch.Tensor, torch.Tensor]: # 반환 타입 변경
    """
    메모리 사용량을 최적화한 SAT 교차 판정 함수.
    교차하는 (가우시안 인덱스, 타일 인덱스)를 직접 반환합니다.
    
    """
    debug_printed = False

    M = box1_corners.shape[0]
    T = box2_corners.shape[0]
    device = box1_corners.device
    dtype  = box1_corners.dtype

    box1_corners = box1_corners.to(device=device, dtype=dtype)
    box1_axes    = box1_axes.to(device=device, dtype=dtype)
    box2_corners = box2_corners.to(device=device, dtype=dtype)
    intersecting_pairs_list = []
    global_axes = torch.tensor([[1,0],[0,1]], device=device, dtype=box1_corners.dtype)

    for g_start in tqdm(range(0, M, gauss_chunk_size), desc="SAT Gaussian Chunks"):
        g_end = min(g_start + gauss_chunk_size, M)
        sub_M = g_end - g_start
        
        b1_corners_chunk = box1_corners[g_start:g_end]
        b1_axes_chunk = box1_axes[g_start:g_end]

        for t_start in range(0, T, tile_chunk_size):
            t_end = min(t_start + tile_chunk_size, T)
            sub_T = t_end - t_start

            b2_corners_chunk = box2_corners[t_start:t_end]
            
            # --- 중간 계산 로직 (이전과 동일) ---
            b1_c = b1_corners_chunk.unsqueeze(1).expand(-1, sub_T, -1, -1)
            b2_c = b2_corners_chunk.unsqueeze(0).expand(sub_M, -1, -1, -1)
            gauss_ax = b1_axes_chunk.unsqueeze(1).expand(-1, sub_T, -1, -1)
            tile_ax  = global_axes.unsqueeze(0).unsqueeze(0).expand(sub_M, sub_T, -1, -1)
            axes   = torch.cat([gauss_ax, tile_ax], dim=2)
            axes_t = axes.permute(0,1,3,2)
            proj1 = torch.matmul(b1_c, axes_t)
            proj2 = torch.matmul(b2_c, axes_t)
            p1_min, p1_max = proj1.min(dim=2).values, proj1.max(dim=2).values
            p2_min, p2_max = proj2.min(dim=2).values, proj2.max(dim=2).values
            sep = (p1_max < p2_min) | (p2_max < p1_min)
            
            chunk_intersects = ~sep.any(dim=2)
            
            local_g_idx, local_t_idx = torch.nonzero(chunk_intersects, as_tuple=True)
            
            if local_g_idx.numel() > 0:
                # ==================== 디버깅 코드 시작 ====================
                if not debug_printed:
                    # 현재 청크에서 교차한 첫 번째 쌍의 정보만 확인
                    first_g_local = local_g_idx[0]
                    first_t_local = local_t_idx[0]
                    
                    # 전역 인덱스
                    first_g_global = first_g_local + g_start
                    first_t_global = first_t_local + t_start
                    '''
                    print("\n--- SAT Intersection Debug ---")
                    print(f"Found a suspicious intersection: Gaussian Index={first_g_global}, Tile Index={first_t_global}")
                    
                    # 1. 교차했다고 판정된 두 박스의 실제 좌표 출력
                    print("\n[Box Coords]")
                    print("  - Gaussian OBB corners:\n", box1_corners[first_g_global])
                    print("  - Tile AABB corners:\n", box2_corners[first_t_global])

                    # 2. SAT 투영 결과 출력
                    print("\n[SAT Projections]")
                    # 4개의 축: 가우시안 축 2개, 타일 축 2개
                    for axis_idx in range(4):
                        p1 = (p1_min[first_g_local, first_t_local, axis_idx].item(), p1_max[first_g_local, first_t_local, axis_idx].item())
                        p2 = (p2_min[first_g_local, first_t_local, axis_idx].item(), p2_max[first_g_local, first_t_local, axis_idx].item())
                        print(f"  - Axis {axis_idx}: Gauss Proj={p1}, Tile Proj={p2}")

                    print("--------------------------------\n")
                    debug_printed = True
                # ==================== 디버깅 코드 끝 ======================
                '''
                global_g_idx = local_g_idx + g_start
                global_t_idx = local_t_idx + t_start
                intersecting_pairs_list.append(torch.stack([global_g_idx, global_t_idx], dim=1))
    
    if not intersecting_pairs_list:
        # 교차 쌍이 하나도 없는 경우, 빈 텐서를 반환
        empty_tensor = torch.empty(0, dtype=torch.long, device=device)
        return empty_tensor, empty_tensor
        
    # ▼▼▼ 수정된 반환 로직 ▼▼▼
    # 리스트에 저장된 교차 쌍들을 하나의 (P, 2) 텐서로 합칩니다.
    final_pairs = torch.cat(intersecting_pairs_list, dim=0)
    # 0번 열(가우시안 인덱스)과 1번 열(타일 인덱스)을 분리하여 반환합니다.
    return final_pairs[:, 0], final_pairs[:, 1]


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
