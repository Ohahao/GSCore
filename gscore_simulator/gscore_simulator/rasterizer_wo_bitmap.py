import numpy as np
import torch
from tqdm import tqdm
import math
from gscore_simulator.structures import Gaussian2D
from gscore_simulator.structures import RenderMetrics
from gaussian_splatting.utils.graphics_utils import fov2focal
from typing import Tuple 


def rasterize_tiles(
    tile_to_chunks: dict,
    culled_gaussians,
    camera,
    config: dict,
    device,
    metrics
):
    H, W = camera.image_height, camera.image_width
    tile_size = config['tile_size']
    termination_threshold = 1e-4
    image_buffer = torch.zeros((3, H, W), dtype=torch.float32, device=device)
    metrics.total_blending_operations = 0
    device = torch.device(config.get('device', 'cuda'))
    chunk_flag = 0

    # --- 0) culled_gaussians 안의 GPU 텐서 보장 ---
    for k, v in culled_gaussians.items():
        if not torch.is_tensor(v):
            culled_gaussians[k] = torch.as_tensor(v, device=device)
        else:
            culled_gaussians[k] = v.to(device)

    # ★ 시뮬레이션을 위한 총 계산 생략 가우시안 카운터
    total_saved_gaussians = 0

    skip_counters = {
        "no_chunks": 0,
        "no_valid_pixels": 0,
        "early_termination_tiles": 0,
        "runtime_error_chunks": 0,
        "chunk_break_all_opaque": 0,
        "gaussian_skip_low_alpha": 0
    }

    print("Running Vectorized VRU (Bitmap Excluded): Processing each tile...")
    saved_counts_all = []
    saved_rates_all  = []
    calculated_counts_all = []
    first_terminated = False
    target = (208, 368)

    for tile_idx, chunks in tqdm(tile_to_chunks.items(), desc="Rasterizing Tiles"):
        origins = chunks[0]["origin"]
        origins= torch.tensor(origins)
        tx, ty = origins[0], origins[1]  #(N_)

        start_y, start_x = int(ty), int(tx)
        tile_w = min(tile_size, W - start_x)
        tile_h = min(tile_size, H - start_y)
        T = torch.ones(tile_h*tile_w, dtype=torch.float32, device=device)
        C = torch.zeros((3, tile_h*tile_w), device=device)
        chunk_flag = 0
        #N_per_tile = len([g["gaussian_idx"] for g in chunk["chunks"]])  # 타일 당 가우시안 수
        

        if not chunks:
            skip_counters["no_chunks"] += 1
            print(f"[DBG] No Gaussians for tile at origin ({tx},{ty})")
            continue

        # ★ 1. Tile 수준 설정 (픽셀 좌표, 버퍼 등)
        image_h, image_w = image_buffer.shape[1], image_buffer.shape[2]
        
        # 타일 내 픽셀들의 로컬/글로벌 좌표를 한 번에 생성
        grid_y, grid_x = torch.meshgrid(
            torch.arange(tile_size, device=device, dtype=torch.float32),
            torch.arange(tile_size, device=device, dtype=torch.float32),
            indexing='ij'
        )
        px_local_tile = grid_x.flatten()
        py_local_tile = grid_y.flatten()
        px_global_tile = tx + px_local_tile     #(tile*tile,)
        py_global_tile = ty + py_local_tile

        # ★ 3. 깊이 순으로 정렬된 Chunk 단위 루프 시작
        for chunk in chunks:  
            chunk_idx = chunk["chunk_id"]  
            g_id_chk = [g["gaussian_idx"] for g in chunk["gaussians"]]
            N_per_chunk = len(g_id_chk)     #chunk 당 가우시안 수
            alpha = torch.zeros(N_per_chunk, tile_h*tile_w, dtype=torch.float32, device=device)
            
            # ★ 4. 청크 처리 전, 타일 전체에 대한 early-termination
            if chunk_flag:
                # early‐termination 발생 시 저장 통계 계산
                saved_count = sum(len(c["gaussians"]) for c in chunks[chunk_idx :])
                calculated_count = sum(len(c["gaussians"]) for c in chunks[:chunk_idx])
                total_count = sum(len(c["gaussians"]) for c in chunks)
                saved_rate  = (100.0 * saved_count / total_count) if total_count>0 else 0.0

                # (1) 전체 목록에 추가
                calculated_counts_all.append(calculated_count)
                saved_counts_all.append(saved_count)
                saved_rates_all.append(saved_rate)

                # (2) 특정 타일 정보가 필요하면 따로 
                
                if (tx, ty) == target:
                    print("[TRAGET DETECTED!]")
                    first_saved_count = saved_count
                    first_saved_rate  = saved_rate
                    first_calculated_count = calculated_count
                    first_terminated_coords = origins
                    first_terminated  = True
       

                chunk_flag = 0
                break

            # 현재 chunk에 있는 gaussian들에 property 할당 (GPU 메모리 이슈로 배치 단위로 실행)
            batch_size = 32  # 배치 크기 (메모리 상황에 맞게 조절)
            n_batches = len(g_id_chk) // batch_size
            all_color_g = []
            all_mean_g = []
            all_opacity_g = []
            all_inv_cov_g = []

            for i in range(n_batches + 1):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(g_id_chk))

                if start_idx >= end_idx:
                    continue

                mean_g = culled_gaussians["mean"][g_id_chk[start_idx:end_idx]]
                all_mean_g.append(mean_g)

                opacity_g = culled_gaussians["opacity"][g_id_chk[start_idx:end_idx]]
                all_opacity_g.append(opacity_g)
 
                color_g = culled_gaussians["colors_precomp"][g_id_chk[start_idx:end_idx]]
                all_color_g.append(color_g)

                gaus_covs = culled_gaussians["cov"][g_id_chk[start_idx:end_idx]]
                  
                try:
                    inv_cov_g_cpu = torch.linalg.inv(gaus_covs.detach().cpu())
                    inv_cov_g = inv_cov_g_cpu.to(device)
                    all_inv_cov_g.append(inv_cov_g)

                except RuntimeError:
                    skip_counters["runtime_error_chunks"] += 1
                    print(f"DEBUG: Skipping chunk in tile {tile_idx} due to tensor/linalg error.")
                    continue
            
            final_color_g = torch.cat(all_color_g, dim=0)
            final_mean_g = torch.cat(all_mean_g, dim=0)
            final_opacity_g = torch.cat(all_opacity_g, dim=0)
            final_inv_cov_g = torch.cat(all_inv_cov_g, dim=0)   
            #print(f"[DEBUG] color_g: {color_g.shape}, mean_ g: {mean_g.shape}, opacity_g: {opacity_g.shape}, inv cov: {inv_cov_g.shape}")      
            #print(f"intensity: {final_opacity_g}")
            # chunk 단위에서 alpha 값 계산
            M_x = final_mean_g[:, 0].unsqueeze(1) #(N_per_chunk, 1)
            M_y = final_mean_g[:, 1].unsqueeze(1) #(N_per_chunk, 1)
            P_x = px_global_tile.unsqueeze(0)   #(1, N_pixels_per_tile)
            P_y = py_global_tile.unsqueeze(0)
            dx, dy = M_x - P_x, M_y - P_y   #(N_per_chunk, N_pixels_per_tile)
            cx = final_inv_cov_g[:,0,0].unsqueeze(1)  #(N_per_chunk, 1)
            cy = final_inv_cov_g[:,0,1].unsqueeze(1)
            cz = final_inv_cov_g[:,1,1].unsqueeze(1)
            
            power = -0.5*(cx*dx*dx + cz*dy*dy) - cy*dx*dy #(N_per_chunk, N_pixels_per_tile)
            
            clamped_power = torch.clamp(power, min=-15.0, max=0.0)      
            expo = torch.exp(clamped_power)
            thresh = torch.full_like(expo, 0.99, device=device)
            alpha = torch.min(thresh, final_opacity_g * expo)  #(N_per_chunk, N_pixels_per_tile)
            metrics.total_blending_operations += 1 * len(g_id_chk) * 256
             # ==================== 디버깅 코드 시작 ====================
            '''
            if dx.numel() > 0: # 청크가 비어있지 않을 때만 출력
                
                print(f"[DEBUG] power stats: min={power.min():.4f}, max={power.max():.4f}, mean={power.mean():.4f}, std={power.std():.4f}")
                print(f"[DEBUG] dx.mean={dx.mean():.2f}, dy.mean={dy.mean():.2f}")
                print(f"[DEBUG] cx range: {cx.min().item():.2e} ~ {cx.max().item():.2e}")
                print(f"[DEBUG] cz range: {cz.min().item():.2e} ~ {cz.max().item():.2e}")
                print(f"[DEBUG] expo mean: {expo.mean():.2e}")
                print(f"[DEBUG] intensity mean: {final_opacity_g.mean().item():.4f}")
                print(f"[DEBUG] alpha mean: {alpha.mean().item():.6f}")
                
            # ==================== 디버깅 코드 끝 ======================
            '''
            # ★ 5. 현재 청크의 가우시안들을 순서대로 처리하며 블렌딩
            for g_idx in range(len(g_id_chk)):
                alpha_i = alpha[g_idx]  #(N_pixels_per_tile,)
                if (alpha_i.max() < 1.0/255.0):
                    continue
                test_T = T * (1-alpha_i)
                if test_T.mean() < termination_threshold:
                    chunk_flag = 1  #chunk 단위 for문도 break
                    break
                #compute color
                C += final_color_g[g_idx].view(3,1) * (alpha_i * T).view(1, tile_h*tile_w)  #(3, N_pixels_per_tile)
                T = test_T
            final_C = C.reshape(3, tile_h, tile_w)

        # ★ 6. 최종 계산된 타일의 색상을 전체 이미지 버퍼에 누적
        
        ts = tile_size
        
        end_y = min(start_y + ts, image_h)
        end_x = min(start_x + ts, image_w)
        
        pixel = final_C.to(device)
        image_buffer[:, start_y:end_y, start_x:end_x] = pixel

    #METRUCS 저장
    metrics.avg_saved_gaussians = (sum(saved_counts_all) / len(saved_counts_all)) if saved_counts_all else 0
    metrics.avg_saved_rate      = (sum(saved_rates_all)  / len(saved_rates_all))  if saved_rates_all  else 0
    metrics.avg_calculated_gaussians = (sum(calculated_counts_all) / len(calculated_counts_all)) if saved_counts_all else 0

    # (4) target 통계 metrics에 병합
    metrics.first_saved_gaussians = first_saved_count
    metrics.first_saved_rate      = first_saved_rate
    metrics.first_calculated_gaussians = first_calculated_count
    metrics.first_terminated_coords = first_terminated_coords

    print(f"[SUMMARY] avg saved gaussians: {metrics.avg_saved_gaussians:.2f}")
    print(f"[SUMMARY] avg saved rate:       {metrics.avg_saved_rate:.2f}%")
    print(f"[SUMMARY] first saved gaussians: {metrics.first_saved_gaussians:.2f}")
    print(f"[SUMMARY] first saved rate: {metrics.first_saved_rate:.2f}%")
    print(f"[SUMMARY] first terminated coordinate: {metrics.first_terminated_coords}")
    print(f"[SUMMARY] total alpha computations: {metrics.total_blending_operations}")
    rendered_image = image_buffer.cpu().numpy()

    return rendered_image, metrics







'''
def rasterize_tiles(
    tile_to_chunks: dict,
    gaussian_property:          dict,
    camera,
    config: dict,
    device,
    metrics
):
    H, W = camera.image_height, camera.image_width
    tile_size = config['tile_size']
    termination_threshold = 1e-4
    image_buffer = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
    metrics.total_blending_operations = 0
    device = torch.device(config.get('device', 'cuda'))

    # --- 0) culled_gaussians 안의 GPU 텐서 보장 ---
    for k, v in gaussian_property.items():
        if not torch.is_tensor(v):
            gaussian_property[k] = torch.as_tensor(v, device=device)
        else:
            gaussian_property[k] = v.to(device)

    # ★ 시뮬레이션을 위한 총 계산 생략 가우시안 카운터
    total_saved_gaussians = 0
    target = (32, 16)   #tile index: 22
    skip_counters = {
        "no_chunks": 0,
        "no_valid_pixels": 0,
        "early_termination_tiles": 0,
        "runtime_error_chunks": 0,
        "chunk_break_all_opaque": 0,
        "gaussian_skip_low_alpha": 0
    }

    print("Running Vectorized VRU (Bitmap Excluded): Processing each tile...")
    
    for tile_idx, chunks in tqdm(tile_to_chunks.items(), desc="Rasterizing Tiles"):
        tx, ty = chunks["txty"]
        
        if not chunks:
            skip_counters["no_chunks"] += 1
            continue

        # ★ 1. Tile 수준 설정 (픽셀 좌표, 버퍼 등)
        image_h, image_w = image_buffer.shape[0], image_buffer.shape[1]
        
        # 타일 내 픽셀들의 로컬/글로벌 좌표를 한 번에 생성
        grid_y, grid_x = torch.meshgrid(
            torch.arange(tile_size, device=device, dtype=torch.float32),
            torch.arange(tile_size, device=device, dtype=torch.float32),
            indexing='ij'
        )
        px_local_tile = grid_x.flatten()
        py_local_tile = grid_y.flatten()
        px_global_tile = tx + px_local_tile
        py_global_tile = ty + py_local_tile
        
        # 이미지 경계 내에 있는 유효한 픽셀들만 필터링
        valid_pixel_mask = (px_global_tile < image_w) & (py_global_tile < image_h)
        num_valid_pixels = valid_pixel_mask.sum()
        if num_valid_pixels == 0:
            skip_counters["no_valid_pixels"] += 1
            continue
            
        # ★ 2. 타일 전체의 픽셀 상태를 추적하는 버퍼
        pixel_color = torch.zeros(num_valid_pixels, 3, device=device)
        pixel_T = torch.ones(num_valid_pixels, device=device) # Transmittance 버퍼

        # [METRICS] 누적합 초기화
        count = 0
        
        # ★ 3. 깊이 순으로 정렬된 Chunk 단위 루프 시작
        for chunk in chunks:  
            chunk_idx = chunk["chunk_id"]  
            g_id_chk = chunk["gaussians"]


            # ★ 4. 청크 처리 전, 타일 전체에 대한 조기 종료 검사
            if pixel_T.max() < termination_threshold:
                #print("[WARNING] EARLY TERMINATION !!")
                # [METRICS] Early-termination 후 saved_gaussians 계산 코드 (로직 유지)
                if (tx, ty) == target:
                    saved_count = sum(len(remaining_chunk) for remaining_chunk in chunks[chunk_idx:])
                    print("[WARNING] EARLY TERMINATION !!")
                    metrics.saved_gaussians = saved_count
                    metrics.total_gaussians = sum(len(c) for c in chunks)
                    if metrics.total_gaussians > 0:
                        metrics.saved_rate_percent = round(100.0 * metrics.saved_gaussians / metrics.total_gaussians, 2)
                    else:
                        metrics.saved_rate_percent = 0.0
                    print(f"[METRICS] saved gaussians at {(tx, ty)}: {metrics.saved_gaussians}")
                    print(f"[METRICS] saved rate at {(tx, ty)}: {metrics.saved_rate_percent}%")
                break

            # 현재 청크의 데이터로부터 텐서 생성 (비트맵 제외)
            try:
                gaus_means = torch.stack([g.mean for g in chunk], dim=0)
                gaus_opacities = torch.tensor([g.opacity for g in chunk], device=device, dtype=torch.float32)
                gaus_colors = torch.stack([g.color_precomp for g in chunk], dim=0)
                gaus_covs = torch.stack([g.cov for g in chunk], dim=0)
                # 역행렬 계산은 불안정할 수 있으므로 CPU에서 처리 후 다시 GPU로 이동
                #gaus_inv_covs_cpu = compute_pixel_inv_cov2d(gaus_covs, fx, fy)
                gaus_inv_covs_cpu = torch.linalg.inv(gaus_covs.detach().cpu())
                gaus_inv_covs = gaus_inv_covs_cpu.to(device)
            except RuntimeError:
                skip_counters["runtime_error_chunks"] += 1
                print(f"DEBUG: Skipping chunk in tile {tile_idx} due to tensor/linalg error.")
                continue

            # ★ 5. 현재 청크의 가우시안들을 순서대로 처리하며 블렌딩
            for g_idx in range(len(chunk)):
                
                # 비트맵 최적화가 제거되었으므로, 아직 완전히 렌더링되지 않은 모든 픽셀을 대상으로 함
                pixel_mask_for_gaus = (pixel_T > termination_threshold)

                if not pixel_mask_for_gaus.any():
                    skip_counters["chunk_break_all_opaque"] += 1
                    break # 현재 청크의 나머지 가우시안들도 처리할 필요 없음

                # 선택된 픽셀들에 대해서만 알파 블렌딩 수행
                px_active = px_global_tile[valid_pixel_mask][pixel_mask_for_gaus]
                py_active = py_global_tile[valid_pixel_mask][pixel_mask_for_gaus]

                #print(f"알파 블렌딩 실시할 x, y: {px_active}, {py_active}")
                
                mean_g, opacity_g, color_g, inv_cov_g = gaus_means[g_idx], gaus_opacities[g_idx], gaus_colors[g_idx], gaus_inv_covs[g_idx]
                
                dx, dy = px_active - mean_g[0], py_active - mean_g[1]
                cx, cxy, cy = inv_cov_g[0, 0], inv_cov_g[0, 1], inv_cov_g[1, 1]
                power = -0.5 * (cx * dx**2 + cy * dy**2 + 2 * cxy * dx * dy)               
                expo = opacity_g * torch.exp(torch.clamp(power, min=-15.0, max=0.0))
                thresh = torch.full_like(expo, 0.99, device=device)
                alpha  = torch.min(thresh, expo) 

                # 유효한 alpha 값을 가진 픽셀들만 추가로 필터링
                alpha_valid_mask = alpha > (1.0/255.0)
                if not alpha_valid_mask.any():
                    #print(f"[DEBUG] dx.min/max = {dx.min().item():.1f}/{dx.max().item():.1f}, "
                    #f"dy.min/max = {dy.min().item():.1f}/{dy.max().item():.1f}")
                    #print(f"[DEBUG] inv_cov diag = {cx.item():.2e}, {cy.item():.2e}, offdiag = {cxy.item():.2e}")
                    #print(f"[DEBUG] power_raw.min/max = {power.min().item():.2e}/{power.max().item():.2e}") 
                    #print(f"[DEBUG] alpha: ", alpha)
                    skip_counters["gaussian_skip_low_alpha"] += 1
                    continue

                # 실제 업데이트가 필요한 픽셀들의 최종 마스크
                final_update_mask = torch.zeros_like(pixel_mask_for_gaus)
                final_update_mask[pixel_mask_for_gaus] = alpha_valid_mask
                metrics.total_blending_operations += final_update_mask.sum().item()

                # 알파 블렌딩
                T_old = pixel_T[final_update_mask]
                alpha_update = alpha[alpha_valid_mask]
                
                pixel_color[final_update_mask] += (T_old * alpha_update).unsqueeze(-1) * color_g
                pixel_T[final_update_mask] *= (1.0 - alpha_update)

        # ★ 6. 최종 계산된 타일의 색상을 전체 이미지 버퍼에 기록
        final_r = torch.zeros_like(px_local_tile, dtype=torch.float32)
        final_g = torch.zeros_like(px_local_tile, dtype=torch.float32)
        final_b = torch.zeros_like(px_local_tile, dtype=torch.float32)
        
        final_r[valid_pixel_mask] = pixel_color[:, 0]
        final_g[valid_pixel_mask] = pixel_color[:, 1]
        final_b[valid_pixel_mask] = pixel_color[:, 2]

        start_y, start_x = int(ty), int(tx)
        ts = tile_size
        
        end_y = min(start_y + ts, image_h)
        end_x = min(start_x + ts, image_w)
        h_slice = end_y - start_y
        w_slice = end_x - start_x
        
        image_buffer[start_y:end_y, start_x:end_x, 0] = final_r.reshape(ts, ts)[:h_slice, :w_slice]
        image_buffer[start_y:end_y, start_x:end_x, 1] = final_g.reshape(ts, ts)[:h_slice, :w_slice]
        image_buffer[start_y:end_y, start_x:end_x, 2] = final_b.reshape(ts, ts)[:h_slice, :w_slice]

    rendered_image = image_buffer.cpu().numpy()

    # ▼▼▼ 2. 루프가 모두 끝난 후, 최종 카운트 결과 출력 ▼▼▼
    print("\n--- Skip/Break Condition Summary ---")
    print(f"  - Tiles skipped (no chunks): {skip_counters['no_chunks']}")
    print(f"  - Tiles skipped (no valid pixels): {skip_counters['no_valid_pixels']}")
    print(f"  - Tiles skipped (Early Termination): {skip_counters['early_termination_tiles']}")
    print(f"  - Chunks skipped (RuntimeError): {skip_counters['runtime_error_chunks']}")
    print(f"  - Chunks broken (all pixels opaque): {skip_counters['chunk_break_all_opaque']}")
    print(f"  - Gaussians skipped (alpha too low): {skip_counters['gaussian_skip_low_alpha']}")
    print("------------------------------------\n")

    return rendered_image, metrics

'''


# ============== helper ================#
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

def compute_pixel_inv_cov2d(
    cov2d_world: torch.Tensor, 
    fx: float, 
    fy: float
) -> torch.Tensor:
    """
    Convert world-space 2×2 covariance matrices to pixel-space and return their inverses.
    
    Args:
        cov2d_world: Tensor of shape (N, 2, 2), world-space covariances.
        fx, fy: focal lengths in pixels (from intrinsics["fx"], intrinsics["fy"]).
    Returns:
        cov2d_inv_pix: Tensor of shape (N, 2, 2), pixel-space inverse covariances.
    """
    # Build pixel-scale matrix
    S = torch.tensor([[fx, 0.0],
                      [0.0, fy]], 
                     device=cov2d_world.device, 
                     dtype=cov2d_world.dtype)        # (2,2)
    # Broadcast S across batch
    S_expanded = S.unsqueeze(0)                       # (1,2,2)
    # Pixel-space covariance: S · cov2d_world · S
    cov2d_pix = S_expanded @ cov2d_world @ S_expanded  # (N,2,2)
    # Invert each 2×2
    cov2d_pix_cpu = cov2d_pix.detach().cpu()
    cov2d_inv_pix = torch.linalg.inv(cov2d_pix_cpu)       # (N,2,2)
    return cov2d_inv_pix