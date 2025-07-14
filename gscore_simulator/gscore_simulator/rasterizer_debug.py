import numpy as np
import torch
from tqdm import tqdm
import math
from gscore_simulator.structures import Gaussian2D
from gscore_simulator.structures import RenderMetrics


def rasterize_tiles(
    sorted_gaussians_by_tile: dict,
    camera,
    config: dict,
    device
):
    H, W = camera.image_height, camera.image_width
    tile_size = config['tile_size']
    subtile_res = config['subtile_res']
    subtile_size = tile_size // subtile_res
    termination_threshold = 1e-4

    image_buffer = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
    
    metrics = RenderMetrics()

    print("Running GPU-accelerated VRU: Processing each tile...")

    # Debug flag to limit output to the first tile with data (avoids excessive logs)
    debug_print_done = False

    for tile_idx, tiles_data in tqdm(sorted_gaussians_by_tile.items(), desc="Rasterizing Tiles"):
        chunks = tiles_data["chunks"]
        txty = tiles_data["txty"]
        tx, ty = txty

        if not chunks:
            continue

        # Debug: Tile-level information
        if not debug_print_done:
            print(f"DEBUG: Processing tile {tile_idx} at position ({tx}, {ty}) with {len(chunks)} chunk(s).")

        # ★ 1. Tile-level setup (pixel coordinates, buffer, subtile map, etc.)
        image_h, image_w = image_buffer.shape[0], image_buffer.shape[1]

        grid_y, grid_x = torch.meshgrid(
            torch.arange(tile_size, device=device, dtype=torch.float32),
            torch.arange(tile_size, device=device, dtype=torch.float32),
            indexing='ij'
        )
        px_local_tile = grid_x.flatten()
        py_local_tile = grid_y.flatten()
        px_global_tile = (tx * tile_size) + px_local_tile
        py_global_tile = (ty * tile_size) + py_local_tile

        valid_pixel_mask = (px_global_tile < image_w) & (py_global_tile < image_h)
        num_valid_pixels = valid_pixel_mask.sum()
        if num_valid_pixels == 0:
            continue

        # ★ 2. Initialize pixel buffers for the whole tile
        pixel_color = torch.zeros(num_valid_pixels, 3, device=device)
        pixel_T = torch.ones(num_valid_pixels, device=device)  # Transmittance buffer

        subtile_x_map = torch.div(px_local_tile[valid_pixel_mask], subtile_size, rounding_mode='trunc')
        subtile_y_map = torch.div(py_local_tile[valid_pixel_mask], subtile_size, rounding_mode='trunc')
        pixel_to_subtile_map = subtile_y_map * subtile_res + subtile_x_map

        num_skipped_gaussians = 0
        # ★ 3. Loop over depth-sorted chunks
        for chunk_idx, chunk in enumerate(chunks):
            # Debug: Chunk-level information
            if not debug_print_done:
                print(f"DEBUG:   Processing chunk {chunk_idx} with {len(chunk)} Gaussian(s).")
                for g_idx, g in enumerate(chunk):
                    bitmap = g.tiles[tile_idx]['bitmap']
                    if bitmap.sum() == 0:
                        num_skipped_gaussians += 1
                
                print(f"[DEBUG] Tile {tile_idx}: skipped {num_skipped_gaussians} Gaussian(s) with all-zero bitmap.")

            # ★ 4. Early termination check for the whole tile
            if pixel_T.max() < termination_threshold:  # 1e-4
                # Early-termination: count how many Gaussians are saved (skipped)
                saved_count = 0
                for remaining_chunk in chunks[chunk_idx:]:
                    saved_count += len(remaining_chunk)

                #saved_gaussians_per_tile 계산
                if tile_idx == 10:
                    saved_gaussians = saved_count
                if not debug_print_done:
                    print(f"DEBUG:   Early termination in tile {tile_idx}: pixel_T.max() < {termination_threshold}, skipping {saved_count} remaining Gaussians.")
                break  # Stop rendering this tile

            # Create tensors from the current chunk's data
            try:
                gaus_means = torch.stack([g.mean for g in chunk], dim=0).to(device)
                gaus_opacities = torch.tensor([g.opacity for g in chunk], device=device)
                gaus_colors = torch.stack([g.color_precomp for g in chunk], dim=0).to(device)
                gaus_bitmaps = torch.stack([g.tiles[tile_idx]['bitmap'] for g in chunk], dim=0).to(device)
                gaus_covs_cpu = torch.stack([g.cov for g in chunk], dim=0).to('cpu')
                gaus_inv_covs_cpu = torch.linalg.inv(gaus_covs_cpu)
                gaus_inv_covs = gaus_inv_covs_cpu.to(device)
            except RuntimeError:
                print(f"DEBUG: Skipping chunk {chunk_idx} in tile {tile_idx} due to tensor/linalg error.")
                continue

            # ★ 5. Blend each Gaussian in the chunk
            for g_idx in range(len(chunk)):
                # Determine which sub-tiles this Gaussian covers
                active_subtiles = (gaus_bitmaps[g_idx] == 1).nonzero().squeeze(-1)
                '''
                if active_subtiles.numel() == 0:
                    if not debug_print_done and g_idx < 5:
                        print(f"DEBUG:     Gaussian {g_idx}: no active sub-tiles in tile {tile_idx}, skipping.")
                    continue
                '''

                # Find which pixels (within this tile) fall into those sub-tiles and are still active
                pixel_mask_for_gaus = torch.isin(pixel_to_subtile_map, active_subtiles)
                pixel_mask_for_gaus &= (pixel_T > termination_threshold)
                if not pixel_mask_for_gaus.any():
                    if not debug_print_done and g_idx < 5:
                        print(f"DEBUG:     Gaussian {g_idx}: no pixels passed mask (either not in sub-tiles or below T threshold), skipping.")
                    continue

                # Select the active pixel coordinates for this Gaussian
                px_active = px_global_tile[valid_pixel_mask][pixel_mask_for_gaus]
                py_active = py_global_tile[valid_pixel_mask][pixel_mask_for_gaus]

                # Debug: 1. Pixel selection check
                if not debug_print_done and g_idx < 5:
                    sample_n = min(px_active.numel(), 5)
                    print(f"DEBUG:     Gaussian {g_idx}: selected {px_active.numel()} pixel(s). "
                        f"Sample coords (px_active, py_active) = {list(zip(px_active[:sample_n].tolist(), py_active[:sample_n].tolist()))}")

                # Retrieve Gaussian parameters
                mean_g    = gaus_means[g_idx]
                opacity_g = gaus_opacities[g_idx]
                color_g   = gaus_colors[g_idx]
                inv_cov_g = gaus_inv_covs[g_idx]

                # Compute squared Mahalanobis distance components for exponent (power) and alpha
                dx = px_active - mean_g[0]
                dy = py_active - mean_g[1]
                cx, cxy, cy = inv_cov_g[0, 0], inv_cov_g[0, 1], inv_cov_g[1, 1]
                power = -0.5 * (cx * dx**2 + cy * dy**2 + 2 * cxy * dx * dy)
                alpha = opacity_g * torch.exp(torch.clamp(power, min=-15.0, max=0.0))

                # Debug: 2. Alpha calculation check (power, inv_cov_g, alpha)
                if not debug_print_done and g_idx < 5:
                    cx_val = cx.item(); cxy_val = cxy.item(); cy_val = cy.item()
                    print(f"DEBUG:     Gaussian {g_idx}: inv_cov_g (cx={cx_val:.3f}, cxy={cxy_val:.3f}, cy={cy_val:.3f}), "
                        f"power range [{power.min().item():.2f}, {power.max().item():.2f}], "
                        f"alpha range [{alpha.min().item():.3e}, {alpha.max().item():.3e}], opacity={opacity_g.item():.3f}")

                # Save old transmittance for selected pixels and update color & transmittance
                T_old = pixel_T[pixel_mask_for_gaus]
                # Debug: 3. Before accumulation (T_old)
                if not debug_print_done and g_idx < 5:
                    sample_n = min(T_old.numel(), 5)
                    print(f"DEBUG:     Gaussian {g_idx}: T_old sample (before blending) = {T_old[:sample_n].tolist()}")

                # Perform alpha compositing for selected pixels
                pixel_color[pixel_mask_for_gaus] += (T_old * alpha).unsqueeze(-1) * color_g
                pixel_T[pixel_mask_for_gaus] *= (1.0 - alpha)

                # Debug: 3. After accumulation (pixel_color, pixel_T)
                if not debug_print_done and g_idx < 5:
                    # Get indices of up to 5 affected pixels to inspect their new color and T values
                    affected_idx = pixel_mask_for_gaus.nonzero(as_tuple=False).squeeze()[:5]
                    pc_samples = pixel_color[affected_idx].tolist()
                    pt_samples = pixel_T[affected_idx].tolist()
                    print(f"DEBUG:     Gaussian {g_idx}: pixel_color samples after blending = {pc_samples}")
                    print(f"DEBUG:     Gaussian {g_idx}: pixel_T samples after blending = {pt_samples}")

        # End of chunk loop for this tile
        # Debug: 3. Summary after processing all Gaussians in the tile
        if not debug_print_done:
            # Compute min/max of pixel_color and pixel_T for this tile
            pc_min = pixel_color.min().item() if pixel_color.numel() > 0 else 0.0
            pc_max = pixel_color.max().item() if pixel_color.numel() > 0 else 0.0
            pt_min = pixel_T.min().item() if pixel_T.numel() > 0 else 0.0
            pt_max = pixel_T.max().item() if pixel_T.numel() > 0 else 0.0
            print(f"DEBUG: Tile {tile_idx} blending complete. pixel_color range = [{pc_min:.6f}, {pc_max:.6f}], "
                f"pixel_T range = [{pt_min:.6f}, {pt_max:.6f}]")

        # Write the accumulated color to the final image buffer
        final_r = torch.zeros_like(px_local_tile, dtype=torch.float32)
        final_g = torch.zeros_like(px_local_tile, dtype=torch.float32)
        final_b = torch.zeros_like(px_local_tile, dtype=torch.float32)
        final_r[valid_pixel_mask] = pixel_color[:, 0]
        final_g[valid_pixel_mask] = pixel_color[:, 1]
        final_b[valid_pixel_mask] = pixel_color[:, 2]
        start_y, start_x = int(ty) * int(tile_size), int(tx) * int(tile_size)
        ts = int(tile_size)
        image_buffer[start_y:start_y+ts, start_x:start_x+ts, 0] = final_r.reshape(ts, ts)
        image_buffer[start_y:start_y+ts, start_x:start_x+ts, 1] = final_g.reshape(ts, ts)
        image_buffer[start_y:start_y+ts, start_x:start_x+ts, 2] = final_b.reshape(ts, ts)

        # Debug: 4. Final pixel placement check (image_buffer values for this tile)
        if not debug_print_done:
            tile_pixels = image_buffer[start_y:start_y+ts, start_x:start_x+ts]
            tb_min = tile_pixels.min().item()
            tb_max = tile_pixels.max().item()
            print(f"DEBUG: Tile {tile_idx} written to image buffer. Output value range = [{tb_min:.6f}, {tb_max:.6f}].")
            # Example pixel values at [0,0] and center of the tile
            h, w, _ = tile_pixels.shape
            sample_y, sample_x = 0, 0
            center_y, center_x = h//2, w//2
            print(f"DEBUG: Sample pixel [0,0] in tile ({tx},{ty}) = {tile_pixels[sample_y, sample_x].tolist()}")
            print(f"DEBUG: Sample center pixel in tile ({tx},{ty}) = {tile_pixels[center_y, center_x].tolist()}")
            # Mark debug as done to avoid printing for other tiles
            debug_print_done = True
        
    print(f"\nSimulation Complete. Total saved Gaussians by hierarchical sorting & early termination")
    rendered_image = image_buffer.cpu().numpy()

    return rendered_image, metrics
