from collections import defaultdict
from tqdm import tqdm
import torch
import sys

# rasterizer가 필요로 하는 완전한 Gaussian2D 객체를 만들기 위해 import 합니다.
from gscore_simulator.structures import Gaussian2D
from gscore_simulator.structures import RenderMetrics
from statistics import mean

def hierarchical_sort_and_group(
    culled_gaussians: dict,
    config: dict,
    metrics
):
    """
    GSU (Gaussian Sorting Unit) 시뮬레이션 함수.
    - CCU의 결과물을 조합하여 완전한 Gaussian2D 객체 생성
    - 생성된 객체들을 교차하는 타일 ID 기준으로 그룹화
    - 각 타일 내에서 객체들을 깊이(depth) 순으로 정렬 후 청크로 분할
    """
    MAC_PER_BLENDING = 4
    # GPU 디바이스 설정 (config에서 지정 가능)
    device = torch.device(config.get('device', 'cuda'))
    chunk_size = config.get('gsu_chunk_size', 256)

    # --- 0) culled_gaussians 안의 GPU 텐서 보장 ---
    for k, v in culled_gaussians.items():
        if not torch.is_tensor(v):
            culled_gaussians[k] = torch.as_tensor(v, device=device)
        else:
            culled_gaussians[k] = v.to(device)

    # --- 1단계: flat된 (gauss_idx, tile_idx, bitmap, depth) 정보로 재구성 ---
    print("Running GSU: Reconstructing intersection info...")
    gauss_idx = culled_gaussians["gauss_idx"]    # (P,)
    tile_idx  = culled_gaussians["tile_idx"]     # (P,)
    bitmap    = culled_gaussians["bitmap"]       # (P, S)
    depth_vec = culled_gaussians["depth"]        # (P,)
    txty      = culled_gaussians["txty"]

    P = gauss_idx.shape[0]
    gaussians_info = defaultdict(lambda: {"depth": 0.0, "tiles": {}})
    tile_id_to_txty = {}    #tile_id - txty 매핑 딕셔너리
    # 한 쌍씩 순회하면서 가우시안별로 depth와 subtile bitmap 저장
    for p in tqdm(range(P), desc="Aggregating gaussian bitmaps"):
        g = int(gauss_idx[p].item())
        t = int(tile_idx[p].item())
        b = bitmap[p]      # (S,) subtile-level 비트맵
        d = float(depth_vec[p].item())
        x0, y0 = txty[p].tolist()

        # depth는 하나만 저장해 두면 충분합니다
        gaussians_info[g]["depth"] = d
        # tiles 맵: key=tile_id, value=subtile 비트맵
        gaussians_info[g].setdefault("tiles", {})[t] = {
        "bitmap": b,
        "txty":  (x0, y0)
        }
        # tile_id_to_txty 딕셔너리에 매핑 정보 저장
        tile_id_to_txty[t] = (x0, y0)

    print(f"gaussian info (depth): {len(gaussians_info)}")

    # --- 이하 기존 2~4단계는 변경 없습니다 ---
    print("Running GSU: Creating full Gaussian2D objects...")
    processed_gaussians = []
    for g_id, gaussians_info in tqdm(gaussians_info.items()):
        gaus_obj = Gaussian2D(
            source_id         = g_id,
            mean              = culled_gaussians["mean"][g_id],
            cov               = culled_gaussians["cov"][g_id],
            depth             = gaussians_info["depth"],
            opacity           = culled_gaussians["opacity"][g_id],
            color_precomp     = culled_gaussians["colors_precomp"][g_id],
            obb_corners       = culled_gaussians["obb_corners"][g_id],
            obb_axes          = culled_gaussians["obb_axes"][g_id],
            tiles             = gaussians_info["tiles"]
        )
        processed_gaussians.append(gaus_obj)

    print("Running GSU: Grouping Gaussian2D objects by tile...")
    gaussians_by_tile = defaultdict(list)

    for gaus_obj in tqdm(processed_gaussians, desc="Grouping by tile"):
        # 수정: intersecting_tiles 대신 gaus_obj.tiles의 키 사용
        for tile_id in gaus_obj.tiles.keys():
            gaussians_by_tile[tile_id].append(gaus_obj)
            tx, ty = tile_id_to_txty[tile_id]

            if tile_id == 42:   #shapeGS와 동일 타일에서 비교
                gaussians_per_tile = len(gaussians_by_tile[tile_id])
                metrics.gaussians_per_tile = gaussians_per_tile
                metrics.tile_coords = (tx, ty)
                print(f"\n[DEBUG] Tile ID {tile_id} at coordinates ({tx}, {ty}) intersects with {gaussians_per_tile} Gaussians.")

    
    tx_, ty_ = metrics.tile_coords
    print(f"[METRICS] Tile ({tx_}, {ty_}) has {metrics.gaussians_per_tile} gaussians.")

    counts_per_tile = [len(gaussians) for gaussians in gaussians_by_tile.values()]
    metrics.avg_gaussians_per_tile = mean(counts_per_tile) if counts_per_tile else 0
    metrics.max_gaussians_per_tile = max(counts_per_tile) if counts_per_tile else 0
    metrics.macs_per_tile = MAC_PER_BLENDING * metrics.gaussians_per_tile
    print(f"[METRICS] average gaussians per tile: {metrics.avg_gaussians_per_tile} gaussians.")
    print(f"[METRICS] Maximum gaussians per tile: {metrics.max_gaussians_per_tile} gaussians.")
    print("Running GSU: Performing hierarchical sorting...")
    sorted_and_chunked_gaussians = {}
    total_gaussians = 0
    total_chunks = 0

    #tile 별 hirarchical sorting
    for tile_id, gaus_list in tqdm(gaussians_by_tile.items(), desc="Sorting & chunking"):
    # tile n에서 겹치는 가우시안 정렬(precise sorting)
        gaus_list.sort()

        # tile n에서 겹치는 가우시안 chunk 분해(approximate sorting)
        chunks = [
            gaus_list[i : i + chunk_size]
            for i in range(0, len(gaus_list), chunk_size)
        ]

        # 누적 카운트
        total_gaussians += len(gaus_list)
        total_chunks += len(chunks)

        # tile_id → (tx, ty) 매핑
        if tile_id in tile_id_to_txty:
            tx, ty = tile_id_to_txty[tile_id]
        else:
            print(f"WARNING: No txty found for tile_id {tile_id}. Skipping.")
            continue 
        
        if chunks:
            sorted_and_chunked_gaussians[tile_id] = {
                "txty": (tx, ty),
                "chunks": chunks
            }

    print(f"[SUMMARY] Total gaussians after sorting/chunking: {total_gaussians}")
    print(f"[SUMMARY] Total chunks: {total_chunks}")
    print(f"    sorted_and_chunked gaussians: {len(sorted_and_chunked_gaussians)}")

    return sorted_and_chunked_gaussians, tile_id_to_txty, metrics