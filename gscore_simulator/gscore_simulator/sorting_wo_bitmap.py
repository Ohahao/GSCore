from collections import defaultdict
from tqdm import tqdm
import torch
import sys
from math import ceil
from statistics import mean

# rasterizer가 필요로 하는 완전한 Gaussian2D 객체를 만들기 위해 import 합니다.
from gscore_simulator.structures import Gaussian2D
from gscore_simulator.structures import RenderMetrics


def hierarchical_sort_and_group(
    G_list: dict,
    config: dict,
    metrics
):
    """
    GSU (Gaussian Sorting Unit) 시뮬레이션 함수 (비트맵 미사용 버전).
    - CCU의 결과물을 조합하여 완전한 Gaussian2D 객체 생성
    - 생성된 객체들을 교차하는 타일 ID 기준으로 그룹화
    - 각 타일 내에서 객체들을 깊이(depth) 순으로 정렬 후 청크로 분할
    """
    
    # GPU 디바이스 설정 (config에서 지정 가능)
    chunk_size = config.get('gsu_chunk_size', 256)

    print("Running GSU: Performing hierarchical sorting...")
    # 타일별 계층적 정렬
    tile_to_chunks = {}


    for tile_idx, data in tqdm(G_list.items(), total=len(G_list), desc="Chunking tiles"):
        origin = data["txty"]
        gauss_list = data["gaussians"]
        # depth 기준 오름차순 정렬
        gauss_list_sorted = sorted(gauss_list, key=lambda e: e["depth"])

        # 전체 청크 개수
        n_chunks = ceil(len(gauss_list_sorted) / chunk_size)
        chunks = []
        for c in range(n_chunks):
            start = c * chunk_size
            end   = start + chunk_size
            chunk_gauss = gauss_list_sorted[start:end]
            chunks.append({
                "origin": origin,
                "chunk_id": c,
                "gaussians": chunk_gauss,
            })

        tile_to_chunks[tile_idx] = chunks

    print(f"[SUMMARY] Total tiles: {len(tile_to_chunks)}")
    return tile_to_chunks, metrics
















'''
# 이전 버전
def hierarchical_sort_and_group(
    culled_gaussians: dict,
    G_list: dict,
    config: dict,
    metrics
):
    """
    GSU (Gaussian Sorting Unit) 시뮬레이션 함수 (비트맵 미사용 버전).
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

    # --- 1단계: flat된 (gauss_idx, tile_idx, depth) 정보로 재구성 ---
    print("Running GSU: Reconstructing intersection info (Bitmap excluded)...")
    gauss_idx = culled_gaussians["gauss_idx"]  # (P,)
    tile_idx  = culled_gaussians["tile_idx"]   # (P,)
    depth_vec = culled_gaussians["depth"]      # (P,)
    txty      = culled_gaussians["txty"]       # (P, 2)

    P = gauss_idx.shape[0]
    # 각 가우시안의 정보(깊이, 교차하는 타일 ID set)를 저장할 딕셔너리
    gaussians_info = defaultdict(lambda: {"depth": 0.0, "tiles": set()})
    tile_id_to_txty = {}  # tile_id - txty 매핑 딕셔너리

    # 한 쌍씩 순회하면서 가우시안별로 depth와 교차 타일  저장
    print(f" gaussian idx shape: {P}")
    print("Running Vectorized Aggregation...")
    

    for p in tqdm(range(P), desc="Aggregating gaussian info"):
        g = int(gauss_idx[p].item())
        t = int(tile_idx[p].item())
        d = float(depth_vec[p].item())
        x0, y0 = txty[p].tolist()

        # depth는 한 번만 저장하면 충분
        gaussians_info[g]["depth"] = d
        # tiles set에 교차하는 타일 ID 추가
        gaussians_info[g]["tiles"].add(t)

        # tile_id_to_txty 딕셔너리에 매핑 정보 저장 (중복 방지)
        if t not in tile_id_to_txty:
            tile_id_to_txty[t] = (x0, y0)


    print(f"Total unique gaussians to process: {len(gaussians_info)}")

    # --- 2단계: 완전한 Gaussian2D 객체 생성 ---
    print("Running GSU: Creating full Gaussian2D objects...")
    processed_gaussians = []
    for g_id, g_info in tqdm(gaussians_info.items(), desc="Creating Gaussian2D objects"):
        gaus_obj = Gaussian2D(
            source_id        = g_id,
            mean             = culled_gaussians["mean"][g_id],
            cov              = culled_gaussians["cov"][g_id],
            depth            = g_info["depth"],
            opacity          = culled_gaussians["opacity"][g_id],
            color_precomp    = culled_gaussians["colors_precomp"][g_id],
            obb_corners      = culled_gaussians["obb_corners"][g_id],
            obb_axes         = culled_gaussians["obb_axes"][g_id],
            tiles            = g_info["tiles"]  # 이제 'tiles'는 tile_id의 set 입니다.
        )
        processed_gaussians.append(gaus_obj)

    # --- 3단계: 타일 ID를 기준으로 Gaussian2D 객체 그룹화 ---
    print("Running GSU: Grouping Gaussian2D objects by tile...")
    gaussians_by_tile = defaultdict(list)

    for gaus_obj in tqdm(processed_gaussians, desc="Grouping by tile"):
        # 수정: gaus_obj.tiles는 이제 tile_id의 set이므로, 바로 순회합니다.
        for tile_id in gaus_obj.tiles:
            gaussians_by_tile[tile_id].append(gaus_obj)

    # --- 4단계: 통계 계산 및 계층적 정렬 ---
    # (이 부분은 로직 변경 없이 이전과 동일하게 동작합니다)
    if 42 in gaussians_by_tile:
        tx, ty = tile_id_to_txty[42]
        gaussians_per_tile = len(gaussians_by_tile[42])
        metrics.gaussians_per_tile = gaussians_per_tile
        metrics.tile_coords = (tx, ty)
        print(f"\n[DEBUG] Tile ID 42 at coordinates ({tx}, {ty}) intersects with {gaussians_per_tile} Gaussians.")

    tx_, ty_ = metrics.tile_coords
    print(f"[METRICS] Tile ({tx_}, {ty_}) has {metrics.gaussians_per_tile} gaussians.")

    counts_per_tile = [len(gaussians) for gaussians in gaussians_by_tile.values()]
    metrics.avg_gaussians_per_tile = mean(counts_per_tile) if counts_per_tile else 0
    metrics.max_gaussians_per_tile = max(counts_per_tile) if counts_per_tile else 0
    metrics.macs_per_tile = MAC_PER_BLENDING * metrics.gaussians_per_tile
    print(f"[METRICS] average gaussians per tile: {metrics.avg_gaussians_per_tile:.2f} gaussians.")
    print(f"[METRICS] Maximum gaussians per tile: {metrics.max_gaussians_per_tile} gaussians.")
    
    print("Running GSU: Performing hierarchical sorting...")
    sorted_and_chunked_gaussians = {}
    total_gaussians = 0
    total_chunks = 0

    # 타일별 계층적 정렬
    for tile_id, gaus_list in tqdm(gaussians_by_tile.items(), desc="Sorting & chunking"):
        # tile n에서 겹치는 가우시안 정렬 (깊이 기준)
        gaus_list.sort()

        # tile n에서 겹치는 가우시안을 청크로 분해
        chunks = [
            gaus_list[i : i + chunk_size]
            for i in range(0, len(gaus_list), chunk_size)
        ]

        # 누적 카운트
        #total_gaussians += len(gaus_list)
        total_chunks += len(chunks)

        # tile_id -> (tx, ty) 매핑
        if tile_id in tile_id_to_txty:
            tx, ty = tile_id_to_txty[tile_id]
        else:
            # 이 경우는 발생하지 않아야 합니다.
            print(f"WARNING: No txty found for tile_id {tile_id}. Skipping.")
            continue 
        
        if chunks:
            sorted_and_chunked_gaussians[tile_id] = {
                "txty": (tx, ty),
                "chunks": chunks
            }

    #print(f"[SUMMARY] Total gaussians after sorting/chunking: {total_gaussians}")
    print(f"[SUMMARY] Total chunks: {total_chunks}")
    print(f"[SUMMARY] Total tiles with gaussians: {len(sorted_and_chunked_gaussians)}")

    return sorted_and_chunked_gaussians, tile_id_to_txty, metrics
    '''