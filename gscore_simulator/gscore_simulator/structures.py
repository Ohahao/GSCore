# gscore_simulator/structures.py

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class RenderMetrics:
    """렌더링 과정의 성능 지표를 저장하는 데이터 클래스"""
    gaussians_per_tile: List[int] = None
    # 기준이 되는 타일 시작 좌표
    tile_coords: List[Tuple[int, int]] = None
    tile_idxs: List[int] = None
    mac_per_tile: List[int] = 0
    avg_gaussians_per_tile: int = 0
    max_gaussians_per_tile: int = 0
    macs_per_tile: int = 0
    avg_saved_gaussians: float = 0.0
    avg_saved_rate: float = 0.0
    avg_calculated_gaussians: float = 0.0
    first_saved_gaussians: float = 0.0
    first_saved_rate: float = 0.0
    first_calculated_gaussians: float = 0.0
    first_terminated_coords: Tuple[int, int] = None
    total_blending_operations: int = 0
    # PSNR, LPIPS 등 최종 품질 지표
    quality_metrics: dict = field(default_factory=dict)

@dataclass
class Gaussian2D:
    """
    3D 가우시안이 2D 이미지 평면에 투영된 후의 정보를 담는 클래스.
    hierarchical_sort_and_group() 에서 생성·정렬·그룹핑에 사용됩니다.
    """
    # 원본 3D 가우시안의 인덱스
    source_id: int

    # 2D 평면에서의 중심 좌표 (x, y)
    mean: torch.Tensor            # shape: (2,)

    # 2D 공분산 행렬 (형태: cov2d)
    cov: torch.Tensor             # shape: (2,2) 혹은 상삼각 요소 등

    # 깊이 값 (정렬에 사용)
    depth: float

    # 불투명도
    opacity: float

    # Spherical Harmonics 계수 (색상 계산에 사용)
    color_precomp: torch.Tensor   # shape: (...,)

    # 각 교차 타일에 대한 subtile-level 비트맵
    # { tile_id: {"bitmap": tensor(shape=(S,)), "start": (x0,y0)} }
    tiles: Dict[int, Dict[str, any]]

    def __lt__(self, other: "Gaussian2D"):
        """깊이(depth)를 기준으로 오름차순 정렬"""
        return self.depth < other.depth