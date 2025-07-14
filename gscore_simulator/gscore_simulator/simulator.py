# gscore_simulator/simulator.py

import time
import numpy as np
from copy import deepcopy
import torch
import torch.nn.functional as F
from gscore_simulator.culling_wo_bitmap import cull_and_convert
from gscore_simulator.sorting_wo_bitmap import hierarchical_sort_and_group
from gscore_simulator.rasterizer_wo_bitmap import rasterize_tiles
from gscore_simulator.utils import calculate_psnr, calculate_lpips
from gscore_simulator.structures import RenderMetrics


class GSCoreSimulator:
    """
    GSCore 하드웨어 가속기의 전체 파이프라인을 시뮬레이션하는 클래스.
    CCU, GSU, VRU의 각 단계를 순차적으로 호출하고 최종 결과와 성능을 보고합니다.
    """
    def __init__(self, config: dict):
        """
        시뮬레이터 초기화

        Args:
            config (dict): 시뮬레이션에 사용될 전역 설정값
                           (e.g., tile_size, obb_test_ratio_threshold, etc.)
        """
        self.config = config
        self.device = self.config.get('device', 'cuda:0')
        print("GSCore Simulator initialized with config:")
        for key, value in config.items():
            print(f"  - {key}: {value}")


    def render(self, camera, gaussian_model):
        """
        주어진 카메라 시점에서 3D 가우시안 모델을 렌더링합니다.

        Args:
            camera: 렌더링할 시점의 카메라 객체 (from gaussian-splatting)
            gaussian_model: 훈련된 3D 가우시안 모델 (from gaussian-splatting)
            gt_image (np.ndarray, optional): 품질 평가를 위한 Ground-Truth 이미지.

        Returns:
            tuple: (렌더링된 이미지, 성능 지표 객체)
        """
        # 원본 camera는 그대로 두고, 해상도만 절반으로 줄인 사본 생성
        #torch.cuda.empty_cache()
        camera_half = deepcopy(camera)
        #camera_half.image_width //= 2
        #camera_half.image_height //= 2
        metrics = RenderMetrics()

        # --- CCU (Culling and Conversion Unit) ---
        start_time = time.time()
        culled_gaussians, G_list, metrics = cull_and_convert(gaussian_model, camera_half, self.config)
        ccu_time = time.time() - start_time
        print(f"CCU finished in {ccu_time:.2f}s.")

        # --- GSU (Gaussian Sorting Unit) ---
        start_time = time.time()
        tile_to_chunks, metrics = hierarchical_sort_and_group(G_list, self.config, metrics)
        gsu_time = time.time() - start_time
        print(f"GSU finished in {gsu_time:.2f}s.")
        
        # --- VRU (Volume Rendering Unit) ---
        start_time = time.time()
        rendered_image, metrics = rasterize_tiles(tile_to_chunks, culled_gaussians, camera_half, self.config, self.device, metrics)
        vru_time = time.time() - start_time
        print(f"VRU finished in {vru_time:.2f}s.")

        return rendered_image, metrics