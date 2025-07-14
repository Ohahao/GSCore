# run_gscore_simulation.py (수정 버전)

import os
import argparse
from argparse import ArgumentParser
import numpy as np
from PIL import Image
import json
from dataclasses import dataclass, asdict
import torch # torch 임포트 추가
import torchvision.utils as vutils

# --- 기존 gaussian-splatting 코드의 유틸리티 임포트 ---
from gaussian_splatting.scene import Scene, GaussianModel
from gaussian_splatting.utils.general_utils import safe_state
#from utils.system_utils import setup_logger
# ----------------------------------------------------

from gscore_simulator.simulator import GSCoreSimulator
# utils.py 함수들을 직접 임포트하여 device 인자를 전달할 수 있게 함
from gscore_simulator import utils as gscore_utils 

def tensor_to_builtin(o):
    if isinstance(o, torch.Tensor):
        if o.numel() == 1:
            return o.item()       # 스칼라 텐서 → 숫자
        else:
            return o.tolist()     # 다차원 텐서 → 리스트
    raise TypeError(f"Type {o.__class__.__name__} not serializable")

def run_simulation(args):
    """GSCore 시뮬레이션 전체 과정을 실행하고 결과를 저장합니다."""
    
    # 지정된 디바이스 설정
    device = torch.device(args.device)
    
    gscore_config = {
        "tile_size": 16,
        "subtile_res": 4, 
        "obb_radius_scale": 3.0,
        "obb_test_ratio_threshold": 2.0,
        "gsu_chunk_size": 256,
        "device": args.device # 설정에 디바이스 추가
    }
    
    #setup_logger()
    gaussians = GaussianModel(args.sh_degree, device=args.device)
    
    # Scene 클래스가 사용할 디바이스를 지정
    args.data_device = args.device 
    setattr(args, 'images', 'images')
    setattr(args, 'depths', '')
    setattr(args, 'eval', True) # 테스트 카메라를 사용하므로 True
    setattr(args, 'train_test_exp', False)
    setattr(args, 'resolution', -1)
    device = args.data_device
    source_path = args.source_path
    scene_name = os.path.basename(os.path.normpath(source_path))

    print("\n=== Args passed to Scene ===")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("============================\n")

    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False, device=device)
    
    view_idx = args.view_index
    if view_idx >= len(scene.getTestCameras()):
        print(f"Error: View index {view_idx} is out of bounds for test set size {len(scene.getTestCameras())}")
        return
    camera = scene.getTestCameras()[view_idx]
    
    gt_image = None
    image_dir = os.path.join(args.source_path, getattr(args, 'images', 'images'))
    gt_image_path = os.path.join(image_dir, camera.image_name)
    print(f" \n gt image path: {gt_image_path}")
    

    # GSCore 시뮬레이터 실행
    simulator = GSCoreSimulator(gscore_config)
    rendered_image, metrics = simulator.render(camera, gaussians)
    
    # 이미지 저장(gt image, rendered image)
    os.makedirs(args.output_dir, exist_ok=True)
    rendered_path = os.path.join(args.output_dir, f"gscore_render_{args.iteration}_{scene_name}_view{view_idx}.png")
    rendered_image = torch.from_numpy(rendered_image).to(device)
    vutils.save_image(rendered_image, rendered_path)

    # gt image 로드
    try:
        
        gt_image_pil = Image.open(gt_image_path)
        gt_image = np.array(gt_image_pil, dtype=np.float32) / 255.0
        if gt_image.shape[2] == 4:
            gt_image = gt_image[:, :, :3]
        
        gt_image_pil = Image.open(gt_image_path).convert("RGB")
        W, H = gt_image_pil.size
        gt_image_pil = gt_image_pil.resize((rendered_image.shape[2], rendered_image.shape[1]), Image.BILINEAR)
        gt_image = np.array(gt_image_pil, dtype=np.float32) / 255.0
    except FileNotFoundError:
        print(f"Warning: Ground truth image not found. PSNR/LPIPS will not be calculated.")

    print(f"\nRendered image saved to: {rendered_path}")

    print("\n--- GSCore Simulation Report ---")
    # ... (이하 결과 출력 부분은 동일) ...
    print(f"Rendered Image Size: {rendered_image.shape[2]}x{rendered_image.shape[1]}")
    psnr = gscore_utils.calculate_psnr(rendered_image, gt_image)
    lpips = gscore_utils.calculate_lpips(rendered_image, gt_image)
    metrics.quality_metrics = {"PSNR": psnr, "LPIPS": lpips}
    if metrics.quality_metrics:
        print(f"PSNR: {metrics.quality_metrics['PSNR']:.4f} dB")
        print(f"LPIPS: {metrics.quality_metrics['LPIPS']:.4f}")
        
    print("\n--- Performance Metrics ---")
    print(f"Number of gaussians at {metrics.tile_coords}: {metrics.gaussians_per_tile}")
    print(f"Avg Gaussians Per Tile: {metrics.avg_gaussians_per_tile:.2f}")
    print(f"Max Gaussians Per Tile: {metrics.max_gaussians_per_tile}")
    print(f"MAC Per Tile: {metrics.macs_per_tile}")
    print(f"Tile Coords (example): {metrics.tile_coords}")
    print(f"[METRICS] Total pixel-blending operations: {metrics.total_blending_operations:,}")
    #print(f"Alpha Blending Operations: {metrics.alpha_blending_ops:,}")
    #print(f"Estimated MAC Operations: {metrics.mac_operations:,}")
  

    # Save metrics to a JSON file
    metrics_dir = "/home/hyoh/GSCore/gscore_renders/metrics"
    metrics_output_path = os.path.join(metrics_dir, f"gscore_metrics_{args.iteration}_{scene_name}_view{view_idx}.json")
    
    # asdict를 사용하여 RenderMetrics 인스턴스를 딕셔너리로 변환
    # dataclass 필드의 타입에 맞게 자동으로 변환됨
    metrics_to_save = asdict(metrics)

    # gaussians_per_tile이 큰 경우 JSON 파일 크기가 커질 수 있으므로,
    # 필요에 따라 이 필드를 저장할지 여부를 결정할 수 있습니다.
    # 여기서는 예시로 포함하여 저장합니다.
    
    with open(metrics_output_path, "w") as f:
        json.dump(metrics_to_save, f, default=tensor_to_builtin, indent=4)
    print(f"Metrics saved to: {metrics_output_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Run GSCore simulation on a 3D Gaussian Splatting scene.")
    # ... (기존 인자) ...
    parser.add_argument("--model_path", "-m", required=True, help="Path to the model directory (output of training).")
    parser.add_argument("--source_path", "-s", required=True, help="Path to the dataset source folder.")
    parser.add_argument("--iteration", "-i", default=30000, type=int, help="Iteration number of the model to load.")
    parser.add_argument("--output_dir", "-o", default="./gscore_renders/images", help="Directory to save rendered images.")
    parser.add_argument("--view_index", default=0, type=int, help="Index of the test camera view to render.")
    parser.add_argument("--sh_degree", default=3, type=int, help="SH degree of the Gaussian model.")

    # --- GPU 장치 지정을 위한 인자 추가 ---
    parser.add_argument("--device", default="cuda:0", help="Device to use for PyTorch operations (e.g., 'cuda:0', 'cuda:1', 'cpu').")
    
    args = parser.parse_args()
    args.white_background = False
    # data_device는 위에서 설정되므로 여기서는 제거
    
    # 'Scene' 클래스 생성 시 반복 횟수를 직접 전달하도록 수정
    # (원래 코드베이스의 Scene 생성자 변경에 따른 대응)
    # 이는 원래 코드의 버전에 따라 유연하게 조정 필요
    # setattr(args, 'load_iteration', args.iteration) 

    run_simulation(args)




'''

def run_simulation(args):
    """GSCore 시뮬레이션 전체 과정을 실행하고 결과를 저장합니다."""
    
    # 지정된 디바이스 설정
    device = torch.device(args.device)
    
    gscore_config = {
        "tile_size": 16,
        "subtile_res": 4, 
        "obb_radius_scale": 3.0,
        "obb_test_ratio_threshold": 2.0,
        "gsu_chunk_size": 256,
        "device": args.device # 설정에 디바이스 추가
    }
    
    #setup_logger()
    gaussians = GaussianModel(args.sh_degree, device=args.device)
    
    # Scene 클래스가 사용할 디바이스를 지정
    args.data_device = args.device 
    setattr(args, 'images', 'images')
    setattr(args, 'depths', '')
    setattr(args, 'eval', True) # 테스트 카메라를 사용하므로 True
    setattr(args, 'train_test_exp', False)
    setattr(args, 'resolution', -1)
    device = args.data_device
    source_path = args.source_path
    scene_name = os.path.basename(os.path.normpath(source_path))

    print("\n=== Args passed to Scene ===")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("============================\n")

    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False, device=device)
    
    view_idx = args.view_index
    if view_idx >= len(scene.getTestCameras()):
        print(f"Error: View index {view_idx} is out of bounds for test set size {len(scene.getTestCameras())}")
        return
    camera = scene.getTestCameras()[view_idx]
    
    gt_image = None
    image_dir = os.path.join(args.source_path, getattr(args, 'images', 'images'))
    gt_image_path = os.path.join(image_dir, camera.image_name)
    print(f" \n gt image path: {gt_image_path}")
    try:
        
        gt_image_pil = Image.open(gt_image_path)
        gt_image = np.array(gt_image_pil, dtype=np.float32) / 255.0
        if gt_image.shape[2] == 4:
            gt_image = gt_image[:, :, :3]
        
        gt_image_pil = Image.open(gt_image_path).convert("RGB")
        W, H = gt_image_pil.size
        gt_image_pil = gt_image_pil.resize((W // 2, H // 2), Image.BILINEAR)
        gt_image = np.array(gt_image_pil, dtype=np.float32) / 255.0
    except FileNotFoundError:
        print(f"Warning: Ground truth image not found. PSNR/LPIPS will not be calculated.")
    
    # GSCore 시뮬레이터 실행 (이 시뮬레이터 자체는 대부분 CPU에서 실행됨)
    simulator = GSCoreSimulator(gscore_config)
    
    # 렌더링 시작 전, 모델과 GT 이미지를 numpy로 변환
    # GSCore 시뮬레이터는 numpy 배열을 입력으로 받기 때문
    rendered_image, metrics = simulator.render(camera, gaussians, gt_image)

    # 최종 품질 평가는 지정된 GPU에서 수행
    if gt_image is not None:
        print(f"Calculating final quality metrics on {device}...")
        psnr = gscore_utils.calculate_psnr(rendered_image, gt_image)
        lpips = gscore_utils.calculate_lpips(rendered_image, gt_image, device)
        metrics.quality_metrics = {"PSNR": psnr, "LPIPS": lpips}

    print("\n--- GSCore Simulation Report ---")
    # ... (이하 결과 출력 부분은 동일) ...
    print(f"Rendered Image Size: {rendered_image.shape[1]}x{rendered_image.shape[0]}")
    
    if metrics.quality_metrics:
        print(f"PSNR: {metrics.quality_metrics['PSNR']:.4f} dB")
        print(f"LPIPS: {metrics.quality_metrics['LPIPS']:.4f}")
        
    print("\n--- Performance Metrics ---")
    print(f"Saved Gaussians Per Tile: {metrics.saved_gaussians_per_tile}")
    print(f"Number of gaussians at {metrics.tile_coords}: {metrics.gaussians_per_tile}")
    print(f"Avg Gaussians Per Tile: {metrics.avg_gaussians_per_tile:.2f}")
    print(f"Max Gaussians Per Tile: {metrics.max_gaussians_per_tile}")
    print(f"MAC Per Tile: {metrics.macs_per_tile}")
    print(f"Tile Coords (example): {metrics.tile_coords}")
    print(f"\n[METRICS] Total pixel-blending operations: {metrics.total_blending_operations:,}")
    #print(f"Alpha Blending Operations: {metrics.alpha_blending_ops:,}")
    #print(f"Estimated MAC Operations: {metrics.mac_operations:,}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"gscore_render_{args.iteration}_{scene_name}_view{view_idx}.png")
    
    rendered_image_uint8 = (np.clip(rendered_image, 0, 1) * 255).astype(np.uint8)
    pil = Image.fromarray(rendered_image_uint8)
    pil = pil.resize((W, H), resample=Image.BILINEAR) #-> 해상도 줄였을 떄만
    pil.save(output_path)
    #Image.fromarray(rendered_image_uint8).save(output_path)
    print(f"\nRendered image saved to: {output_path}")

    # Save metrics to a JSON file
    metrics_output_path = os.path.join(args.output_dir, f"gscore_metrics_{args.iteration}_{scene_name}_view{view_idx}.json")
    
    # asdict를 사용하여 RenderMetrics 인스턴스를 딕셔너리로 변환
    # dataclass 필드의 타입에 맞게 자동으로 변환됨
    metrics_to_save = asdict(metrics)

    # gaussians_per_tile이 큰 경우 JSON 파일 크기가 커질 수 있으므로,
    # 필요에 따라 이 필드를 저장할지 여부를 결정할 수 있습니다.
    # 여기서는 예시로 포함하여 저장합니다.
    
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
    print(f"Metrics saved to: {metrics_output_path}")

'''