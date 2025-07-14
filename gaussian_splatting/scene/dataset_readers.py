#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool
    data_device: str

'''
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool
    train_frames: list = None
    test_frames: list = None
'''
class SceneInfo:
    def __init__(self, point_cloud, train_cameras, test_cameras, nerf_normalization, ply_path, is_nerf_synthetic, device):
        self.point_cloud = point_cloud
        self.train_cameras = train_cameras
        self.test_cameras = test_cameras
        self.nerf_normalization = nerf_normalization
        self.ply_path = ply_path
        self.is_nerf_synthetic = is_nerf_synthetic
        self.train_frames = None
        self.test_frames = None
        self.device = device

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}
    
    
def get_center(cam_info):
  def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal
        
  cam_centers = []

  for cam in cam_info:
      W2C = getWorld2View2(cam.R, cam.T)
      C2W = np.linalg.inv(W2C)
      cam_centers.append(C2W[:3, 3:4])

  center, diagonal = get_center_and_diag(cam_centers)
  
  return center
  

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list, device):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        # n_remove는 파일 확장자를 제거하기 위함
        n_remove = len(extr.name.split('.')[-1]) + 1 if '.' in extr.name else 0
        depth_params = None
        if depths_params is not None:
            try:
                # 파일 확장자를 제외한 이름으로 depth 파라미터 검색
                depth_params_key = extr.name[:-n_remove] if n_remove > 0 else extr.name
                depth_params = depths_params[depth_params_key]
            except KeyError:
                 print(f"\nWarning: {extr.name} not found in depths_params, proceeding without it.")


        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = ""
        if depths_folder:
            depth_key = extr.name[:-n_remove] if n_remove > 0 else extr.name
            depth_path = os.path.join(depths_folder, f"{depth_key}.png")

        # [수정] CameraInfo 객체 생성 시 device 인자를 전달합니다.
        # CameraInfo 클래스의 __init__ 메서드가 이 값을 받아 내부 텐서를 생성합니다.
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list,
                              data_device=device) # <<< 이 부분 추가
        
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, depths, eval, train_test_exp, device: str = "cuda:0", llffhold=8):
    """
    COLMAP 데이터를 읽어 SceneInfo 객체를 생성하는 함수 (device 지정 최종 버전)

    Args:
        path (str): 데이터셋 루트 경로.
        images (str): 이미지 폴더 이름.
        depths (str): 깊이 폴더 이름.
        eval (bool): 평가 모드 여부.
        train_test_exp (bool): 특정 실험을 위한 플래그.
        device (str, optional): 텐서를 생성할 장치. 기본값은 "cuda:0".
        llffhold (int, optional): LLFF 데이터셋 테스트 분할 주기. 기본값은 8.
    """
    print(f"Reading COLMAP scene info with device: {device}")
    
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    depths_params = None
    if depths and depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale
        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            test_txt_path = os.path.join(path, "sparse/0", "test.txt")
            if os.path.exists(test_txt_path):
                 with open(test_txt_path, 'r') as file:
                    test_cam_names_list = [line.strip() for line in file]
            else:
                # 360 데이터셋의 경우 test.txt가 없을 수 있음
                test_cam_names_list = []
    else:
        test_cam_names_list = []

    reading_dir = "images" if images is None else images
    
    # [수정] readColmapCameras 헬퍼 함수에 device 인자 전달
    # 이 함수 내부에서 Camera 객체를 생성할 때 device를 사용해야 합니다.
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=os.path.join(path, depths) if depths and depths != "" else "", 
        test_cam_names_list=test_cam_names_list,
        device=device  # device 전달
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    # [수정] getNerfppNorm 헬퍼 함수에 device 인자 전달
    # 이 함수가 반환하는 정규화 정보(텐서)가 올바른 장치에 생성되도록 합니다.
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False,
                           device=device)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png", device='cuda:0'):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx     = contents.get("camera_angle_x", None)

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            file_path = frame["file_path"].lstrip("./")  # 문자열의 처음에 있는 '.' 제거
            name, ext = os.path.splitext(file_path)
            if ext == "":
                file_path = name + extension   
            else:
                file_path = os.path.join('images', file_path)
            cam_name = os.path.join(path, file_path)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""
            

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, 
                            depth_params=None, is_test=is_test, data_device=device))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, dataset_type, extension=".png", device='cuda:0'):

    if dataset_type == "nerf_synthetic":
        depths_folder=os.path.join(path, depths) if depths != "" else ""

        # nerf synthetic dataset
        print("Reading nerf synthetic dataset")
        train_transforms_path = os.path.join(path, "transforms_train.json")
        with open(train_transforms_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        train_frames = train_data.get("frames", [])
        
        test_transforms_path = os.path.join(path, "transforms_test.json")
        with open(test_transforms_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        test_frames = test_data.get("frames", [])

        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
        print("Reading Test Transforms")
        test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    elif dataset_type == "omniblender":
        depths_folder=os.path.join(path, depths) if depths != "" else ""

        # OmniBlender dataset
        print("Reading OmniBlender dataset")
        transforms_path = os.path.join(path, "transform.json")
        with open(transforms_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        train_frames = [frame for frame in data["frames"] if int(frame["file_path"].split(".")[0]) % 4 == 0]
        test_frames = [frame for frame in data["frames"] if int(frame["file_path"].split(".")[0]) % 4 == 2]

        print("Reading Training Transforms in OmniBlender")
        train_cam_infos = readCamerasFromTransforms(path, "transform.json", depths_folder, white_background, False, extension)
        print("Reading Test Transforms in OmniBlender")
        test_cam_infos = readCamerasFromTransforms(path, "transform.json", depths_folder, white_background, True, extension)

    if not eval:
       train_cam_infos.extend(test_cam_infos)
       test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True,
                           device=device)

    #return 값에 frame 정보 추가
    scene_info.train_frames = train_frames
    scene_info.test_frames = test_frames

    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}