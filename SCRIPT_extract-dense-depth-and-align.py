__author__ = 'Xuanli CHEN'
"""
Xuanli Chen
Research Domain: Computer Vision, Machine Learning
Email: xuanli(dot)chen(at)icloud.com
LinkedIn: https://be.linkedin.com/in/xuanlichen
"""
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

import struct
import os
import sys

import torch
# Set Torch Hub to /d_disk/torch_hub
torch.hub.set_dir("/d_disk/torch_hub")

def extract_sparse_depth_and_point_from_reconstruction(reconstruction):
    """
    Extracts sparse depth and 3D points from the reconstruction.

    Args:
        predictions (dict): Contains reconstruction data with a 'reconstruction' key.

    Returns:
        dict: Updated predictions with 'sparse_depth' and 'sparse_point' keys.
    """

    sparse_depth = defaultdict(list)
    sparse_point = defaultdict(list)
    # Extract sparse depths from SfM points
    for point3D_idx in reconstruction.points3D:
        pt3D = reconstruction.points3D[point3D_idx]
        for track_element in pt3D.track.elements:
            pyimg = reconstruction.images[track_element.image_id]
            pycam = reconstruction.cameras[pyimg.camera_id]
            img_name = pyimg.name
            projection = pyimg.cam_from_world * pt3D.xyz
            depth = projection[-1]
            # NOTE: uv here cooresponds to the (x, y)
            # at the original image coordinate
            # instead of the padded&resized one
            uv = pycam.img_from_cam(projection)
            sparse_depth[img_name].append(np.append(uv, depth))
            sparse_point[img_name].append(np.append(pt3D.xyz, point3D_idx))

    return sparse_depth, sparse_point


def extract_dense_depth_maps(depth_model, image_paths, dp_depth):
    """
    Extract dense depth maps from a list of image paths
    Note that the monocular depth model outputs disp instead of real depth map
    """

    print("Extracting dense depth maps")
    disp_dict = {}
    original_images = {}
    for idx in tqdm(
            range(len(image_paths)), desc="Predicting monocular depth maps"
    ):

        fp_output = dp_depth / (image_paths[idx].stem + ".bin")

        if fp_output.exists():
            disp_map = read_array(fp_output)
            disp_dict[image_paths[idx].name] = disp_map
            continue

        img_fname = image_paths[idx].as_posix()
        basename = os.path.basename(img_fname)


        raw_img = cv2.imread(img_fname)
        original_images[basename] = raw_img

        # raw resolution
        disp_map = depth_model.infer_image(
            raw_img, min(1024, max(raw_img.shape[:2]))
        )

        disp_dict[basename] = disp_map
        write_array(disp_map, fp_output)

    print("Monocular depth maps complete. Depth alignment to be conducted.")
    return disp_dict, original_images


def align_dense_depth_maps(
        reconstruction,
        sparse_depth,
        disp_dict,
        original_images,
):
    """
    https://www.notion.so/VGGSfM-2-0-Try-1058ff730465803da77cc61e5f841323?pvs=4
    """
    # For dense depth estimation
    from sklearn.linear_model import RANSACRegressor
    from sklearn.linear_model import LinearRegression

    # Define disparity and depth limits
    disparity_max = 10000
    disparity_min = 0.0001
    depth_max = 1 / disparity_min
    depth_min = 1 / disparity_max

    depth_dict = {}
    unproj_dense_points3D = {}
    fname_to_id = {
        reconstruction.images[imgid].name: imgid
        for imgid in reconstruction.images
    }

    for img_basename in tqdm(
            sparse_depth, desc="Load monocular depth and Align"
    ):
        sparse_uvd = np.array(sparse_depth[img_basename])

        if len(sparse_uvd) <= 0:
            raise ValueError("Too few points for depth alignment")

        disp_map = disp_dict[img_basename]

        ww, hh = disp_map.shape
        # Filter out the projections outside the image
        int_uv = np.round(sparse_uvd[:, :2]).astype(int)
        maskhh = (int_uv[:, 0] >= 0) & (int_uv[:, 0] < hh)
        maskww = (int_uv[:, 1] >= 0) & (int_uv[:, 1] < ww)
        mask = maskhh & maskww
        sparse_uvd = sparse_uvd[mask]
        int_uv = int_uv[mask]

        # Nearest neighbour sampling
        sampled_disps = disp_map[int_uv[:, 1], int_uv[:, 0]]

        # Note that dense depth maps may have some invalid values such as sky
        # they are marked as 0, hence filter out 0 from the sampled depths
        positive_mask = sampled_disps > 0
        sampled_disps = sampled_disps[positive_mask]
        sfm_depths = sparse_uvd[:, -1][positive_mask]

        sfm_depths = np.clip(sfm_depths, depth_min, depth_max)

        thres_ratio = 30
        target_disps = 1 / sfm_depths

        # RANSAC
        X = sampled_disps.reshape(-1, 1)
        y = target_disps
        ransac_thres = np.median(y) / thres_ratio

        if ransac_thres <= 0:
            raise ValueError("Ill-posed scene for depth alignment")

        ransac = RANSACRegressor(
            LinearRegression(),
            min_samples=2,
            residual_threshold=ransac_thres,
            max_trials=20000,
            loss="squared_error",
        )
        ransac.fit(X, y)
        scale = ransac.estimator_.coef_[0]
        shift = ransac.estimator_.intercept_
        # inlier_mask = ransac.inlier_mask_

        nonzero_mask = disp_map != 0

        # Rescale the disparity map
        disp_map[nonzero_mask] = disp_map[nonzero_mask] * scale + shift

        valid_depth_mask = (disp_map > 0) & (disp_map <= disparity_max)
        disp_map[~valid_depth_mask] = 0

        # Convert the disparity map to depth map
        depth_map = np.full(disp_map.shape, np.inf)
        depth_map[disp_map != 0] = 1 / disp_map[disp_map != 0]
        depth_map[depth_map == np.inf] = 0
        depth_map = depth_map.astype(np.float32)

        depth_dict[img_basename] = depth_map


        # TODO: remove the dirty codes here
        pyimg = reconstruction.images[fname_to_id[img_basename]]
        pycam = reconstruction.cameras[pyimg.camera_id]

        # Generate the x and y coordinates
        x_coords = np.arange(hh)
        y_coords = np.arange(ww)
        xx, yy = np.meshgrid(x_coords, y_coords)

        # valid_depth_mask_hw = np.copy(valid_depth_mask)
        sampled_points2d = np.column_stack((xx.ravel(), yy.ravel()))
        # sampled_points2d = sampled_points2d + 0.5 # TODO figure it out if we still need +0.5

        depth_values = depth_map.reshape(-1)
        valid_depth_mask = valid_depth_mask.reshape(-1)

        sampled_points2d = sampled_points2d[valid_depth_mask]
        depth_values = depth_values[valid_depth_mask]

        unproject_points = pycam.cam_from_img(sampled_points2d)
        unproject_points_homo = np.hstack(
            (unproject_points, np.ones((unproject_points.shape[0], 1)))
        )
        unproject_points_withz = (
                unproject_points_homo * depth_values.reshape(-1, 1)
        )
        unproject_points_world = (
                pyimg.cam_from_world.inverse() * unproject_points_withz
        )

        rgb_image = original_images[img_basename] / 255.0
        rgb = rgb_image.reshape(-1, 3)
        rgb = rgb[valid_depth_mask]

        unproj_dense_points3D[img_basename] = np.array(
            [unproject_points_world, rgb]
        )

    # if not visual_dense_point_cloud:
    #     unproj_dense_points3D = None
    #
    # # subsampled_points, subsampled_colors = fuse_and_subsample_point_clouds(unproj_dense_points3D)

    return depth_dict, unproj_dense_points3D


def build_monocular_depth_model(device='cuda', model='DepthAnything'):
    """
    Builds the monocular depth model and loads the checkpoint.

    This function initializes the DepthAnythingV2 model,
    downloads the pre-trained weights from a URL, and loads these weights into the model.
    The model is then moved to the appropriate device and set to evaluation mode.
    """
    if model == 'DepthAnything':
        from dependency.depth_any_v2.depth_anything_v2.dpt import (
            DepthAnythingV2,
        )

        print("Building DepthAnythingV2")
        model_config = {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        }
        depth_model = DepthAnythingV2(**model_config)
        checkpoint_path = "ckpt/checkpoints/depth_anything_v2_vitl.pth"
        checkpoint = torch.load(checkpoint_path)
        depth_model.load_state_dict(checkpoint)
        print(f"DepthAnythingV2 built successfully")
    elif model == 'Metric3D':
        depth_model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
        # pred_depth, confidence, output_dict = model.inference({'input': rgb})

    else:
        raise ValueError(f"Model {model} not supported")


    return depth_model.to(device)


def save_dense_depth_into_point_cloud(unproj_dense_points3D, output_dir):
    """
    Save the dense point cloud to disk.

    Args:
        unproj_dense_points3D (dict): Dictionary containing unprojected dense points.
        output_dir (str): Directory to save the point cloud.
    """
    point_cloud_dir = os.path.join(output_dir, "point_cloud")
    os.makedirs(point_cloud_dir, exist_ok=True)
    for img_basename in tqdm(unproj_dense_points3D):
        fn_output_path = Path(point_cloud_dir) / (img_basename.split('.')[0] + '.ply')
        write_unproj_dense_points3D_to_ply(
            unproj_dense_points3D_ins=unproj_dense_points3D[img_basename],
            filename=fn_output_path
        )
    # fuse_points, fuse_colors = fuse_and_subsample_point_clouds(
    #     unproj_dense_points3D, stem="frames_0"
    # )
    # write_unproj_dense_points3D_to_ply(
    #     unproj_dense_points3D_ins=(fuse_points, fuse_colors),
    #     filename=os.path.join(point_cloud_dir, "fused_points.ply")
    # )


def write_unproj_dense_points3D_to_ply(unproj_dense_points3D_ins, filename):
    # Extract xyz coordinates and rgb values
    xyz = unproj_dense_points3D_ins[0]
    rgb = unproj_dense_points3D_ins[1] * 255.0  # Color was scaled to [0, 1] before

    num_points = xyz.shape[0]

    # Open the file in write mode
    with open(filename, 'w') as ply_file:
        # Write the header
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {num_points}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("end_header\n")

        # Write the point data
        for i in range(num_points):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            ply_file.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")

    print("Point cloud saved to", filename, "containing %i points." % num_points)


def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def write_array(array, path):
    """
    see: src/mvs/mat.h
        void Mat<T>::Write(const std::string& path)
    """
    assert array.dtype == np.float32
    if len(array.shape) == 2:
        height, width = array.shape
        channels = 1
    elif len(array.shape) == 3:
        height, width, channels = array.shape
    else:
        assert False

    with open(path, "w") as fid:
        fid.write(str(width) + "&" + str(height) + "&" + str(channels) + "&")

    with open(path, "ab") as fid:
        if len(array.shape) == 2:
            array_trans = np.transpose(array, (1, 0))
        elif len(array.shape) == 3:
            array_trans = np.transpose(array, (1, 0, 2))
        else:
            assert False
        data_1d = array_trans.reshape(-1, order="F")
        data_list = data_1d.tolist()
        endian_character = "<"
        format_char_sequence = "".join(["f"] * len(data_list))
        byte_data = struct.pack(
            endian_character + format_char_sequence, *data_list
        )
        fid.write(byte_data)


if __name__ == '__main__':

    import pycolmap

    dp_sfm_rec_path = Path(r"/e_disk/ZMData/ZhiNengDao/sample-left-front-right-20-from-2075-to-94-480P/results_superpoint+lightglue_matching_lowres_quality_high/reconstruction")
    dp_images = Path(r"/e_disk/ZMData/ZhiNengDao/sample-left-front-right-20-from-2075-to-94-480P/images")

    # 1. Load SfM Reconstruction
    sfm_rec = pycolmap.Reconstruction(dp_sfm_rec_path.as_posix())

    # 2. Compute
    sparse_depth, sparse_point = extract_sparse_depth_and_point_from_reconstruction(
        sfm_rec
    )
    image_paths = []
    for image in sfm_rec.images.values():
        img_name = image.name
        img_path = dp_images / img_name
        assert img_path.exists(), FileNotFoundError(f"Image {img_path} not found")
        image_paths.append(img_path)

    # 3. Extract Dense Depth
    dp_monodepth = dp_sfm_rec_path.parent / "monodepth"
    dp_monodepth.mkdir(exist_ok=True)
    depth_model = build_monocular_depth_model(device='cuda', model='DepthAnything')
    disp_dict, dict_original_images = extract_dense_depth_maps(depth_model, image_paths, dp_depth=dp_monodepth)

    # 4. Align Dense Depth
    depth_dict, unproj_dense_points3D = align_dense_depth_maps(sfm_rec, sparse_depth, disp_dict, dict_original_images)

    # 5. Write the Ply Out
    dp_point_cloud_output = dp_sfm_rec_path.parent / "mono-depth-aligned"
    dp_point_cloud_output.mkdir(exist_ok=True)
    for pcd_3d in zip(unproj_dense_points3D):
        save_dense_depth_into_point_cloud(unproj_dense_points3D, dp_point_cloud_output.as_posix())