import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset


def random_rotation_matrix():
    """
    Create a random rotation matrix.
    """
    angle = np.random.uniform(0, 2 * np.pi)
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    R = np.array(
        [
            [
                cos_angle + axis[0] ** 2 * (1 - cos_angle),
                axis[0] * axis[1] * (1 - cos_angle) - axis[2] * sin_angle,
                axis[0] * axis[2] * (1 - cos_angle) + axis[1] * sin_angle,
            ],
            [
                axis[1] * axis[0] * (1 - cos_angle) + axis[2] * sin_angle,
                cos_angle + axis[1] ** 2 * (1 - cos_angle),
                axis[1] * axis[2] * (1 - cos_angle) - axis[0] * sin_angle,
            ],
            [
                axis[2] * axis[0] * (1 - cos_angle) - axis[1] * sin_angle,
                axis[2] * axis[1] * (1 - cos_angle) + axis[0] * sin_angle,
                cos_angle + axis[2] ** 2 * (1 - cos_angle),
            ],
        ]
    )
    return R


def random_mirror_matrix():
    """
    Create a random mirror matrix.
    """
    if np.random.rand() < 0.75:
        axis = np.random.choice([0, 1, 2], size=1)[0]
        M = np.eye(3)
        M[axis, axis] = -1
    else:
        M = np.eye(3)
    return M


def apply_transformation(points, normals, transform):
    """
    Apply a transformation matrix to points and normals.
    """
    transformed_points = np.dot(points, transform.T)
    if normals is not None:
        norms = np.linalg.norm(normals, axis=1, keepdims=True)

        epsilon = 1e-6
        norms[norms == 0] = epsilon
        norms[np.isinf(norms)] = 1
        norms[np.isnan(norms)] = 1

        normals /= norms

        transformed_normals = np.dot(normals, transform.T)
        norms = np.linalg.norm(transformed_normals, axis=1, keepdims=True)

        epsilon = 1e-6
        norms[norms == 0] = epsilon
        norms[np.isinf(norms)] = 1
        norms[np.isnan(norms)] = 1

        transformed_normals /= norms
    else:
        transformed_normals = None
    return transformed_points, transformed_normals


class TSDFDataset(Dataset):
    def __init__(
        self,
        dataset_root_folder_path: str,
        sdf_folder_name: str,
        split: str = "train",
        n_supervision: list = [21384, 10000, 10000],
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.split = split
        self.n_supervision = n_supervision

        self.sdf_folder_path = self.dataset_root_folder_path + sdf_folder_name + "/"
        assert os.path.exists(self.sdf_folder_path)

        self.paths_list = []

        print("[INFO][TSDFDataset::__init__]")
        print("\t start load dataset:", self.sdf_folder_path)
        for root, _, files in os.walk(self.sdf_folder_path):
            for file in files:
                if not file.endswith(".npz"):
                    continue

                rel_file_basepath = (
                    os.path.relpath(root, self.sdf_folder_path) + "/" + file[:-4]
                )

                sdf_file_path = self.sdf_folder_path + rel_file_basepath + ".npz"
                assert os.path.exists(sdf_file_path)

                self.paths_list.append(sdf_file_path)

        self.paths_list.sort()
        return

    def __len__(self):
        return len(self.paths_list)

    def getRandomItem(self):
        random_idx = random.randint(0, len(self.paths_list) - 1)
        return self.__getitem__(random_idx)

    def __getitem__(self, index):
        index = index % len(self.paths_list)

        if self.split == "train":
            np.random.seed()
        else:
            np.random.seed(1234)

        sdf_file_path = self.paths_list[index]

        try:
            sdf_data = np.load(sdf_file_path)
            sdf_dict = {key: sdf_data[key] for key in sdf_data.files}
            coarse_surface = sdf_dict["fps_coarse_surface"].reshape(-1, 6)
            sharp_surface = sdf_dict["fps_sharp_surface"].reshape(-1, 6)
            near_sharp_pts = sdf_dict["sharp_near_surface"]
            rand_pts = sdf_dict["rand_points"]
        except:
            return self.getRandomItem()

        coarse_rand_points = rand_pts[:, :3]
        coarse_sdfs = rand_pts[:, 3]
        sharp_near_points = near_sharp_pts[:, :3]
        sharp_sdfs = near_sharp_pts[:, 3]

        rng = np.random.default_rng()
        ind2 = rng.choice(
            sharp_near_points.shape[0], self.n_supervision[0], replace=False
        )
        ind3 = rng.choice(
            coarse_rand_points[:400000].shape[0],
            self.n_supervision[1],
            replace=False,
        )
        ind4 = rng.choice(
            coarse_rand_points[400000:].shape[0],
            self.n_supervision[2],
            replace=False,
        )
        rand_points2 = sharp_near_points[ind2]
        rand_points3 = coarse_rand_points[:400000][ind3]
        rand_points4 = coarse_rand_points[400000:][ind4]
        rand_points = np.concatenate([rand_points2, rand_points3, rand_points4], axis=0)

        sdf2 = sharp_sdfs[ind2]
        sdf3 = coarse_sdfs[:400000][ind3]
        sdf4 = coarse_sdfs[400000:][ind4]
        sdfs = np.concatenate([sdf2, sdf3, sdf4], axis=0)
        nan_mask = np.isnan(sdfs)
        if np.any(nan_mask):
            print("nan exist in sdfs")
        sdfs = np.where(nan_mask, 0, sdfs)
        tsdfs = sdfs.flatten().astype(np.float32).clip(-0.015, 0.015) / 0.015

        if self.split == "train":
            mirror_matrix = random_mirror_matrix()
            rotation_matrix = random_rotation_matrix()

            mirrored_points, mirrored_normals = apply_transformation(
                coarse_surface[:, :3], coarse_surface[:, 3:], mirror_matrix
            )
            surface, normal = apply_transformation(
                mirrored_points, mirrored_normals, rotation_matrix
            )
            coarse_surface = np.concatenate([surface, normal], axis=1)

            mirrored_points, mirrored_normals = apply_transformation(
                sharp_surface[:, :3], sharp_surface[:, 3:], mirror_matrix
            )
            surface, normal = apply_transformation(
                mirrored_points, mirrored_normals, rotation_matrix
            )
            sharp_surface = np.concatenate([surface, normal], axis=1)

            mirrored_points, _ = apply_transformation(rand_points, None, mirror_matrix)
            rand_points, _ = apply_transformation(
                mirrored_points, None, rotation_matrix
            )

        feed_dict = {
            "coarse_surface": torch.tensor(coarse_surface).float(),
            "sharp_surface": torch.tensor(sharp_surface).float(),
            "rand_points": torch.tensor(rand_points).float(),
            "tsdf": torch.tensor(tsdfs).float(),
            "number_sharp": self.n_supervision[0],
        }

        return feed_dict
