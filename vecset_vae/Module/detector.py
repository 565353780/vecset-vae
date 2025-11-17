import os
import torch
import trimesh
from torch import nn
from typing import Union

from vecset_vae.Dataset.tsdf import TSDFDataset
from vecset_vae.Model.vecset_vae import VecSetVAE
from vecset_vae.Method.tomesh import extractMesh
from vecset_vae.Metric.tsdf import getTSDFAccPos, getTSDFAccNeg


class Detector(object):
    def __init__(
        self,
        model_file_path: Union[str, None] = None,
        use_ema: bool = True,
        batch_size: int = 1200000,
        resolution: int = 128,
        device: str = "cpu",
    ) -> None:
        self.batch_size = batch_size
        self.resolution = resolution
        self.device = device

        self.model = VecSetVAE().to(self.device)

        if model_file_path is not None:
            self.loadModel(model_file_path, use_ema)

        self.tsdf_dataset = TSDFDataset(
            "/home/chli/chLi/Dataset/",
            "Objaverse_82K/sharp_edge_sdf",
            split="val",
            n_supervision=[21384, 10000, 10000],
        )
        return

    def loadModel(self, model_file_path: str, use_ema: bool = True) -> bool:
        if not os.path.exists(model_file_path):
            print("[ERROR][Detector::loadModel]")
            print("\t model file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        state_dict = torch.load(model_file_path, map_location="cpu")

        if use_ema:
            model_state_dict = state_dict["ema_model"]
        else:
            model_state_dict = state_dict["model"]

        self.model.load_state_dict(model_state_dict)
        self.model.eval()

        print("[INFO][Detector::loadModel]")
        print("\t load model success!")
        print("\t model_file_path:", model_file_path)
        return True

    @torch.no_grad()
    def detect(
        self,
        coarse_surface: torch.Tensor,
        sharp_surface: torch.Tensor,
    ) -> Union[trimesh.Trimesh, None]:
        shape_latents = self.model.encode(coarse_surface, sharp_surface)
        kl_embed, _ = self.model.encode_kl_embed(shape_latents, sample_posterior=False)
        latents = self.model.decode(kl_embed)
        mesh = extractMesh(
            latents,
            self.model,
            self.resolution,
            self.batch_size,
            mode="odc",
        )
        return mesh

    @torch.no_grad()
    def detectDataset(self, data_idx: int) -> Union[trimesh.Trimesh, None]:
        data_dict = self.tsdf_dataset.__getitem__(data_idx)

        coarse_surface = data_dict["coarse_surface"].unsqueeze(0).to(self.device)
        sharp_surface = data_dict["sharp_surface"].unsqueeze(0).to(self.device)

        mesh = self.detect(coarse_surface, sharp_surface)

        queries = data_dict["rand_points"].unsqueeze(0).to(self.device)
        gt_tsdf = data_dict["tsdf"].unsqueeze(0).to(self.device)
        number_sharp = data_dict["number_sharp"]

        shape_latents = self.model.encode(coarse_surface, sharp_surface)
        kl_embed, _ = self.model.encode_kl_embed(shape_latents, sample_posterior=False)
        latents = self.model.decode(kl_embed)
        pred_tsdf = self.model.query(queries, latents)

        gt_sharp_tsdf = gt_tsdf[:, :number_sharp]
        gt_coarse_tsdf = gt_tsdf[:, number_sharp:]

        pred_sharp_tsdf = pred_tsdf[:, :number_sharp]
        pred_coarse_tsdf = pred_tsdf[:, number_sharp:]

        loss_sharp_tsdf = nn.L1Loss()(pred_sharp_tsdf, gt_sharp_tsdf)

        loss_coarse_tsdf = nn.L1Loss()(pred_coarse_tsdf, gt_coarse_tsdf)

        tsdf_dist_to_unit_dist_scale = 2.0 / 0.015
        acc_1536_pos = getTSDFAccPos(
            gt_tsdf, pred_tsdf, tsdf_dist_to_unit_dist_scale / 1536.0
        )
        acc_1024_pos = getTSDFAccPos(
            gt_tsdf, pred_tsdf, tsdf_dist_to_unit_dist_scale / 1024.0
        )
        acc_512_pos = getTSDFAccPos(
            gt_tsdf, pred_tsdf, tsdf_dist_to_unit_dist_scale / 512.0
        )
        acc_256_pos = getTSDFAccPos(
            gt_tsdf, pred_tsdf, tsdf_dist_to_unit_dist_scale / 256.0
        )
        acc_1536_neg = getTSDFAccNeg(
            gt_tsdf, pred_tsdf, tsdf_dist_to_unit_dist_scale / 1536.0
        )
        acc_1024_neg = getTSDFAccNeg(
            gt_tsdf, pred_tsdf, tsdf_dist_to_unit_dist_scale / 1024.0
        )
        acc_512_neg = getTSDFAccNeg(
            gt_tsdf, pred_tsdf, tsdf_dist_to_unit_dist_scale / 512.0
        )
        acc_256_neg = getTSDFAccNeg(
            gt_tsdf, pred_tsdf, tsdf_dist_to_unit_dist_scale / 256.0
        )

        print("loss_coarse_tsdf:", loss_coarse_tsdf)
        print("loss_sharp_tsdf:", loss_sharp_tsdf)
        print("acc_1536_pos:", acc_1536_pos)
        print("acc_1024_pos:", acc_1024_pos)
        print("acc_512_pos:", acc_512_pos)
        print("acc_256_pos:", acc_256_pos)
        print("acc_1536_neg:", acc_1536_neg)
        print("acc_1024_neg:", acc_1024_neg)
        print("acc_512_neg:", acc_512_neg)
        print("acc_256_neg:", acc_256_neg)
        return mesh
