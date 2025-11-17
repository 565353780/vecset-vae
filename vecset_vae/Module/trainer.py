import torch
from torch import nn
from typing import Union

from base_trainer.Module.base_trainer import BaseTrainer

from vecset_vae.Dataset.tsdf import TSDFDataset
from vecset_vae.Loss.eikonal import eikonal_loss_fn
from vecset_vae.Model.vecset_vae import VecSetVAE
from vecset_vae.Model.detailed_vecset_vae import DetailedVecSetVAE
from vecset_vae.Metric.tsdf import getTSDFAccPos, getTSDFAccNeg


def split_tsdf_loss(loss_fn, pred_tsdf, gt_tsdf):
    loss_all = loss_fn(pred_tsdf, gt_tsdf)  # element-wise loss

    mask_pos = gt_tsdf >= 0
    mask_neg = gt_tsdf < 0

    # 避免某一边全空，返回0
    loss_pos = (
        loss_all[mask_pos].mean() if mask_pos.any() else pred_tsdf.new_tensor(0.0)
    )
    loss_neg = (
        loss_all[mask_neg].mean() if mask_neg.any() else pred_tsdf.new_tensor(0.0)
    )

    return loss_pos, loss_neg


class Trainer(BaseTrainer):
    def __init__(
        self,
        dataset_root_folder_path: str,
        batch_size: int = 5,
        accum_iter: int = 10,
        num_workers: int = 16,
        model_file_path: Union[str, None] = None,
        weights_only: bool = False,
        device: str = "cuda:0",
        dtype=torch.float32,
        warm_step_num: int = 2000,
        finetune_step_num: int = -1,
        lr: float = 2e-4,
        lr_batch_size: int = 256,
        ema_start_step: int = 5000,
        ema_decay_init: float = 0.99,
        ema_decay: float = 0.999,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
        best_model_metric_name: Union[str, None] = None,
        is_metric_lower_better: bool = True,
        sample_results_freq: int = -1,
        use_amp: bool = False,
        quick_test: bool = False,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path

        self.gt_sample_added_to_logger = False

        self.loss_fn = nn.L1Loss(reduction="none")

        super().__init__(
            batch_size,
            accum_iter,
            num_workers,
            model_file_path,
            weights_only,
            device,
            dtype,
            warm_step_num,
            finetune_step_num,
            lr,
            lr_batch_size,
            ema_start_step,
            ema_decay_init,
            ema_decay,
            save_result_folder_path,
            save_log_folder_path,
            best_model_metric_name,
            is_metric_lower_better,
            sample_results_freq,
            use_amp,
            quick_test,
        )
        return

    def createDatasets(self) -> bool:
        eval = False
        self.dataloader_dict["tsdf"] = {
            "dataset": TSDFDataset(
                self.dataset_root_folder_path,
                "Objaverse_82K/sharp_edge_sdf",
                split="train",
                n_supervision=[21384, 10000, 10000],
            ),
            "repeat_num": 1,
        }

        if eval:
            self.dataloader_dict["eval"] = {
                "dataset": TSDFDataset(
                    self.dataset_root_folder_path,
                    "Objaverse_82K/sharp_edge_sdf",
                    split="val",
                    n_supervision=[21384, 10000, 10000],
                ),
            }

        if "eval" in self.dataloader_dict.keys():
            self.dataloader_dict["eval"]["dataset"].paths_list = self.dataloader_dict[
                "eval"
            ]["dataset"].paths_list[:4]

        return True

    def createModel(self) -> bool:
        mode = 2
        if mode == 1:
            self.model = VecSetVAE().to(self.device)
        elif mode == 2:
            self.model = DetailedVecSetVAE(
                [
                    [64, 64],
                ],
            ).to(self.device)
            dora_pretrained_model_file_path = (
                "/home/chli/chLi/Model/Dora/dora_vae_1_1.pth"
            )
            self.model.loadDoraVAE(dora_pretrained_model_file_path)
        return True

    def getLossDict(self, data_dict: dict, result_dict: dict) -> dict:
        lambda_sharp_logits = 2.0
        lambda_coarse_logits = 1.0
        lambda_kl = 0.001

        gt_tsdf = data_dict["tsdf"]
        pred_tsdf = result_dict["tsdf"]
        kl = result_dict["kl"]
        number_sharp = data_dict["number_sharp"][0]

        gt_sharp_tsdf = gt_tsdf[:, :number_sharp]
        gt_coarse_tsdf = gt_tsdf[:, number_sharp:]

        pred_sharp_tsdf = pred_tsdf[:, :number_sharp]
        pred_coarse_tsdf = pred_tsdf[:, number_sharp:]

        # loss_sharp_tsdf = self.loss_fn(pred_sharp_tsdf, gt_sharp_tsdf)
        # loss_coarse_tsdf = self.loss_fn(pred_coarse_tsdf, gt_coarse_tsdf)

        loss_pos_sharp, loss_neg_sharp = split_tsdf_loss(
            self.loss_fn, pred_sharp_tsdf.float(), gt_sharp_tsdf.float()
        )
        loss_pos_coarse, loss_neg_coarse = split_tsdf_loss(
            self.loss_fn, pred_coarse_tsdf.float(), gt_coarse_tsdf.float()
        )

        loss_sharp_tsdf = loss_pos_sharp + loss_neg_sharp
        loss_coarse_tsdf = loss_pos_coarse + loss_neg_coarse

        loss_kl = torch.mean(kl.float())

        # lambda_eikonal = 1.0
        # queries = data_dict["rand_points"]
        # loss_eikonal = eikonal_loss_fn(pred_tsdf, queries, gt_tsdf, trunc=1.0)

        loss = (
            lambda_sharp_logits * loss_sharp_tsdf
            + lambda_coarse_logits * loss_coarse_tsdf
            + lambda_kl * loss_kl
            # + lambda_eikonal * loss_eikonal
        )

        tsdf_dist_to_unit_dist_scale = 2.0 / 0.015
        loss_dict = {
            "Loss": loss,
            "LossCoarseTSDF+": loss_pos_coarse,
            "LossCoarseTSDF-": loss_neg_coarse,
            "LossSharpTSDF+": loss_pos_sharp,
            "LossSharpTSDF-": loss_neg_sharp,
            "Acc@1536+": getTSDFAccPos(
                gt_tsdf, pred_tsdf, tsdf_dist_to_unit_dist_scale / 1536.0
            ),
            "Acc@1536-": getTSDFAccNeg(
                gt_tsdf, pred_tsdf, tsdf_dist_to_unit_dist_scale / 1536.0
            ),
            "Acc@1024+": getTSDFAccPos(
                gt_tsdf, pred_tsdf, tsdf_dist_to_unit_dist_scale / 1024.0
            ),
            "Acc@1024-": getTSDFAccNeg(
                gt_tsdf, pred_tsdf, tsdf_dist_to_unit_dist_scale / 1024.0
            ),
            "Acc@512+": getTSDFAccPos(
                gt_tsdf, pred_tsdf, tsdf_dist_to_unit_dist_scale / 512.0
            ),
            "Acc@512-": getTSDFAccNeg(
                gt_tsdf, pred_tsdf, tsdf_dist_to_unit_dist_scale / 512.0
            ),
            "Acc@256+": getTSDFAccPos(
                gt_tsdf, pred_tsdf, tsdf_dist_to_unit_dist_scale / 256.0
            ),
            "Acc@256-": getTSDFAccNeg(
                gt_tsdf, pred_tsdf, tsdf_dist_to_unit_dist_scale / 256.0
            ),
            "LossKL": loss_kl,
            # "LossEikonal": loss_eikonal,
        }

        return loss_dict

    def preProcessData(self, data_dict: dict, is_training: bool = False) -> dict:
        if is_training:
            data_dict["split"] = "train"
        else:
            data_dict["split"] = "val"

        # eikonal loss needed grads
        # data_dict["rand_points"].requires_grad_(True)

        return data_dict

    @torch.no_grad()
    def sampleModelStep(self, model: nn.Module, model_name: str) -> bool:
        # FIXME: skip this since it will occur NCCL error
        return True

        dataset = self.dataloader_dict["dino"]["dataset"]

        model.eval()

        data_dict = dataset.__getitem__(1)

        print("[INFO][BaseDiffusionTrainer::sampleModelStep]")
        print("\t start sample shape code....")

        if not self.gt_sample_added_to_logger:
            # render gt here

            # self.logger.addPointCloud("GT_MASH/gt_mash", pcd, self.step)

            self.gt_sample_added_to_logger = True

        # self.logger.addPointCloud(model_name + "/pcd_" + str(i), pcd, self.step)

        return True
