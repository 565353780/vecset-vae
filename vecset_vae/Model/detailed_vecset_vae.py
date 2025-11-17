import os
import torch
from torch import nn
from typing import Union

from vecset_vae.Model.vecset_vae import build_dora_vae, build_refine_vae


def set_model_pretrained(model: nn.Module) -> bool:
    for param in model.parameters():
        param.requires_grad = False
    return True


class DetailedVecSetVAE(nn.Module):
    def __init__(
        self,
        refine_resolutions: Union[list, None] = [
            [64, 64],
        ],
    ) -> None:
        super().__init__()

        self.dora_vae = build_dora_vae(64)
        set_model_pretrained(self.dora_vae)

        self.refine_vae_list = nn.ModuleList()
        if refine_resolutions is not None:
            for i in range(len(refine_resolutions)):
                num_latents, embed_dim = refine_resolutions[i]
                refine_vae = build_refine_vae(num_latents, embed_dim)
                self.refine_vae_list.append(refine_vae)

        self.is_refine_vae_pretrained = [False for _ in self.refine_vae_list]
        return

    def loadDoraVAE(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print("[ERROR][DetailedVecSetVAE::loadDoraVAE]")
            print("\t model file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        state_dict = torch.load(model_file_path, map_location="cpu")["model"]
        self.dora_vae.load_state_dict(state_dict)
        return True

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print("[ERROR][DetailedVecSetVAE::loadModel]")
            print("\t model file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        state_dict = torch.load(model_file_path, map_location="cpu")["model"]
        self.load_state_dict(state_dict, strict=False)

        for i, refine_vae in enumerate(self.refine_vae_list):
            prefix = f"refine_vae_list.{i}."
            has_weights = any(k.startswith(prefix) for k in state_dict.keys())
            if has_weights:
                set_model_pretrained(refine_vae)
                self.is_refine_vae_pretrained[i] = True
            else:
                print(
                    f"[INFO] refine_vae_list[{i}] not found in checkpoint, keep trainable."
                )

        return True

    def forward(
        self,
        data_dict: dict,
    ) -> dict:
        data_dict["sample_posterior"] = False
        result_dict = self.dora_vae(data_dict)

        for i in range(len(self.refine_vae_list)):
            data_dict["sample_posterior"] = not self.is_refine_vae_pretrained[i]
            data_dict["coarse_tsdf"] = result_dict["tsdf"]

            refine_result_dict = self.refine_vae_list[i](data_dict)

            result_dict["tsdf"] = result_dict["tsdf"] + refine_result_dict["tsdf"]

        return result_dict
