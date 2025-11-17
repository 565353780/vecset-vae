import os
import torch


def convert_ckpt_to_pth(ckpt_path: str, pth_path: str):
    """
    将 Lightning 的 .ckpt 文件转换为普通 .pth 权重文件
    :param ckpt_path: 输入的 ckpt 文件路径
    :param pth_path: 输出的 pth 文件路径
    """
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Lightning 保存时会带有 'state_dict'
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        raise KeyError(
            "Checkpoint does not contain 'state_dict' key. Maybe not a Lightning ckpt?"
        )

    # 去掉可能的 'model.' 前缀
    new_state_dict = {
        k.replace("model.", "").replace("shape_", ""): v for k, v in state_dict.items()
    }

    new_checkpoint = {
        "model": new_state_dict,
    }

    # 保存为 .pth
    torch.save(new_checkpoint, pth_path)
    print(f"Converted state_dict saved to: {pth_path}")


if __name__ == "__main__":
    home = os.environ['HOME']
    dora_pretrained_model_file_path = home + "/chLi/Model/Dora/dora_vae_1_1.ckpt"
    save_dora_pretrained_model_file_path = dora_pretrained_model_file_path[:-4] + 'pth'
    convert_ckpt_to_pth(
        dora_pretrained_model_file_path, save_dora_pretrained_model_file_path
    )
