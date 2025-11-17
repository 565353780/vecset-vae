from vecset_vae.Model.detailed_vecset_vae import DetailedVecSetVAE

def demo():
    device = 'cpu'

    model = DetailedVecSetVAE(
        [
            [64, 64],
        ],
    ).to(device)
    dora_pretrained_model_file_path = (
        "/home/chli/chLi/Model/Dora/dora_vae_1_1.pth"
    )
    # model.loadDoraVAE(dora_pretrained_model_file_path)

    return True
