import torch
import torchvision.transforms as T
from PIL import Image
from huggingface_hub import hf_hub_download
from model.MIRNet.model import MIRNet


def lowlight(image_path, result_path):
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"[Info] Using device: {device}")

    # Downloads the pretrained weights automatically (first run only,
    # then it's cached locally)
    print("[Info] Downloading/loading pretrained weights...")
    model_path = hf_hub_download(
        repo_id="dblasko/mirnet-low-light-img-enhancement",
        filename="mirnet_finetuned.pth",
    )

    model = MIRNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
    model.eval()

    img = Image.open(image_path).convert("RGB")

    img_tensor = T.Compose([
        T.Resize(400),
        T.ToTensor(),
        T.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    ])(img).unsqueeze(0).to(device)

    # MIRNet requires dimensions divisible by 8
    if img_tensor.shape[2] % 8 != 0:
        img_tensor = img_tensor[:, :, : -(img_tensor.shape[2] % 8), :]
    if img_tensor.shape[3] % 8 != 0:
        img_tensor = img_tensor[:, :, :, : -(img_tensor.shape[3] % 8)]

    with torch.no_grad():
        output = model(img_tensor)

    output_img = T.ToPILImage()(output.squeeze(0).clamp(0, 1).cpu())
    output_img.save(result_path)
    print(f"[Done] Saved: {result_path}")


if __name__ == "__main__":
    image_path = r"D:\College\Projects\ICML-Image\my_test\test1.jpg"
    result_path = r"D:\College\Projects\ICML-Image\my_test\test_enhanced_mirnet_1.jpg"
    lowlight(image_path, result_path)
