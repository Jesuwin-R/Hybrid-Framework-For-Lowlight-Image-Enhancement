import torch
import numpy as np
from PIL import Image
from model import Finetunemodel


def lowlight(image_path, result_path, weights_path):

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available — SCI's Finetunemodel requires CUDA.")

    # Load image
    data_lowlight = Image.open(image_path).convert("RGB")
    data_lowlight = np.asarray(data_lowlight) / 255.0

    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.unsqueeze(0).cuda()

    # Create model (weights are loaded directly inside Finetunemodel's __init__)
    model = Finetunemodel(weights_path)
    model = model.cuda()
    model.eval()

    # Inference
    with torch.no_grad():
        i, r = model(data_lowlight)
        # r = the enhanced/reflectance image (this is what SCI's own test.py saves)

    # Save output
    image_numpy = r[0].cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype("uint8"))
    im.save(result_path)

    print(f"Saved: {result_path}")


if __name__ == "__main__":

    image_path = r"D:\College\Projects\ICML-Image\my_test\test.jpg"
    result_path = r"D:\College\Projects\ICML-Image\my_test\test_enhanced_sci.jpg"

    # difficult.pt was trained on DarkFace — matches the theme of your
    # other extreme-low-light tests. Switch to medium.pt or easy.pt if
    # you want to compare.
    weights_path = r"D:\College\Projects\ICML-Image\models\SCI-main\CVPR\weights\difficult.pt"

    lowlight(image_path, result_path, weights_path)

    print("\nEnhancement completed successfully.")
