import torch
import torchvision
import os
import model
import numpy as np
from PIL import Image
import time


def lowlight(image_path):

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Load image
    data_lowlight = Image.open(image_path).convert("RGB")

    # Zero-DCE++ requires width/height divisible by scale_factor (12).
    # Resize down slightly if needed so the model doesn't crash.
    scale_factor = 12
    w, h = data_lowlight.size
    w = (w // scale_factor) * scale_factor
    h = (h // scale_factor) * scale_factor
    data_lowlight = data_lowlight.resize((w, h), Image.LANCZOS)

    data_lowlight = np.asarray(data_lowlight) / 255.0

    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.unsqueeze(0).cuda()

    # Create model
    DCE_net = model.enhance_net_nopool(scale_factor=scale_factor).cuda()

    # Load checkpoint (pretrained Zero-DCE++ weights)
    checkpoint = torch.load(
        r"D:\College\Projects\ICML-Image\models\Zero-DCE++\snapshots_Zero_DCE++\Epoch99.pth"
    )

    # Support both a raw state_dict and a full checkpoint dict.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        DCE_net.load_state_dict(checkpoint["model_state_dict"])
    else:
        DCE_net.load_state_dict(checkpoint)

    DCE_net.eval()

    # Inference
    start = time.time()

    with torch.no_grad():
        enhanced_image, _ = DCE_net(data_lowlight)

    end = time.time()

    print(f"Processing time: {end-start:.3f} seconds")

    # Save output
    result_path = (
        r"D:\College\Projects\ICML-Image\my_test\test_enhanced_dce_pp1.jpg"
    )

    torchvision.utils.save_image(
        enhanced_image,
        result_path
    )

    print(f"Saved: {result_path}")


if __name__ == "__main__":

    image_path = (
        r"D:\College\Projects\ICML-Image\my_test\test.jpg"
    )

    lowlight(image_path)

    print("\nEnhancement completed successfully.")
