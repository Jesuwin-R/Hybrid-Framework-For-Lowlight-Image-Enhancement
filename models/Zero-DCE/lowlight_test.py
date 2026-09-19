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
    data_lowlight = np.asarray(data_lowlight) / 255.0

    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.unsqueeze(0).cuda()

    # Create model
    DCE_net = model.enhance_net_nopool().cuda()

    # Load checkpoint (v2 — trained on filtered low-light-only dataset, no AMP)
    checkpoint = torch.load(
        r"D:\College\Projects\ICML-Image\snapshots_SICE_v2\best_model.pth"
    )

    DCE_net.load_state_dict(checkpoint["model_state_dict"])

    DCE_net.eval()

    # Inference
    start = time.time()

    with torch.no_grad():

        _, enhanced_image, _ = DCE_net(data_lowlight)

    end = time.time()

    print(f"Processing time: {end-start:.3f} seconds")

    # Save output (v2 filename — kept separate from the original result
    # so you can compare old vs. new side by side)
    result_path = (
        r"D:\College\Projects\ICML-Image\my_test\test_enhanced_v3.jpg"
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