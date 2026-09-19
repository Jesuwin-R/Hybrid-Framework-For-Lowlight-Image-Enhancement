import os
import cv2
import numpy as np

REPO_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "SCI-main\CVPR")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "SCI-main\CVPR\weights\difficult.pt")


class SCI:
    def __init__(self):
        self.model = None
        self.torch = None
        self._try_load()

    def _try_load(self):
        try:
            import torch
            import sys

            for mod_name in list(sys.modules):
                if mod_name == "model" or mod_name.startswith("model."):
                    del sys.modules[mod_name]

            if REPO_PATH not in sys.path:
                sys.path.insert(0, REPO_PATH)
            from model import Finetunemodel

            if not torch.cuda.is_available():
                print("[SCI] No GPU available. Falling back.")
                return

            model = Finetunemodel(WEIGHTS_PATH)
            model = model.cuda()
            model.eval()

            self.model = model
            self.torch = torch
        except Exception as e:
            print(f"[SCI] Falling back to classical enhancement: {e}")
            self.model = None

    def enhance(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.model is not None:
            return self._enhance_torch(image_bgr)
        return self._enhance_classical(image_bgr)

    def _enhance_torch(self, image_bgr: np.ndarray) -> np.ndarray:
        torch = self.torch
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        data = image_rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0).cuda()

        with torch.no_grad():
            i, r = self.model(tensor)

        out = r.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        out = (out * 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    def _enhance_classical(self, image_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        merged = cv2.merge((l, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)



