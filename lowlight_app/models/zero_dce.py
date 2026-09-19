import os
import cv2
import numpy as np

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "snapshots_SICE_v2", "best_model.pth")
REPO_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "Zero-DCE")


class ZeroDCE:
    def __init__(self):
        self.net = None
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
            import model as zero_dce_model

            if not os.path.exists(CHECKPOINT_PATH):
                return

            net = zero_dce_model.enhance_net_nopool()
            checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            net.load_state_dict(state_dict)
            net.eval()

            self.net = net
            self.torch = torch
        except Exception as e:
            print(f"[ZeroDCE] Falling back to classical enhancement: {e}")
            self.net = None

    def enhance(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.net is not None:
            return self._enhance_torch(image_bgr)
        return self._enhance_classical(image_bgr)

    def _enhance_torch(self, image_bgr: np.ndarray) -> np.ndarray:
        torch = self.torch
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        data = image_rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            _, enhanced, _ = self.net(tensor)

        out = enhanced.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
        out = (out * 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    def _enhance_classical(self, image_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        merged = cv2.merge((l, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


