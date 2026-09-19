import os
import glob
import random
 
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image, UnidentifiedImageError
 
random.seed(1143)
 
# ---------------------------------------------------------------------------
# Supported extensions
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
 
 
def populate_train_list(images_path):
    """
    Scans `images_path` for all supported image files and returns a
    shuffled list of absolute file paths.
 
    Supports: .jpg, .JPG, .jpeg, .JPEG, .png, .PNG
    (covers the SICE dataset's `img_1.JPG ... img_4803.JPG` naming
    as well as .jpg/.png datasets like LOL).
    """
    if not os.path.isdir(images_path):
        raise FileNotFoundError(f"Dataset directory not found: {images_path}")
 
    image_list = []
    for pattern in SUPPORTED_EXTENSIONS:
        image_list.extend(glob.glob(os.path.join(images_path, pattern)))
 
    # De-duplicate: on Windows the filesystem is case-insensitive, so
    # "*.jpg" and "*.JPG" can both match the same file. We de-dupe by
    # the normalized (lowercased) path while keeping the original path
    # string for actually opening the file.
    seen = set()
    deduped = []
    for path in image_list:
        key = os.path.normcase(os.path.abspath(path))
        if key not in seen:
            seen.add(key)
            deduped.append(path)
 
    image_list = sorted(deduped)
 
    if len(image_list) == 0:
        raise RuntimeError(
            f"No images found in '{images_path}'. "
            f"Supported extensions: {SUPPORTED_EXTENSIONS}"
        )
 
    random.shuffle(image_list)
    return image_list
 
 
class lowlight_loader(data.Dataset):
    """
    Dataset class for Zero-DCE training.
 
    Each __getitem__ call loads and resizes one image to
    (image_size, image_size) and returns a normalized [0, 1] float
    tensor of shape (3, H, W).
 
    Corrupted / truncated / unreadable images are caught and skipped:
    a warning is printed and a different random image from the
    dataset is substituted, so a single bad file never crashes
    training.
    """
 
    def __init__(self, lowlight_images_path, image_size=256):
        self.data_list = populate_train_list(lowlight_images_path)
        self.size = image_size
 
        print("[Dataloader] Dataset path      :", lowlight_images_path)
        print("[Dataloader] Total images found:", len(self.data_list))
 
    def _load_image(self, image_path):
        """Loads, converts, resizes, and normalizes a single image."""
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize((self.size, self.size), Image.LANCZOS)
 
        img = np.asarray(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).float()
        return img.permute(2, 0, 1)  # HWC -> CHW
 
    def __getitem__(self, index):
        max_attempts = 10
        attempt = 0
        current_index = index
 
        while attempt < max_attempts:
            image_path = self.data_list[current_index]
            try:
                return self._load_image(image_path)
 
            except (UnidentifiedImageError, OSError, ValueError) as err:
                print(
                    f"[WARNING] Corrupted/unreadable image skipped: "
                    f"'{image_path}' ({type(err).__name__}: {err}). "
                    f"Substituting a random image instead."
                )
                attempt += 1
                current_index = random.randint(0, len(self.data_list) - 1)
 
        # Only reached if max_attempts consecutive images are bad,
        # which almost certainly means the dataset itself is broken.
        raise RuntimeError(
            f"Failed to load a valid image after {max_attempts} attempts "
            f"(started at index {index}). Please check your dataset for "
            f"widespread corruption."
        )
 
    def __len__(self):
        return len(self.data_list)
 