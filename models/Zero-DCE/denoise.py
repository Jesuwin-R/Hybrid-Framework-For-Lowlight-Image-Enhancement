import cv2
import numpy as np

INPUT_PATH = r"D:\College\Projects\ICML-Image\my_test\test_enhanced_v2.jpg"
OUTPUT_PATH = r"D:\College\Projects\ICML-Image\my_test\test_enhanced_v2_clean.jpg"


def gray_world_white_balance(img_bgr):
    result = img_bgr.astype(np.float32)
    b, g, r = cv2.split(result)

    mean_b, mean_g, mean_r = b.mean(), g.mean(), r.mean()
    mean_gray = (mean_b + mean_g + mean_r) / 3.0

    b = b * (mean_gray / (mean_b + 1e-6))
    g = g * (mean_gray / (mean_g + 1e-6))
    r = r * (mean_gray / (mean_r + 1e-6))

    balanced = cv2.merge([b, g, r])
    return np.clip(balanced, 0, 255).astype(np.uint8)


def denoise(img_bgr):
    return cv2.fastNlMeansDenoisingColored(
        img_bgr,
        None,
        h=10,
        hColor=10,
        templateWindowSize=7,
        searchWindowSize=21,
    )


img = cv2.imread(INPUT_PATH)
if img is None:
    raise FileNotFoundError(f"Could not read: {INPUT_PATH}")

print("Step 1/2: Fixing color...")
balanced = gray_world_white_balance(img)

print("Step 2/2: Removing noise (this can take 10-30 seconds)...")
cleaned = denoise(balanced)

cv2.imwrite(OUTPUT_PATH, cleaned)
print(f"Done! Saved: {OUTPUT_PATH}")