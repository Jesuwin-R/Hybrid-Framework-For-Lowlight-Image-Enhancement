import cv2
import numpy as np

INPUT_PATH = r"D:\College\Projects\ICML-Image\my_test\test_enhanced_v2.jpg"
OUTPUT_PATH = r"D:\College\Projects\ICML-Image\my_test\test_enhanced_v2_balanced.jpg"

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


img = cv2.imread(INPUT_PATH)
if img is None:
    raise FileNotFoundError(f"Could not read: {INPUT_PATH}")

balanced = gray_world_white_balance(img)
cv2.imwrite(OUTPUT_PATH, balanced)
print(f"Saved: {OUTPUT_PATH}")