import os
import glob
import shutil

SRC = r"D:\College\Projects\ICML-Image\dataset\SICE\Dataset\train"
DST = r"D:\College\Projects\ICML-Image\dataset\SICE\SICE_ZeroDCE_filtered"
os.makedirs(DST, exist_ok=True)

# How many of the darkest images to take per scene. 1 = strictest, most
# faithful to the original Zero-DCE paper. 2 gives you more training data
# (SICE sequences aren't all the same length, so sort by filename number
# rather than assuming a fixed count).
N_DARKEST = 2

scene_folders = [f for f in glob.glob(os.path.join(SRC, "*")) if os.path.isdir(f)]
print(f"Found {len(scene_folders)} scene folders")

kept = 0
for scene in scene_folders:
    images = glob.glob(os.path.join(scene, "*.JPG")) + glob.glob(os.path.join(scene, "*.jpg"))
    if not images:
        continue

    # Sort numerically by filename (1.JPG, 2.JPG, ... not alphabetically)
    def sort_key(path):
        name = os.path.splitext(os.path.basename(path))[0]
        return int(name) if name.isdigit() else 999

    images.sort(key=sort_key)

    darkest = images[:N_DARKEST]
    scene_id = os.path.basename(scene)

    for img_path in darkest:
        img_num = os.path.splitext(os.path.basename(img_path))[0]
        new_name = f"scene{scene_id}_img{img_num}.JPG"
        shutil.copy(img_path, os.path.join(DST, new_name))
        kept += 1

print(f"Done. Copied {kept} low-light images to {DST}")