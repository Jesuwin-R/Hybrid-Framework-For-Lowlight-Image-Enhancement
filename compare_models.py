import os
import cv2
import matplotlib.pyplot as plt

# ===== CHANGE THESE PATHS =====
original_dir = r"D:\College\Projects\ICML-Image\dataset\LOL\lol_dataset\eval15\low"

zero_dce_dir = r"D:\College\Projects\ICML-Image\models\Zero-DCE\data\result"

zero_dcepp_dir = r"D:\College\Projects\ICML-Image\models\Zero-DCE++\data\result_Zero_DCE++\real"
# ==============================

images = sorted(os.listdir(original_dir))

index = 0

while True:

    img_name = images[index]

    original = cv2.imread(os.path.join(original_dir, img_name))
    dce = cv2.imread(os.path.join(zero_dce_dir, img_name))
    dcepp = cv2.imread(os.path.join(zero_dcepp_dir, img_name))

    if original is None or dce is None or dcepp is None:
        print("Missing:", img_name)
        index += 1
        if index >= len(images):
            break
        continue

    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    dce = cv2.cvtColor(dce, cv2.COLOR_BGR2RGB)
    dcepp = cv2.cvtColor(dcepp, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(18,6))

    plt.subplot(1,3,1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(dce)
    plt.title("Zero-DCE")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(dcepp)
    plt.title("Zero-DCE++")
    plt.axis("off")

    plt.suptitle(img_name)

    plt.show()

    cmd = input("Enter=Next | b=Back | q=Quit : ")

    if cmd.lower() == "q":
        break

    elif cmd.lower() == "b":
        index = max(0, index-1)

    else:
        index += 1

        if index >= len(images):
            break