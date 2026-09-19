import glob, random
from PIL import Image
files = glob.glob(r"D:\College\Projects\ICML-Image\dataset\SICE\SICE_ZeroDCE_filtered\*.JPG")
print("Total files:", len(files))
sample = random.sample(files, 12)
for f in sample:
    img = Image.open(f).convert('L')
    mean = sum(img.getdata()) / (img.width * img.height)
    print(f, "mean brightness:", mean)