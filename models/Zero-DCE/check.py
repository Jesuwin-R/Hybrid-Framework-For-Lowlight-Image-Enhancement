import dataloader

ds = dataloader.lowlight_loader(r"D:\College\Projects\ICML-Image\dataset\SICE\SICE_ZeroDCE", image_size=256)
sample = ds[0]
print(sample.shape, sample.min().item(), sample.max().item())
print(sample[:, 0, 0])