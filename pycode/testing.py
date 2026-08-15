from utils import *
from measure_nbt import *
import csv
import numpy as np
import skimage as sk

img_type = (".jpg", ".jpeg", ".tif", ".tiff", ".png", ".bmp", ".czi")
jpg_path = "C:/jklai/project/Ath_NBT_CV-system/img/czi/testing/nbt_01.jpg"
png_path = "C:/jklai/project/Ath_NBT_CV-system/img/czi/testing/gray.png"
bmp_path = "C:/jklai/project/Ath_NBT_CV-system/img/czi/testing/gray.bmp"
tiff_path = "C:/jklai/project/Ath_NBT_CV-system/img/czi/testing/ath_nbt_blueish.tiff"
tiff_path_2 = "C:/jklai/project/Ath_NBT_CV-system/img/czi/testing/nbt.tif"
czi_path = "C:/jklai/project/Ath_NBT_CV-system/img/czi/testing/ath_nbt_reddish.czi"

img_list = Path("C:/jklai/project/Ath_NBT_CV-system/img/czi/testing").rglob("*")
img_list = [i for i in img_list if i.is_file() and i.suffix.lower() in img_type]
img_list = [str(i).replace("\\", "/") for i in img_list]
print(f"\nFound {len(img_list)} images.\n")

input_folder = "C:/jklai/project/Ath_NBT_CV-system/img/czi/testing"
output_folder = "C:/jklai/project/Ath_NBT_CV-system/img/czi/OUT_testing"

ret = measure_nbt_multproc("C:/jklai/project/Ath_NBT_CV-system/img/czi/testing", use_cores=8)


