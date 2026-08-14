from utils import *
from measure_nbt import *


jpg_path = "C:/jklai/project/Ath_NBT_CV-system/img/czi/testing/nbt_01.jpg"
png_path = "C:/jklai/project/Ath_NBT_CV-system/img/czi/testing/gray.png"
bmp_path = "C:/jklai/project/Ath_NBT_CV-system/img/czi/testing/gray.bmp"
tiff_path = "C:/jklai/project/Ath_NBT_CV-system/img/czi/testing/ath_nbt_blueish.tiff"
czi_path = "C:/jklai/project/Ath_NBT_CV-system/img/czi/testing/ath_nbt_reddish.czi"

img = measure_nbt(tiff_path)

img