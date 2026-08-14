import os
import sys
from pathlib import Path
import numpy as np
import skimage as sk
from PIL import Image, TiffImagePlugin
import czifile


#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Get the icon path during runtime
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def resource_path(relative_path):
    """ Get absolute path to resource (works for PyInstaller) """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Read images
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def is_czi(img_path: str):
    return Path(img_path).suffix.lower() == ".czi"


def is_tiff(img_path: str):
    return Path(img_path).suffix.lower() in [".tif", ".tiff"]


def read_czi(img_path: str):
    czi = czifile.CziFile(img_path)
    scene = czi.scenes()
    arr = scene.asarray()
    micrometer_per_pixel = scene.mpp

    if scene.ndim == 2:
        height, width = scene.shape
        channel = 1
    elif scene.ndim == 3:
        height, width, channel = scene.shape
    else:
        raise ValueError("Invalid dimensions.")
    
    XResolution = 10000 / micrometer_per_pixel[0]  # Pixels per cm
    YResolution = 10000 / micrometer_per_pixel[1]  # Pixels per cm
    ResolutionUnit = 3  # ResolutionUnit: 1=none, 2=inch, 3=centimeter
    ImageWidth = width
    ImageLength = height
    tiff_info = TiffImagePlugin.ImageFileDirectory_v2()
    tiff_info[282] = XResolution
    tiff_info[283] = YResolution
    tiff_info[296] = ResolutionUnit
    tiff_info[256] = ImageWidth
    tiff_info[257] = ImageLength

    return arr, height, width, channel, tiff_info


def read_tiff(img_path: str):
    tif = Image.open(img_path)
    tiff_info = tif.tag_v2
    arr = np.array(tif)

    if arr.ndim == 2:
        height, width = arr.shape
        channel = 1
    elif arr.ndim == 3:
        height, width, channel = arr.shape
    else:
        raise ValueError("Invalid dimensions.")
    
    return arr, height, width, channel, tiff_info


def read_image(img_path: str, bw_invert: bool = False):
    RGB = None
    GRAY = None
    PARAMS = None
    if is_czi(img_path):
        arr, height, width, channel, tiff_info = read_czi(img_path)
    elif is_tiff(img_path):
        arr, height, width, channel, tiff_info = read_tiff(img_path)
    else:
        arr = sk.io.imread(img_path)

    tiff_info[270] = "Bug report: https://github.com/mylab-root/NBT-stain_Ath/issues"
    
    if arr.ndim == 2:
        height, width = arr.shape
        channel = 1

    if arr.ndim == 3:
        height, width, channel = arr.shape

    if channel == 3:
        RGB = convert_to_uint8(arr)
        GRAY = convert_to_uint8(np.mean(arr, axis=2))
    else:
        RGB = sk.color.gray2rgb(arr)
        GRAY = convert_to_uint8(arr)

    if bw_invert is True:
        GRAY = np.bitwise_invert(GRAY)

    PARAMS = {
        "height": height,
        "width": width,
        "channel": channel,
        "tiff_info": tiff_info,
    }


    return RGB, GRAY, PARAMS


#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Image processing
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def scale_min_max(arr: np.ndarray, range=(0, 1)):
    arr = np.double(arr)
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = arr * (range[1] - range[0]) + range[0]
    return arr


def convert_to_uint8(arr: np.ndarray):
    arr = np.uint8(scale_min_max(arr) * 255)
    return arr


def estimate_kernel_size(arr: np.ndarray):
    kernel_size = np.sqrt(np.sum(arr > 0) / (512 * 512) * 100 * 0.66)
    kernel_size = np.ceil(kernel_size)
    # coerce to odd value
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    kernel_size = 3 if kernel_size < 3 else kernel_size
    return kernel_size


def show_image(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
        img.show()