import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import skimage as sk
from utils import *


def measure_nbt(img_path: str):
    ret = {
        "img_dirname": None,
        "img_basename": None,
        "nbt_area": -999,
        "nbt_mean": -999,
        "nbt_total": -999,
        "note": "failed"
    }

    try:
        img_dirname = os.path.dirname(img_path)
        img_basename = os.path.basename(img_path)
        img_suffix = Path(img_basename).suffix

        output_folder = os.path.join(os.path.dirname(img_dirname), f"OUT_{os.path.basename(img_dirname)}")
        NBT_output_subfolder = os.path.join(output_folder, "OUT_NBT")
        RGB_output_subfolder = os.path.join(output_folder, "OUT_RGB")
        GRAY_output_subfolder = os.path.join(output_folder, "OUT_GRAY")

        if not os.path.exists(NBT_output_subfolder):
            os.makedirs(NBT_output_subfolder, exist_ok=True)
            os.makedirs(RGB_output_subfolder, exist_ok=True)
            os.makedirs(GRAY_output_subfolder, exist_ok=True)
            
        NBT_img_output_path = os.path.join(NBT_output_subfolder, img_basename.replace(img_suffix, " (NBT).tiff"))
        RGB_img_output_path = os.path.join(RGB_output_subfolder, img_basename.replace(img_suffix, " (RGB).tiff"))
        GRAY_img_output_path = os.path.join(GRAY_output_subfolder, img_basename.replace(img_suffix, " (GRAY).tiff"))

        RGB, GRAY, PARAMS = read_image(img_path, bw_invert=True)
        height = PARAMS["height"]
        width = PARAMS["width"]

        if np.std(GRAY) == 0:
            ret["img_dirname"] = img_dirname
            ret["img_basename"] = img_basename
            ret["nbt_area"] = 0
            ret["nbt_mean"] = 0
            ret["nbt_total"] = 0
            return ret

        # Reduce image size to accelerate computation
        GRAY_resized = sk.transform.resize(GRAY, output_shape=(512, 512), anti_aliasing=True, preserve_range=True)
        GRAY_resized = convert_to_uint8(GRAY_resized)

        # Extract root region, this is the first step roughly to extract the NBT stained area
        threshold = sk.filters.threshold_otsu(GRAY_resized)
        root_region = GRAY_resized * np.uint8(GRAY_resized >= threshold)

        # Eliminate noises such as the cell wall or other tissues that were mis-stained
        kernel_size = estimate_kernel_size(GRAY)
        kernel = sk.morphology.disk(kernel_size)
        root_region = sk.filters.rank.median(root_region, kernel)

        # Extract ROI (the NBT stained area) from the root region
        # The opening operation is erosion followed by dilation.
        # I expect this operation will reduce the abnormal shape of the ROI, which
        # is usually caused by excessive staining in the middle. 
        # But the other problem is, if the NBT solution is not fully penetrated into 
        # the middle tissue, then this operation may yield even more abnormal shape.
        # So this part needs to be further tested and optimized.
        threshold = sk.filters.threshold_multiotsu(root_region, classes=4)
        ROI = root_region >= np.median(threshold)
        for _ in range(5):
            ROI = sk.morphology.opening(ROI, kernel)

        # Keep only the largest blob
        blobs = sk.measure.label(ROI)
        blobs = sk.measure.regionprops(blobs)
        if not blobs:
            raise ValueError("No spots detected")
        area = [i.area for i in blobs]
        area = max(area) - 1
        ROI = sk.morphology.remove_small_objects(ROI, max_size=area)
        # Slightly enlarge the ROI to cover the NBT stained area more completely
        kernel2 = sk.morphology.disk(np.ceil(kernel_size**0.5))
        ROI = sk.morphology.dilation(ROI, kernel2)
        # Resize the ROI back to the original dimension
        ROI = sk.transform.resize(ROI, (height, width), anti_aliasing=False, preserve_range=True)
        ROI_GRAY = np.multiply(GRAY, ROI)

        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # Output NBT measures
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        nbt_total = np.sum(ROI_GRAY)
        nbt_area = np.sum(ROI)  # How many pixels
        nbt_mean = nbt_total / nbt_area if nbt_area > 0 else 0

        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # Draw contour
        # Visualize the NBT stained area by drawing contour on the original image
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        NBT_img = Image.fromarray(RGB)
        draw = ImageDraw.Draw(NBT_img)
        contours = sk.measure.find_contours(ROI, level = 0)
        contour_color = (255, 0, 0) if PARAMS['channel'] == 3 else 255
        text = f"Avg: {round(nbt_mean, 2)} = {round(nbt_total/1_000_000, 2)} M / {nbt_area} pixels"
        font_settings = ImageFont.load_default(size=90)
            
        for contour in contours:
            points = [(c[1], c[0]) for c in contour]    # (x, y)
            draw.line(points, fill = contour_color, width = 7)
            draw.text(xy = (30, 10), text = text, fill = contour_color, font = font_settings)
        
        RGB_img = Image.fromarray(RGB)
        GRAY_img = Image.fromarray(GRAY)
        
        RGB_img.save(RGB_img_output_path, format="TIFF", tiffinfo=PARAMS['tiff_info'])
        GRAY_img.save(GRAY_img_output_path, format="TIFF", tiffinfo=PARAMS['tiff_info'])
        NBT_img.save(NBT_img_output_path, format="TIFF", tiffinfo=PARAMS['tiff_info'])

        ret["img_dirname"] = img_dirname
        ret["img_basename"] = img_basename
        ret["nbt_area"] = nbt_area
        ret["nbt_mean"] = nbt_mean
        ret["nbt_total"] = nbt_total
        ret["note"] = "ok"
        
    except:
        raise Warning(f"Failed to process {img_path}.")
    
    return ret

