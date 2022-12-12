import os
from pathlib import Path
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
from PIL import Image
import cv2
import concurrent.futures
from tqdm import tqdm


def read_xray(path, voi_lut = True, fix_monochrome = True):
    dicom = pydicom.read_file(path)
    max_px = 2 ** dicom.BitsStored - 1
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = max_px - data
        
    data = data / max_px
    data = (data * 255).astype(np.uint8)
        
    return data

def crop_roi_from_image(img: np.ndarray):
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 3)
    # _, breast_mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, breast_mask = cv2.threshold(blur,0,255, 16)
    
    _, cnts, _ = cv2.findContours(breast_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return img[y:y+h, x:x+w, ...]

def convert_fun(img_path: Path, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(crop_roi_from_image(read_xray(img_path))).save(dest_path)

def convert_dataset(dataset_dir: Path, output_dir: Path):
    """Util to convert dicom dataset to a png one.
    Folder structure should be dataset_dir/patient_id/<dicom_filename>.dcm
    We want to keep the patient_id folder!

    Args:
        dataset_dir (_type_): _description_
        output_dir (_type_): _description_
    """
    input_paths = list(dataset_dir.glob("**/*.dcm"))
    output_paths = [output_dir / img_path.relative_to(img_path.parent.parent).with_suffix('').with_suffix('.png') for img_path in input_paths]

    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as worker:
        _ = list(tqdm(worker.map(
            convert_fun,
            input_paths,
            output_paths
        ), total=len(input_paths)))
    

if "__main__" in __name__:
    # img_path = Path("/data/rsna-breast-cancer-detection/train_images/5325/324732883.dcm")
    # Image.fromarray(read_xray(img_path)).save("img.png")
    # Image.fromarray(crop_roi_from_image(read_xray(img_path))).save("roi_only.png")
    dataset_path = Path("/data/rsna-breast-cancer-detection/train_images")
    output_path = Path("/data/rsna-breast-cancer-detection/train_images_png")
    convert_dataset(dataset_path, output_path)