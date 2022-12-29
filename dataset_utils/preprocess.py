import os
from pathlib import Path
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
# import dicomsdl
import numpy as np
from PIL import Image
import cv2
import concurrent.futures
from timeit import default_timer as timer
from tqdm import tqdm

depth_to_dtype = {
    8: np.uint8,
    16: np.uint16
}

# def new_read_xray(path):
#     dicom = dicomsdl.open(str(path))
#     bits = dicom.BitsStored
#     data = dicom.pixelData()

#     if dicom.getPixelDataInfo()['PhotometricInterpretation'] == "MONOCHROME1":
#         data = (2**bits - 1) - data
    
#     return data, bits

def read_xray(path, voi_lut = True, fix_monochrome = True):
    dicom = pydicom.read_file(path)
    bits = dicom.BitsStored

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = (2**bits - 1) - data
       
    return data, bits


def crop_roi_from_image(img: np.ndarray):
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 3)
    # _, breast_mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, breast_mask = cv2.threshold(blur,0,255, 16)
    
    cnts, _ = cv2.findContours(breast_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    # return img[y:y+h, x:x+w, ...]
    
    return x, y, w, h

def convert_fun(img_path: Path, dest_path: Path, depth: int=8):
    # Create directory
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    # Actual reading
    # img, bits = new_read_xray(img_path)
    # img, bits = read_xray(img_path, voi_lut=True)
    img, bits = read_xray(img_path, voi_lut=True)
    # Convert to uint8 to cut out interesting portion of image (not working in uint16)
    img_8 = (img * 2**(8 - bits)).astype(np.uint8)
    x, y, w, h = crop_roi_from_image(img_8)
    cropped_img = img_8[y:y+h, x:x+w, ...]

    # # Working with multiple depths
    # cropped_img = img[y:y+h, x:x+w, ...]
    # cropped_img = (cropped_img * 2**(depth - bits)).astype(depth_to_dtype[depth])

    # Save cropped image
    try:
        Image.fromarray(cropped_img).save(dest_path)
        # np.save(dest_path, cropped_img)
    except Exception:
        print(f"It was not possible to save {str(img_path)}, skipping...")
    

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

    # with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as worker:
    #     print(list(tqdm(worker.map(
    #         convert_fun,
    #         input_paths,
    #         output_paths
    #     ), total=len(input_paths))))
    i = 0
    start = timer()
    progressbar = tqdm(zip(input_paths, output_paths))
    for input_path, output_path in progressbar:
        convert_fun(input_path, output_path)
        i += 1
        progressbar.set_description(f"imgs/s: {i / (timer()-start):.4f}")

    

if "__main__" in __name__:
    # img_path = Path("/data/rsna-breast-cancer-detection/train_images/5325/324732883.dcm")
    # Image.fromarray(read_xray(img_path)).save("img.png")
    # Image.fromarray(crop_roi_from_image(read_xray(img_path))).save("roi_only.png")
    # dataset_path = Path("/original_dataset/")
    dataset_path = Path("/data/rsna-breast-cancer-detection/test_images")
    output_path = Path("/data/rsna-breast-cancer-detection/deleteme")
    convert_dataset(dataset_path, output_path)
