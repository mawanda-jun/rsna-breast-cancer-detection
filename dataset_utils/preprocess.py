from itertools import repeat
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
from fileformat_pb2 import PBImage

depth_to_dtype = {
    8: np.uint8,
    12: np.uint16,
    16: np.uint16
}

def save_pb(dest_path, img, depth):
    shape0, shape1 = img.shape[:2]
    data = img.flatten().tolist()
    pb_img = PBImage()
    pb_img.shape0 = shape0
    pb_img.shape1 = shape1
    pb_img.depth = depth
    pb_img.data.extend(data)
    with open(dest_path, 'wb') as writer:
        writer.write(pb_img.SerializeToString())

# def new_read_xray(path):
#     dicom = dicomsdl.open(str(path))
#     bits = dicom.BitsStored
#     data = dicom.pixelData()

#     if dicom.getPixelDataInfo()['PhotometricInterpretation'] == "MONOCHROME1":
#         data = (2**bits - 1) - data
    
#     return data, bits

def read_xray(path, voi_lut = True, fix_monochrome = True):
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
        bits = dicom.BitsStored
    else:
        data = dicom.pixel_array
        if data.max() < 10000:
            bits = 12
            data[data > 2**bits - 1] = 2**bits - 1
        else:
            bits = 16

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

def convert_fun(img_path: Path, dest_path: Path, depth: int=8, voi_lut: bool=True):
    # Create directory
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    # Actual reading
    # img, bits = new_read_xray(img_path)
    # img, bits = read_xray(img_path, voi_lut=True)
    img, bits = read_xray(img_path, voi_lut=voi_lut)
    # Convert to uint8 to cut out interesting portion of image (not working in uint16)
    img_8 = (img * 2**(8 - bits)).astype(np.uint8)
    x, y, w, h = crop_roi_from_image(img_8)
    
    # cropped_img = img_8[y:y+h, x:x+w, ...]
    # Working with multiple depths
    cropped_img = img[y:y+h, x:x+w, ...]
    cropped_img = (cropped_img * 2**(depth - bits)).astype(depth_to_dtype[depth])

    # Save cropped image
    try:
        # pass
        Image.fromarray(cropped_img).save(dest_path)
        # np.save(dest_path, cropped_img)
        # save_pb(dest_path, cropped_img, depth)
    except Exception:
        print(f"It was not possible to save {str(img_path)}, skipping...")

def convert_dataset(dataset_dir: Path, output_dir: Path, depth: int=8, voi_lut: bool=True):
    """Util to convert dicom dataset to a png one.
    Folder structure should be dataset_dir/patient_id/<dicom_filename>.dcm
    We want to keep the patient_id folder!

    Args:
        dataset_dir (_type_): _description_
        output_dir (_type_): _description_
    """
    input_paths = list(dataset_dir.glob("**/*.dcm"))
    if depth == 8:
        ext = "png"
    else:
        ext = "tiff"
    output_paths = [output_dir / img_path.relative_to(img_path.parent.parent).with_suffix('').with_suffix(f'.{ext}') for img_path in input_paths]

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as worker:
        list(tqdm(worker.map(
            convert_fun,
            input_paths,
            output_paths,
            repeat(depth, len(input_paths)),
            repeat(voi_lut, len(input_paths))
        ), total=len(input_paths)))
    # i = 0
    # start = timer()
    # progressbar = tqdm(zip(input_paths, output_paths))
    # for input_path, output_path in progressbar:
    #     convert_fun(input_path, output_path, depth, voi_lut)
    #     i += 1
    #     progressbar.set_description(f"imgs/s: {i / (timer()-start):.4f}")

if "__main__" in __name__:
    dataset_path = Path("/original_dataset/train_images")
    output_path = Path("/data/rsna-breast-cancer-detection/train_images_12")
    depth = 12
    voi_lut = False
    convert_dataset(dataset_path, output_path, depth, voi_lut)
