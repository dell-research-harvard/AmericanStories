import os
import argparse
from glob import glob
import pandas as pd
from PIL import Image
import io
import math
from typing import Tuple, Union

from tqdm import tqdm
from pikepdf import Pdf, PdfImage, Name
from cv2 import cv2, data
import numpy as np
# from deskew import determine_skew
from scipy.ndimage import interpolation as inter
import psutil


def pdfs_to_images(source_path, save_path, data_source, nested=True, resize=False, deskew=False):

    # Get all PDFS
    if os.path.isdir(source_path):
        if data_source == 'loc':
            all_paths = glob(f'{source_path}/**/*.jp2', recursive=True)
            if len(all_paths) == 0:
                all_paths = glob(f'{source_path}/**/*.jpg', recursive=True)
        else:
            all_paths = glob(f'{source_path}/**/*.pdf', recursive=True)
    elif os.path.isfile(source_path):
        all_paths = [source_path]
    else:
        return

    # Initialize output dir
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        print(f"Creating Directory {save_path}")

    # Initialize data table with converted images
    data_table = []

    # Initialize error log
    error_table = []

    for filepath in all_paths:

        if data_source == 'loc' and nested:
            filename = '_'.join(os.path.splitext(filepath)[0].split('/')[-4:])
        else:
            filename = os.path.splitext(os.path.basename(filepath))[0]

        try:

            if data_source == 'loc':
                image = Image.open(filepath)
            else:
                # Grab pdf an dload first page
                pdf_file = Pdf.open(filepath)
                page1 = pdf_file.pages[0]
                # Grab image layer
                #rawimage = page1.images['/im1']
                relevant_key = [key for key in page1.images.keys()][0]
                rawimage = page1.images[relevant_key]
                # Convert image
                # print(repr(rawimage.ColorSpace))
                pdfimage = PdfImage(rawimage)
                image = pdfimage.as_pil_image()

            # Deskew
            if deskew:
                image_as_array = np.array(image, dtype=np.float32)
                if len(image_as_array.shape) == 2:
                    grayscale = image_as_array
                else:
                    grayscale = cv2.cvtColor(image_as_array, cv2.COLOR_BGR2GRAY)
                deskew_angle = determine_skew(grayscale)
                deskew_angle = deskew_angle if deskew_angle >= 0 else 90 + deskew_angle
                if abs(deskew_angle) >= 5:
                    image = image.rotate(deskew_angle)
                else:
                    pass

            # convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            if resize:
                width, height = image.size
                image = image.resize((int(round(width / 4)), int(round(height / 4))))

            # Save and add to data table
            image.save(f'{save_path}/{filename}.jpg', quality=60)
            data_table.append([filepath, f'{filename}.jpg'])

        except Exception as e:
            print(filename)
            print(repr(e))

            # log error
            error_table.append([filename, repr(e)])

    # Save data table
    df = pd.DataFrame(data_table, columns=['source_path', 'save_name'])
    df.to_csv(f'{save_path}/data_table.csv', index=None)

    # Save  error log
    df = pd.DataFrame(error_table, columns=['pdf_name', 'error_code'])
    df.to_csv(f'{save_path}/error_table.csv', index=None)
