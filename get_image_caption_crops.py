import os
import json
from glob import glob
from tqdm import tqdm
import random
import numpy as np
import pandas as pd

import rclone
import dropbox

import cv2

from PIL import Image, ImageDraw, ImageFont
from pikepdf import Pdf, PdfImage


def pdfs_to_images(source_path, save_path):

    filename = os.path.splitext(os.path.basename(source_path))[0]
    
    # Grab pdf an dload first page
    pdf_file = Pdf.open(source_path)
    page1 = pdf_file.pages[0]
    # Grab image layer
    relevant_key = [key for key in page1.images.keys()][0]
    rawimage = page1.images[relevant_key]
    # Convert image
    pdfimage = PdfImage(rawimage)
    image = pdfimage.as_pil_image()

    # convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Save and add to data table
    image.save(save_path, quality=60)


def create_rclone_client_from_config(config_path):
    with open(config_path) as f:
        cfg = f.read()
    rclone_client = rclone.with_config(cfg)
    return rclone_client


def get_dbx_path(image_file_name, dbx_client):
    search = image_file_name.replace(".jpg", "")
    res = dbx_client.files_search(path="/organized_pdfs", query=search)
    try:
        out = res.matches[0].metadata.path_display[1:]
        return out
    except Exception:
        print(res)
        print(image_file_name)
        return None


def random_rgba(alpha=100):
    return random.randint(0,255), random.randint(0,255), random.randint(0,255), alpha


def bbox_center(bbox):
    x0, y0, x1, y1 = bbox
    return (x0 + x1)/2, (y0 + y1)/2


def bbox_min_side(bbox):
    x0, y0, x1, y1 = bbox
    return min(y1 - y0, x1 - x0)

def get_crops(crops_data, img_save_dir, pdf_save_dir, crop_save_dir, rclone_client=None, dbx_client=None):

    # Load scans from dropbox
    print("Loading scans from dropbox ...")
    for scan_path in tqdm(list(crops_data.keys())):
        if not os.path.exists(os.path.join(pdf_save_dir, scan_path[:-4] + '.pdf')):
            dbx_loc = get_dbx_path(scan_path, dbx_client)
            if dbx_loc:
                metadata, res = dbx_client.files_download('/' + dbx_loc)
                with open(os.path.join(pdf_save_dir, dbx_loc.split('/')[-1]), 'wb') as f:
                    f.write(res.content)
            else:
                print(f"Could not find {scan_path} in dropbox")

    print("Converting scans to jpgs ...")
    for scan_path in tqdm(list(crops_data.keys())):
        if not os.path.exists(os.path.join(img_save_dir, scan_path)):
            try:
                pdfs_to_images(os.path.join(pdf_save_dir, scan_path[:-4] + '.pdf'), os.path.join(img_save_dir, scan_path))
            except OSError:
                print(f"Could not convert {scan_path} to jpg")


    print("Cropping ...")
    empty_img_count = 0
    empty_imgs = []
    for scan_path in tqdm(list(crops_data.keys())):
        scan_image = cv2.imread(os.path.join(img_save_dir, scan_path))

        for i, bbox in enumerate(crops_data[scan_path]):
            file_name = scan_path[:-4]
            crop_image_name = f'{file_name}_{i}'

            if not os.path.exists(f'{crop_save_dir}/{crop_image_name}.png'):
                # convert everything to integers with some padding
                left = max(int(bbox[0] - 5), 0)
                bottom = max(int(bbox[1] - 5), 0)
                right = min(int(bbox[2] + 5), scan_image.shape[1])
                top = min(int(bbox[3] + 5), scan_image.shape[0])

                cropped_image = scan_image[bottom:top, left:right]

                try:
                    cv2.imwrite(f'{crop_save_dir}/{crop_image_name}.png', cropped_image)
                except:
                    empty_img_count += 1
                    empty_imgs.append(crop_image_name)

    print(empty_img_count, "empty images")
    print(empty_imgs)

if __name__ == '__main__':
    
    os.chdir(r'C:\Users\bryan\Documents\NBER\datasets\image_captions')

    with open("dbx_access_token.txt") as f:
        dbx_access_token = f.read().strip()
    dbx_client = dropbox.Dropbox(dbx_access_token)
    rclone_client = create_rclone_client_from_config("rclone.conf")

    with open('image_caption_locs.json', 'r') as f:
        data = json.load(f)
    
    get_crops(data, 'jpegs', 'pdfs', 'caption_crops', rclone_client, dbx_client)
