from stages.images_to_layouts import get_onnx_input_name, letterbox, non_max_suppression
# from stages.pdfs_to_images import pdfs_to_images
from effocr.dataset_utils import create_paired_transform, create_paired_transform_word
from effocr.infer_ocr_onnx_multi import run_effocr
from effocr.infer_ocr_onnx_multi_word import run_effocr_word
from effocr.engines.localizer_engine import EffLocalizer
from effocr.engines.recognizer_engine import EffRecognizer
from effocr.engines.yolov8_ops import non_max_suppression as non_max_supression_yolov8
from generate_manifest import *
from utils.ca_metadata_utils import *

import os
import requests
import json
import faiss
import argparse
from tqdm import tqdm
from timeit import default_timer as timer
import torch
import onnx
import onnxruntime as ort
import numpy as np
import logging
from PIL import Image, ImageDraw
import cv2
import gc
import psutil
import pkg_resources
from torchvision.ops import nms
from torchvision import transforms
from symspellpy import SymSpell, Verbosity
import random
import multiprocessing
import time
from math import floor, ceil
from pytorch_metric_learning.utils.inference import FaissKNN
from pytorch_metric_learning.utils import common_functions as c_f

'''
Avoid rate limiting:

- wget images
- stagger jobs
- different subnets -- randomize ips on the subnets?
- Different vns for different regions, etc?
- Metadata readd?? - slow things down
- user agents -- for metadata too? or dump if wgetting
'''

LAYOUT_TYPES_TO_EFFOCR = ['article', 'author', 'headline', 'image_caption']
IMG_FILE_EXS = ('jpg', 'png', 'jp2')
LEGIBLE_LABEL_MAP = ['Legible', 'Questionable', 'Illegible']
LAYOUT_COLOR_MAP = {'article': 'blue', 'headline': 'red', 'cartoon_or_advertisement': 'orange'}

USER_HEADERS = ['Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/37.0.2062.94 Chrome/37.0.2062.94 Safari/537.36',
                                    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.85 Safari/537.36',
                                    'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
                                    'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.0',
                                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/600.8.9 (KHTML, like Gecko) Version/8.0.8 Safari/600.8.9',
                                    'Mozilla/5.0 (iPad; CPU OS 8_4_1 like Mac OS X) AppleWebKit/600.1.4 (KHTML, like Gecko) Version/8.0 Mobile/12H321 Safari/600.1.4',
                                    'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.85 Safari/537.36',
                                    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.85 Safari/537.36',
                                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.10240',
                                    'Mozilla/5.0 (Windows NT 6.3; WOW64; rv:40.0) Gecko/20100101 Firefox/40.0',
                                    'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko',
                                    'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.85 Safari/537.36',
                                    'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
                                    'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:40.0) Gecko/20100101 Firefox/40.0',
                                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_4) AppleWebKit/600.7.12 (KHTML, like Gecko) Version/8.0.7 Safari/600.7.12',
                                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.85 Safari/537.36',
                                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.10; rv:40.0) Gecko/20100101 Firefox/40.0',
                                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/600.8.9 (KHTML, like Gecko) Version/7.1.8 Safari/537.85.17',
                                    'Mozilla/5.0 (iPad; CPU OS 8_4 like Mac OS X) AppleWebKit/600.1.4 (KHTML, like Gecko) Version/8.0 Mobile/12H143 Safari/600.1.4',
                                    'Mozilla/5.0 (iPad; CPU OS 8_3 like Mac OS X) AppleWebKit/600.1.4 (KHTML, like Gecko) Version/8.0 Mobile/12F69 Safari/600.1.4']

def ord_str_to_word(w):
    return ''.join([chr(int(c)) for c in w.split('_')])

def get_crops_from_layout_image(image):
    im_width, im_height = image.size[0], image.size[1]
    if im_height <= im_width * 2:
        return [image]
    else:
        y0 = 0
        y1 = im_width * 2
        crops = []
        while y1 <= im_height:
            crops.append(image.crop((0, y0, im_width, y1)))
            y0 += int(im_width * 1.5)
            y1 += int(im_width * 1.5)
        
        crops.append(image.crop((0, y0, im_width, im_height)))
        return crops
    
def readjust_line_detections(line_preds, orig_img_width):
    y0 = 0
    dif = int(orig_img_width * 1.5)
    all_preds, final_preds = [], []
    for j in range(len(line_preds)):
        preds, probs, labels = line_preds[j]
        for i, pred in enumerate(preds):
            all_preds.append((pred[0], pred[1] + y0, pred[2], pred[3] + y0, probs[i]))
        y0 += dif
    
    all_preds = torch.tensor(all_preds)
    if all_preds.dim() > 1:
        keep_preds = nms(all_preds[:, :4], all_preds[:, -1], iou_threshold=0.15)
        filtered_preds = all_preds[keep_preds, :4]
        filtered_preds = filtered_preds[filtered_preds[:, 1].sort()[1]]
        for pred in filtered_preds:
            x0, y0, x1, y1 = torch.round(pred)
            x0, y0, x1, y1 = x0.item(), y0.item(), x1.item(), y1.item()
            final_preds.append((x0, y0, x1, y1))
        return final_preds
    else:
        return []

# Get all layout predictions for an image
def get_layout_predictions(layout_session, label_map_layout, ca_img, layout_output, f_idx, input_name, backend='yolo'):

    #finetuned yolov5 ONNX model

    # Resize and reshape image to fit model input
    im = letterbox(ca_img, (1280, 1280), auto=False)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.expand_dims(np.ascontiguousarray(im), axis = 0).astype(np.float32) / 255.0  # contiguous

    layout_predictions = layout_session.run(
        None,
        {input_name: im}
    )

    layout_predictions = torch.from_numpy(layout_predictions[0])
    if backend == 'yolo':
        layout_predictions = non_max_suppression(layout_predictions, conf_thres = 0.05, iou_thres=0.01, max_det=300, agnostic = True)[0] 
        print(layout_predictions.size())    
    elif backend == 'yolov8':
        layout_predictions = non_max_supression_yolov8(layout_predictions, conf_thres = 0.01, iou_thres=0.1, max_det=2000, agnostic = True)[0]
    
    layout_bboxes, layout_probs, layout_labels = layout_predictions[:, :4], layout_predictions[:, -2], layout_predictions[:, -1]

    crops_for_effocr = []
    layout_img = Image.fromarray(ca_img)
    im_width, im_height = layout_img.size[0], layout_img.size[1]
    
    # Precompute ratios and translations to rescale bounding boxes to original image size
    if im_width > im_height:
        w_ratio = 1280
        h_ratio = (im_width / im_height) * 1280
        w_trans = 0
        h_trans = 1280 * ((1 - (im_height / im_width)) / 2)
    else:
        h_trans = 0
        h_ratio = 1280
        w_trans = 1280 * ((1 - (im_width / im_height)) / 2)
        w_ratio = 1280 * (im_width / im_height)

    # Set up for drawing bounding boxes on image if requested
    if layout_output:
        draw = ImageDraw.Draw(layout_img)

    # Iterate through predicted bounding boxes, cropping out each from the original image
    layout_crops = []
    for i, (line_bbox, pred_class) in enumerate(zip(layout_bboxes, layout_labels)):
        x0, y0, x1, y1 = torch.round(line_bbox) # Grab the bounding box coordinates on resized image
        # Convert coordinages to original image size (messy)
        x0, y0, x1, y1 = int(floor((x0.item() - w_trans) * im_width / w_ratio)), int(floor((y0.item() - h_trans) * im_height / h_ratio)), \
                        int(ceil((x1.item() - w_trans) * im_width / w_ratio)), int(ceil((y1.item() - h_trans) * im_height  / h_ratio))

        # Crop out image and append to list of layout crops
        layout_crop = layout_img.crop((x0, y0, x1, y1))
        layout_crops.append((pred_class, (x0, y0, x1, y1), layout_crop))
        
        # Append chunked crop to a separate list of crops to EffOCR if in one of the desired types
        if label_map_layout[int(pred_class.item())] in LAYOUT_TYPES_TO_EFFOCR:
            crops = get_crops_from_layout_image(layout_crop) # Chunk the crop if it's a poor aspect ratio for line detection
            for crop in crops:
                crops_for_effocr.append((i, crop))
            
        if layout_output: # optionally save layout bounding boxes to file
            draw.rectangle((x0, y0, x1, y1), outline=LAYOUT_COLOR_MAP.get(label_map_layout[int(pred_class.item())], 'black'), width=5)
            # layout_crop.save(os.path.join(layout_output, 'bbox_{}_{}_{}.jpg'.format(f_idx, i, label_map_layout[int(pred_class.item())])))
    
    if layout_output: # Save the image with drawn bounding boxes if requested
        layout_img.save(os.path.join(layout_output, f'layout_boxes_{f_idx}.jpg'))

    return crops_for_effocr, layout_crops

def get_onnx_input_name(model):
    input_all = [node.name for node in model.graph.input]
    input_initializer =  [node.name for node in model.graph.initializer]
    net_feed_input = list(set(input_all)  - set(input_initializer))
    return net_feed_input[0]

def get_line_predictions(line_session, input_name, crops_for_effocr, backend = 'yolo'):

    bbox_preds = {}
    for bbox_idx, line_bbox in crops_for_effocr:
        gc.collect()
        im = letterbox(np.array(line_bbox), (640, 640), auto=False)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.expand_dims(np.ascontiguousarray(im), axis = 0).astype(np.float32) / 255.0  # contiguous

        line_predictions = line_session.run(
            None,
            {input_name: im}
        )

        line_predictions = torch.from_numpy(line_predictions[0])
        if backend == 'yolo':
            line_predictions = non_max_suppression(line_predictions, conf_thres = 0.2, iou_thres=0.15, max_det=200)[0]

        elif backend == 'yolov8':
            line_predictions = non_max_supression_yolov8(line_predictions, conf_thres = 0.2, iou_thres=0.15, max_det=200)[0]

        line_predictions = line_predictions[line_predictions[:, 1].sort()[1]]
        line_bboxes, line_confs, line_labels = line_predictions[:, :4], line_predictions[:, -2], line_predictions[:, -1]
        im_width, im_height = line_bbox.size[0], line_bbox.size[1]

        if im_width > im_height:
            h_ratio = (im_height / im_width) * 640
            h_trans = 640 * ((1 - (im_height / im_width)) / 2)
        else:
            h_trans = 0
            h_ratio = 640

        line_proj_crops = []
        for line_bbox in line_bboxes:
            x0, y0, x1, y1 = torch.round(line_bbox)
            x0, y0, x1, y1 = 0, int(floor((y0.item() - h_trans) * im_height / h_ratio)), \
                            im_width, int(ceil((y1.item() - h_trans) * im_height  / h_ratio))
        
            line_proj_crops.append((x0, y0, x1, y1))
        if bbox_idx not in bbox_preds.keys():
            bbox_preds[bbox_idx] = [(line_proj_crops, line_confs, line_labels)]
        else:
            bbox_preds[bbox_idx].append((line_proj_crops, line_confs, line_labels))
        

    return bbox_preds

def spell_check_results(results, delimiter = None):
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt"
    )
    bigram_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_bigramdictionary_en_243_342.txt"
    )

    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)
    results = results.copy()

    for k in results.keys():
        try:
            if delimiter == ' ':
                results[k] = ' '.join([sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)[0].term for word in results[k].split()])
            elif delimiter:
                results[k] = delimiter.join([sym_spell.lookup_compound(word, max_edit_distance=2, transfer_casing=True)[0].term for word in results[k].split(delimiter)])
            else:
                suggestions = sym_spell.lookup_compound(results[k], max_edit_distance=2, transfer_casing=True)
                results[k] = suggestions[0].term
        except:
            pass

    return results

def main(args):

    # Save arguments
    output_save_path = args.output_save_path
    img_save_path = os.path.join(output_save_path, "images")
    checkpoint_path_layout = args.checkpoint_path_layout
    checkpoint_path_line = args.checkpoint_path_line

    label_map_path_layout = args.label_map_path_layout

    recognizer_dir = args.effocr_recognizer_dir
    char_recognizer_dir = args.effocr_char_recognizer_dir
    localizer_dir = args.effocr_localizer_dir

    spell_check = args.spell_check
    first_n = args.first_n
    layout_line_only = args.layout_line_only

    layout_output = args.layout_output
    line_output = args.line_output
    localizer_output = args.localizer_output
    manifest_path = args.manifest_path
    legibility_classifier = args.legibility_classifier
    bbox_output = args.bbox_output
    word_level = args.word_level_effocr
    layout_model_backend = args.layout_model_backend
    localizer_model_backend = args.localizer_model_backend
    line_model_backend = args.line_model_backend

    localizer_iou_thresh = args.localizer_iou_thresh
    localizer_conf_thresh = args.localizer_conf_thresh
    recognizer_word_thresh = args.recognizer_word_thresh
    ad_hoc_textlines = args.ad_hoc_textlines
    punc_padding = args.punc_padding

    # Create output directories
    os.makedirs(output_save_path, exist_ok=True)
    errors_log = []

    # Read in manifest of scans to process
    if os.path.isdir(manifest_path):
        filenames = [os.path.join(manifest_path, p) for p in os.listdir(manifest_path)]
    elif os.path.isfile(manifest_path):
        with open(args.manifest_path) as infile:
            filenames = infile.readlines()
    else:
        raise FileNotFoundError('Could not find manifest in {}'.format(manifest_path))
    
    # Create extra output directories for visualization
    if layout_output: os.makedirs(layout_output, exist_ok=True)
    if line_output: os.makedirs(line_output, exist_ok=True)

    # Truncate if requested
    if first_n:
        filenames = filenames[:first_n]

    # Import pdf converter if needed
    if any([f.endswith('.pdf') for f in filenames]):
        '''
        NOTE: PDF functionality is not included in the requirements.txt file because it tends to 
        create compatiblity problems on Azure machines. If you want to use the functionality, 
        first run
            `pip install pikepdf`
        in your environment. Then provide one or more pdf files (can be downloaded from 
        chronicling america or stored locally) in either your manifest or your directory. 
        '''
        from stages.pdfs_to_images import pdfs_to_images

    # Set up logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=os.path.join(output_save_path, 'ca_transcription.txt'), level=logging.INFO)
    logging.getLogger(c_f.LOGGER_NAME).setLevel(logging.WARNING)
    logging.info('Transcribing {} files'.format(len(filenames)))
    
    # Load Layout Model
    layout_base_model = onnx.load(checkpoint_path_layout)
    layout_input_name = get_onnx_input_name(layout_base_model)
    del layout_base_model
    layout_inf_sess = ort.InferenceSession(checkpoint_path_layout)

    # Load layout label map
    with open(label_map_path_layout) as jf:
        label_map_data = json.load(jf)
        label_map_layout = {int(k): v for k, v in label_map_data.items()}
        del label_map_data

    # Load line model
    base_line_model = onnx.load(checkpoint_path_line)
    line_input_name = get_onnx_input_name(base_line_model)
    del base_line_model
    line_inf_sess = ort.InferenceSession(checkpoint_path_line)

    # Check that all effocr models are present
    assert os.path.exists(os.path.join(recognizer_dir, 'enc_best.onnx')), 'Recognizer model not found! Should be in {}/enc_best.onnx'.format(recognizer_dir)
    assert os.path.exists(os.path.join(localizer_dir, 'localizer_model_new.onnx')), 'Localizer model not found! Should be in {}/localizer_model.onnx'.format(localizer_dir)
    if char_recognizer_dir:
        assert os.path.exists(os.path.join(char_recognizer_dir, 'enc_best.onnx')), 'Character recognizer model not found! Should be in {}/enc_best.onnx'.format(char_recognizer_dir)
    print(f'Processing with {multiprocessing.cpu_count()} cores testing')
    print(localizer_model_backend)
    #Create localizer engine
    localizer_engine = EffLocalizer(
        os.path.join(localizer_dir, 'localizer_model_new.onnx'),
        iou_thresh=localizer_iou_thresh,
        conf_thresh=localizer_conf_thresh,
        vertical=False,
        num_cores=multiprocessing.cpu_count(),
        model_backend=localizer_model_backend,
        input_shape=(640, 640)
    )

    # Build legibility classifier
    legibility_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.CenterCrop([256, 256])])

    base_leg = onnx.load(legibility_classifier)
    leg_input_name = get_onnx_input_name(base_leg)

    legibility_classifier = ort.InferenceSession(legibility_classifier)

    # Build recognizer engine for character recognition
    char_transform = create_paired_transform(lang='en')
    word_transform = create_paired_transform_word(lang='en')
    if char_recognizer_dir:
        char_recognizer_engine = EffRecognizer(
            model = os.path.join(char_recognizer_dir, 'enc_best.onnx'),
            transform = char_transform,
            num_cores=4
        )
        char_knn_func = FaissKNN(
            index_init_fn=faiss.IndexFlatIP,
            reset_before=False, reset_after=False
        )
        char_knn_func.load(os.path.join(char_recognizer_dir, "ref.index"))

        with open(os.path.join(char_recognizer_dir, "ref.txt")) as ref_file:
            candidate_chars = ref_file.read().split()
            print(f"{len(candidate_chars)} candidate chars!")

    else:
        # If there isn't a char recognizer specified, we use the same recognizer as the main (word) recognizer
        char_recognizer_engine = EffRecognizer(
            model = os.path.join(recognizer_dir, 'enc_best.onnx'),
            transform = char_transform,
            num_cores=4
        )

        char_knn_func = FaissKNN(
            index_init_fn=faiss.IndexFlatIP,
            reset_before=False, reset_after=False
        )
        char_knn_func.load(os.path.join(recognizer_dir, "ref.index"))
        
        with open(os.path.join(recognizer_dir, "ref.txt")) as ref_file:
            candidate_chars = ref_file.read().split()
            print(f"{len(candidate_chars)} candidate chars!")

    # Build recognizer engine for word recognition, if we are doing word-level effocr
    if word_level:
        word_recognizer_engine = EffRecognizer(
            model = os.path.join(recognizer_dir, 'enc_best.onnx'),
            transform = word_transform,
            num_cores=4
        )
        word_knn_func = FaissKNN(
            index_init_fn=faiss.IndexFlatIP,
            reset_before=False, reset_after=False
        )
        word_knn_func.load(os.path.join(recognizer_dir, "ref.index"))
        with open(os.path.join(recognizer_dir, "ref.txt")) as ref_file:
            candidate_words = ref_file.read().split()
            candidate_words = [ord_str_to_word(c) for c in candidate_words]
            print(f"{len(candidate_words)} candidate words!")

    # Check recognizer word threshold
    recognizer_word_thresh = float(recognizer_word_thresh)
    assert 0 <= recognizer_word_thresh <= 1, 'Recognizer word threshold must be between 0 and 1'

    if ad_hoc_textlines:
        # Load ad hoc textlines
        line_crops = []
        for i, textline in enumerate(os.listdir(ad_hoc_textlines)):
            line_crops.append((textline.split('.')[0], cv2.imread(os.path.join(ad_hoc_textlines, textline)), None))
        inference_results, inference_coco = run_effocr_word(line_crops, localizer_engine, word_recognizer_engine, char_recognizer_engine, 
                                                                candidate_chars, candidate_words, 'en', word_knn_func, char_knn_func, num_streams=4, 
                                                                vertical=False, localizer_output = localizer_output, conf_thres=0.5, 
                                                                recognizer_thresh = recognizer_word_thresh, punc_padding = punc_padding)
        with open(os.path.join(output_save_path, "inference_results_ad_hoc.json"), "w") as f:
            json.dump(inference_results, f, indent=2)
        return
    
    img_download_session = requests.Session()

    # TODO: this should really be in an OOP format
    scan_time = 100
    pg_num = 0
    old_ed = ''
    for f_idx, f in enumerate(filenames):
        if scan_time < 5:
            time.sleep(10)
        scan_start_time = timer()
        gc.collect()
        #========== Fetch Metadata =============================================
        # start_time = timer()
        # try:
        #     if not os.path.isfile(f): # Only fetch if getting files from the Chronicling America
        #         lccn = f.strip().split('/')[-4]
        #         year_ed = f.strip().split('/')[-2]
        #         lccn_metadata = get_lccn_metadata(lccn)
        #         edition_metadata = get_edition_metadata(lccn, year_ed)
        #         page_num = find_page_number_from_filename(f.strip())
        #         if page_num is not None:
        #             scan_metadata = get_scan_metadata(lccn, year_ed[:8], year_ed[8:], page_num)
        #         else:
        #             scan_metadata = {}
        #         logging.info('Got scan metadata')

        #         metadata = {'lccn': lccn_metadata,
        #                     'edition': edition_metadata,
        #                     'page_number': page_num,
        #                     'scan': scan_metadata }
                    
        #         metadata_time = timer() - start_time
        #         logging.info('Fetch Metadata: {}'.format(metadata_time))
        #     else:
        #         metadata = {'page_number': 'na', 'scan_url': f, 'scan_ocr': 'na'}
        # except Exception as e:
        #     logging.error('Error fetching metadata: {}'.format(e))
        #     errors_log.append((f.strip(), 'Metadata', str(e)))
        #     scan_time = timer() - scan_start_time
        #     continue

        #========== CA Image Download =============================================
        metadata = {'page_number': 'na', 'scan_url': f, 'scan_ocr': 'na', 'scan':{}}
        logging.info(f.strip())

        start_time = timer()
        try:
            if os.path.isfile(f):
                if f.endswith(IMG_FILE_EXS):
                    ca_img = cv2.imread(f, cv2.IMREAD_COLOR)
                elif f.endswith('.pdf'):
                    pdfs_to_images(
                        source_path=f,
                        save_path=manifest_path,
                        data_source='na',
                        nested=False,
                        resize=False,
                        deskew=False
                    )
                    ca_img = cv2.imread(f[:-4] + '.jpg', cv2.IMREAD_COLOR)
                else:
                    print('Unknown file type for {}, only jpg, png, jp2, pdf supported!'.format(f))
                    continue
            else:
                # os.system(f'wget -O ca_img.jp2 {f.strip()}')
                # data = cv2.imread('ca_img.jp2')
                ca_batch_url = f.strip()
                lccn, reel, ed, scan = f.split('/')[-4:]

                if ed == old_ed: pg_num += 1
                else: pg_num = 1

                if len(str(pg_num)) == 1: pg_num_str = '0' + str(pg_num)
                else: pg_num_str = str(pg_num)
                date = ed[:8]
                year, month, day = date[:4], date[4:6], date[6:]
                ed_num = ed[8:]
                ca_url = f'https://chroniclingamerica.loc.gov/lccn/{lccn}/{year}-{month}-{day}/ed-{ed_num}/seq-{pg_num_str}.jp2'
                logging.info('Downloading image from {}'.format(ca_url))
                
                response = img_download_session.get(ca_url, headers = {'User-Agent': random.choice(USER_HEADERS)})
                data = response.content
                ca_img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if ca_img is None:
                    logging.info('Image not found: {}'.format(f.strip()))
                    logging.info('Response code: {}'.format(response.status_code))
                    logging.info('Response: {}'.format(response))
                    logging.info('Response headers: {}'.format(response.headers))
                    logging.info('Response content: {}'.format(response.content))
                    logging.info('Response url: {}'.format(response.url))
                    logging.info('Response text: {}', response.text)
                    continue
                else:
                    metadata['scan']['height'] = ca_img.shape[0]
                    metadata['scan']['width'] = ca_img.shape[1]
                del data
        except Exception as e:
            logging.error('Error downloading image: {}'.format(e))
            errors_log.append((f.strip(), 'Image Download', str(e)))
            scan_time = timer() - scan_start_time
            continue

        gc.collect()
        image_download_time = timer() - start_time
        logging.info('Image Download Time: {}'.format(image_download_time))

        if ca_img is None:
            print(f'Image not found: {f}')
            continue
        #========== images-to-layouts ==========================================

        start_time = timer()
        # TODO: Don't load the model in with each layout prediction!!
        try:
            crops_for_effocr, layout_crops = get_layout_predictions(layout_inf_sess, label_map_layout, ca_img, layout_output, 
                                                                f_idx, layout_input_name, backend = layout_model_backend)
        except Exception as e:
            logging.error('Error getting layout predictions: {}'.format(e))
            errors_log.append((f.strip(), 'Layout Prediction', str(e)))
            scan_time = timer() - scan_start_time
            continue

        gc.collect()

        images_to_layout_time = timer() - start_time
        logging.info(f'Images to Layouts: {images_to_layout_time}')

        #===============immediately output if not english=======================
        # if metadata['lccn']['languages'] != ['English']:
        #     metadata['bboxes'] = []

        #     for i, (layout_cls, (x0, y0, x1, y1), _) in enumerate(layout_crops):
        #         bbox_data = {
        #             'id': i,
        #             'bbox': {'x0':x0, 'y0':y0, 'x1':x1, 'y1':y1},
        #             'class': label_map_layout[int(layout_cls.item())],
        #             'raw_text': '',
        #             'legibility': 'NA'
        #         }
        #         metadata['bboxes'].append(bbox_data)

        #     with open(os.path.join(output_save_path, "{}.json".format('_'.join(filenames[f_idx].strip()[:-4].split('/')[-4:]))), "w") as f:
        #         json.dump(metadata, f, indent=2)
            
        #     continue

        # ====================== Check legibility ================================

        start_time = timer()
        legible_crops = []
        leg_dict = {}
        print('Num crops to effocr: ', len(crops_for_effocr))
        illegible_crops = []
        try:
            for i, crop in enumerate(crops_for_effocr):            
                leg_preds = legibility_classifier.run(None, {leg_input_name: legibility_transform(crop[1]).unsqueeze(0).numpy()})
                leg_dict[crop[0]] = LEGIBLE_LABEL_MAP[np.argmax(leg_preds[0][0])]
                if leg_dict[crop[0]] == 'Illegible':
                    illegible_crops.append(i)
            
            # Remove illegbile crops from the list
            for i in sorted(illegible_crops, reverse=True):
                crops_for_effocr.pop(i)
        
        except Exception as e:
            logging.error('Error getting legibility predictions: {}'.format(e))
            errors_log.append((f.strip(), 'Legibility Prediction', str(e)))
            scan_time = timer() - scan_start_time
            continue
        
        #========== layouts-to-lines ============================================

        start_time = timer()
        line_crops = []
        try:
            bbox_line_preds = get_line_predictions(line_inf_sess, line_input_name, crops_for_effocr, backend = line_model_backend)

            for bbox_idx in bbox_line_preds.keys():
                if line_output:
                    os.makedirs(os.path.join(line_output, str(f_idx), str(bbox_idx)), exist_ok=True)
                    draw = ImageDraw.Draw(layout_crops[bbox_idx][2])

                layout_img = layout_crops[bbox_idx][2]
                im_width, im_height = layout_img.size[0], layout_img.size[1]
                line_proj_crops = readjust_line_detections(bbox_line_preds[bbox_idx], im_width)
                
                for i, line_proj_crop in enumerate(line_proj_crops):
                    x0, y0, x1, y1 = line_proj_crop
                    line_crop = layout_img.crop((x0, y0, x1, y1))
                    if line_crop.size[0] == 0 or line_crop.size[1] == 0:
                        continue

                    # Line crops becomes a list of tuples (bbox_id, line_crop [the image itself], line_proj_crop [the coordinates of the line in the layout image])
                    line_crops.append((bbox_idx, np.array(line_crop).astype(np.float32), line_proj_crop))
                            
                    if line_output:
                        draw.rectangle((x0, y0, x1, y1), outline="red")
                        line_crop.save(os.path.join(line_output, str(f_idx), str(bbox_idx), 'line_{}.jpg'.format(i)))


                if line_output:
                    layout_img.save(os.path.join(line_output, f'line_boxes_{f_idx}_{bbox_idx}.jpg'))
        except Exception as e:
            logging.error('Error getting line predictions: {}'.format(e))
            errors_log.append((f.strip(), 'Line Prediction', str(e)))
            scan_time = timer() - scan_start_time
            continue
        
        logging.info('Num textlines: {}'.format(len(line_crops)))  
        layouts_to_lines_times = timer() - start_time
        logging.info(f'layouts-to-lines: {layouts_to_lines_times}')
        
        
        #========== lines-to-text ============================================
        start_time = timer()
        print('Num line crops: ', len(line_crops))
        # run the effocr pipeline
        try:
            if layout_line_only:
                inference_results, inference_coco = {}, {}
            elif word_level:
                inference_results, inference_coco = run_effocr_word(line_crops, localizer_engine, word_recognizer_engine, char_recognizer_engine, 
                                                                    candidate_chars, candidate_words, 'en', word_knn_func, char_knn_func, num_streams=4, 
                                                                    vertical=False, localizer_output = localizer_output, conf_thres=0.5, 
                                                                    recognizer_thresh = recognizer_word_thresh, punc_padding = punc_padding)
            else:
                inference_results, inference_coco = run_effocr(line_crops, localizer_engine, char_recognizer_engine, 
                                                            char_transform, 'en', num_streams = 4, vertical= False,
                                                            knn_func = char_knn_func, candidate_chars = candidate_chars,
                                                            localizer_output = localizer_output, insert_paragraph_breaks=True, 
                                                            bbox_output=bbox_output)
        except Exception as e:
            logging.error('Error running effocr pipeline: {}'.format(e))
            errors_log.append((f.strip(), 'Effocr Pipeline', str(e)))
            scan_time = timer() - scan_start_time
            continue

        
        lines_to_text_time = timer() - start_time
        logging.info(f'lines-to-text: {lines_to_text_time}')

        if spell_check:
            start_time = timer()
            # Spell checking inference results:
            inference_results_full_text = spell_check_results(inference_results, None)
            inference_results_period = spell_check_results(inference_results, '.')
            inference_results_paragraph = spell_check_results(inference_results, '\n\n')
            inference_results_word = spell_check_results(inference_results, ' ')
            spell_check_time = timer() - start_time
            logging.info(f'Spell check time: {spell_check_time}')

        metadata['bboxes'] = []

        for i, (layout_cls, (x0, y0, x1, y1), _) in enumerate(layout_crops):
            bbox_data = {
                'id': i,
                'bbox': {'x0':x0, 'y0':y0, 'x1':x1, 'y1':y1},
                'class': label_map_layout[int(layout_cls.item())],
                'raw_text': inference_results.get(i, ''),
                'legibility': leg_dict.get(i, 'NA')
            }
            if spell_check:
                bbox_data['spell_check_full_text'] = inference_results_full_text.get(i, '')
                bbox_data['spell_check_period'] = inference_results_period.get(i, '')
                bbox_data['spell_check_paragraph'] = inference_results_paragraph.get(i, '')
                bbox_data['spell_check_word'] = inference_results_word.get(i, '')
            
            metadata['bboxes'].append(bbox_data)
        

        # convert the inference results to COCO format
        if os.path.isdir(manifest_path):
            with open(os.path.join(output_save_path, os.path.basename(filenames[f_idx]).replace('jp2', 'json').replace('pdf', 'json')), 'w') as f:
                json.dump(metadata, f, indent=2)
        else:
            with open(os.path.join(output_save_path, "{}.json".format('_'.join(filenames[f_idx].strip()[:-4].split('/')[-4:]))), "w") as f:
                json.dump(metadata, f, indent=2)
        
        if bbox_output:
            with open(os.path.join(output_save_path, "inference_coco_{}.json".format(filenames[f_idx].strip().split('/')[-1][:-4])), "w") as f:
                json.dump(inference_coco, f, indent=2)

    # Save the error log as a csv
    with open(os.path.join(output_save_path, "error_table.csv"), 'w') as outfile:
        outfile.write('filename,stage,error\n')
        outfile.writelines([','.join(error) + '\n' for error in errors_log])
        
    scan_time = timer() - scan_start_time

if __name__ == '__main__':
    print("Start!")
    print('Test push')
    # gc.set_debug(gc.DEBUG_LEAK)

    #========== inputs =============================================

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_save_path",
        help="Path to directory for saving outputs of pipeline inference")
    parser.add_argument("--config_path_layout",
        help="Path to YOLOv5 config file")
    parser.add_argument("--config_path_line",
        help="Path to  YOLOv5 config file")
    parser.add_argument("--checkpoint_path_layout",
        help="Path to Detectron2 compatible checkpoint file (model weights file)")
    parser.add_argument("--checkpoint_path_line",
        help="Path to Detectron2 compatible checkpoint file (model weights file)")
    parser.add_argument("--label_map_path_layout",
        help="Path to JSON file mapping numeric object classes to their labels")
    parser.add_argument("--label_map_path_line",
        help="Path to JSON file mapping numeric object classes to their labels")
    parser.add_argument("--effocr_recognizer_dir",
        help="")
    parser.add_argument("--effocr_localizer_dir",
        help="")
    parser.add_argument("--effocr_char_recognizer_dir",
        help="")
    parser.add_argument("--spell_check", action='store_true', default=False,
        help="Run output through spell checking (symspellpy)")
    parser.add_argument("--layout-output", default=None,
        help="Path to save layout model images with detections drawn on them")
    parser.add_argument("--line-output", default=None,
        help="Path to save line model images with detections drawn on them")
    parser.add_argument("--localizer-output", default=None, 
        help="Path to save localizer outputs with detections")
    parser.add_argument("--localizer_iou_thresh", default=0.10)
    parser.add_argument("--localizer_conf_thresh", default=0.15)
    parser.add_argument("--recognizer_word_thresh", default=0.86)
    parser.add_argument("--manifest_path", default='manifest_0.txt')
    parser.add_argument("--first_n", default=None, type=int)
    parser.add_argument("--layout-line-only", action='store_true', default=False)
    parser.add_argument("--legibility-classifier", type=str, default=None)
    parser.add_argument("--bbox_output", action='store_true', default=False)
    parser.add_argument("--word-level-effocr", action='store_true', default=False)
    parser.add_argument("--localizer_model_backend", type=str, default="yolo")
    parser.add_argument("--line_model_backend", type=str, default="yolo")
    parser.add_argument("--layout_model_backend", type=str, default='yolo')
    parser.add_argument("--ad_hoc_textlines", type=str, default=None)
    parser.add_argument("--punc_padding", type=int, default=0)

    args = parser.parse_args()
    main(args)

    
