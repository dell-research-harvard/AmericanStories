from stages.images_to_layouts import get_onnx_input_name, letterbox, non_max_suppression
from effocr.dataset_utils import create_paired_transform
from effocr.infer_ocr_onnx_multi import run_effocr
from effocr.engines.localizer_engine import EffLocalizer
from effocr.engines.recognizer_engine import EffRecognizer
from generate_manifest import *

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
from symspellpy import SymSpell
import multiprocessing
import time
from math import floor, ceil
from models.encoders import *
from pytorch_metric_learning.utils.inference import FaissKNN
from pytorch_metric_learning.utils import common_functions as c_f
from torchvision.ops import nms

    
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
    keep_preds = nms(all_preds[:, :4], all_preds[:, -1], iou_threshold=0.15)
    filtered_preds = all_preds[keep_preds, :4]
    filtered_preds = filtered_preds[filtered_preds[:, 1].sort()[1]]
    for pred in filtered_preds:
        x0, y0, x1, y1 = torch.round(pred)
        x0, y0, x1, y1 = x0.item(), y0.item(), x1.item(), y1.item()
        final_preds.append((x0, y0, x1, y1))
    return final_preds
    
def get_line_predictions(checkpoint_path_line, crops_for_effocr):

    # Get line model input names:
    base_model = onnx.load(checkpoint_path_line)
    input_name = get_onnx_input_name(base_model)
    del base_model

    #GUPPY - mnt/data02/e2e2e/line_det....
    line_model = ort.InferenceSession(checkpoint_path_line)
    bbox_preds = {}
    bbox_idx = 0
    for line_bbox in crops_for_effocr:
        gc.collect()
        im = letterbox(np.array(line_bbox), (640, 640), auto=False)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.expand_dims(np.ascontiguousarray(im), axis = 0).astype(np.float32) / 255.0  # contiguous

        line_predictions = line_model.run(
            None,
            {input_name: im}
        )

        line_predictions = torch.from_numpy(line_predictions[0])
        line_predictions = non_max_suppression(line_predictions, conf_thres = 0.2, iou_thres=0.15, max_det=200)[0]
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

def spell_check_results(results):
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt"
    )
    bigram_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_bigramdictionary_en_243_342.txt"
    )

    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

    for k in results.keys():
        try:
            suggestions = sym_spell.lookup_compound(results[k], max_edit_distance=2, transfer_casing=True)
            results[k] = suggestions[0].term
        except:
            pass

    return results

def main(args):

    output_save_path = args.output_save_path
    img_save_path = os.path.join(output_save_path, "images")
    checkpoint_path_layout = args.checkpoint_path_layout
    checkpoint_path_line = args.checkpoint_path_line

    label_map_path_layout = args.label_map_path_layout

    recognizer_dir = args.effocr_recognizer_dir
    localizer_dir = args.effocr_localizer_dir

    spell_check = args.spell_check

    layout_output = args.layout_output
    line_output = args.line_output
    localizer_output = args.localizer_output
    inference_dir = args.inference_dir

    os.makedirs(output_save_path, exist_ok=True)

    effocr_results = {}

    for f in tqdm(os.listdir(inference_dir)):
        gc.collect()
        

        #========== layouts-to-text ============================================
        crops_for_effocr = [Image.open(os.path.join(inference_dir, f))]
        start_time = timer()
        bbox_preds = get_line_predictions(checkpoint_path_line, crops_for_effocr)
        gc.collect()
        line_crops = []

        for bbox_idx in bbox_preds.keys():
            img = Image.open(os.path.join(inference_dir, f))
            im_width, im_height = img.size[0], img.size[1]
            line_proj_crops = readjust_line_detections(bbox_preds[bbox_idx], im_width)
            
            if line_output:
                draw = ImageDraw.Draw(img)
                os.makedirs(os.path.join(line_output, str(bbox_idx)), exist_ok=True)

            
            for line_proj_crop in line_proj_crops:
                x0, y0, x1, y1 = line_proj_crop
                line_crop = img.crop((x0, y0, x1, y1))
                line_crops.append((bbox_idx, np.array(line_crop).astype(np.float32)))
                        
                if line_output:
                    draw.rectangle((x0, y0, x1, y1), outline="red")
                    # line_crop.save(os.path.join(line_output, str(bbox_idx), 'line_{}.jpg'.format(i)))

            if line_output:
                img.save(os.path.join(line_output, f'line_boxes_{f_idx}_{bbox_idx}.jpg'))
        
        #========== lines-to-text ============================================
        start_time = timer()

            # confirm that the ONNX models we're expecting are in fact present:
        assert os.path.exists(os.path.join(recognizer_dir, 'enc_best.onnx')), 'Recognizer model not found! Should be in {}/enc_best.onnx'.format(recognizer_dir)
        assert os.path.exists(os.path.join(localizer_dir, 'localizer_model.onnx')), 'Localizer model not found! Should be in {}/localizer_model.onnx'.format(localizer_dir)

        #Create localizer engine
        localizer_engine = EffLocalizer(
            os.path.join(localizer_dir, 'localizer_model.onnx'),
            iou_thresh=0.07,
            conf_thresh=0.27,
            vertical=False,
            num_cores=multiprocessing.cpu_count(),
            model_backend='yolo',
            input_shape=(640, 640)
        )

        char_transform = create_paired_transform(lang='en')

        recognizer_engine = EffRecognizer(
            model = os.path.join(recognizer_dir, 'enc_best.onnx'),
            transform = char_transform,
            num_cores=4
        )

        knn_func = FaissKNN(
            index_init_fn=faiss.IndexFlatIP,
            reset_before=False, reset_after=False
        )
        knn_func.load(os.path.join(recognizer_dir, "ref.index"))

        with open(os.path.join(recognizer_dir, "ref.txt")) as ref_file:
            candidate_chars = ref_file.read().split()
            candidate_chars_dict = {c:idx for idx, c in enumerate(candidate_chars)}

        logging.info(f'Loading effocr models: {timer() - start_time}')

        start_time = timer()
        # run the effocr pipeline
        inference_results, inference_coco = run_effocr(line_crops, localizer_engine, recognizer_engine, 
                                                    char_transform, 'en', num_streams = 4, vertical= False,
                                                    knn_func = knn_func, candidate_chars = candidate_chars,
                                                    localizer_output = localizer_output, insert_paragraph_breaks = True)

        
        lines_to_text_time = timer() - start_time
        logging.info(f'lines-to-text: {lines_to_text_time}')

        try:
            effocr_results[f] = inference_results[0]   
        except KeyError:
            print(inference_results)

    with open(os.path.join(output_save_path, 'effocr_results.json'), 'w') as f:
        json.dump(effocr_results, f, indent=4)

        
if __name__ == '__main__':
    print("Start!")
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
    parser.add_argument("--data_source", choices=['newspaper_archive', 'loc'],
        help="Specifies the type of newspaper data being worked with; currently only 'newspaper_archive'")
    parser.add_argument("--filter_duplicates",
        help="Filter out duplicate scans within newspaper editions",
        action='store_true')
    parser.add_argument("--viz_ocr_texts",
        help="Output visualizations of OCR text in the context of a scan's layout",
        action='store_true')
    parser.add_argument("--full_articles_out",
        help="Generate and output full newspaper articles (i.e., headline, article body) using rule-based reading order predictions",
        action='store_true')
    parser.add_argument("--language", choices=['en', 'jp'], default='en',
        help="Language the input scans are in")
    parser.add_argument("--effocr_recognizer_dir",
        help="")
    parser.add_argument("--effocr_localizer_dir",
        help="")
    parser.add_argument('--tesseract', action='store_true', default=False,
        help="")
    parser.add_argument('--nonnested', action='store_false', default=True,
        help="")
    parser.add_argument('--trocr', action='store_true', default=False,
        help="")
    parser.add_argument('--saving', action='store_true', default=False,
        help="")
    parser.add_argument('--resize', action='store_true', default=False,
        help="")
    parser.add_argument("--device", default='cuda',
        help="")
    parser.add_argument("--spell_check", action='store_true', default=False,
        help="Run output through spell checking (symspellpy)")
    parser.add_argument("--spell_check_jit", default=False,
        help="Call a sped up version of spell checking with JIT compelation through pypy")
    parser.add_argument("--spell_check_homoglyphs_path", default=None,
        help="Path to dictionary storing scores with visual similariy between characters")
    parser.add_argument("--save_every", default=10,
        help="Specify how large of groups to save output in. Used primarily if using --saving with a large batch of input files (not recommended)")
    parser.add_argument("--layout-output", default=None,
        help="Path to save layout model images with detections drawn on them")
    parser.add_argument("--line-output", default=None,
        help="Path to save line model images with detections drawn on them")
    parser.add_argument("--localizer-output", default=None, 
        help="Path to save localizer outputs with detections")
    parser.add_argument("--localizer-iou_thresh", default=0.07)
    parser.add_argument("--localizer-conf-thresh", default=0.27)
    parser.add_argument("--inference_dir", default=None)

    args = parser.parse_args()
    main(args)