import torch
from torchvision import transforms as T
import numpy as np
import queue
from collections import defaultdict
import threading
from glob import glob
import os
import sys
from PIL import Image, ImageDraw
import time

sys.path.insert(0, "../")
from utils.datasets_utils import *
from datasets.effocr_datasets import *
from utils.localizer_utils import *
from utils.coco_utils import *
from utils.spell_check_utils import *

LARGE_NUMBER = 1000000000
PARAGRAPH_BREAK = "\n\n"
PARA_WEIGHT_L = 3
PARA_WEIGHT_R = 1
PARA_THRESH = 5
ERROR_TEXT = 'XXX_ERROR_XXX'

def en_preprocess(bboxes_char, bboxes_word, vertical=False):

    sorted_bboxes_char = sorted(bboxes_char, key=lambda x: x[1] if vertical else x[0])
    sorted_bboxes_word = sorted(bboxes_word, key=lambda x: x[1] if vertical else x[0])

    word_end_idx = []
    closest_idx = 0
    sorted_bboxes_char_rights = [x[2] for x in sorted_bboxes_char]
    sorted_bboxes_word_lefts = [x[0] for x in sorted_bboxes_word]
    for wordleft in sorted_bboxes_word_lefts:
        prev_dist = LARGE_NUMBER
        for idx, charright in enumerate(sorted_bboxes_char_rights):
            dist = abs(wordleft-charright)
            if dist < prev_dist and charright > wordleft:
                prev_dist = dist
                closest_idx = idx
        word_end_idx.append(closest_idx)
    assert len(word_end_idx) == len(sorted_bboxes_word)

    return sorted_bboxes_char, word_end_idx

def en_postprocess(line_output, word_end_idx, charheights, charbottoms, anchor_margin=None, anchor_multiplier = 4):

    assert len(line_output) == len(charheights) == len(charbottoms), f"{len(line_output)} == {len(charheights)} == {len(charbottoms)}; {line_output}; {charbottoms}; {charheights}"

    if any(map(lambda x: len(x)==0, (line_output, word_end_idx, charheights, charbottoms))):
        return None

    outchars_w_spaces = [" " + x if idx in word_end_idx else x for idx, x in enumerate(line_output)]
    charheights_w_spaces = list(flatten([(LARGE_NUMBER, x) if idx in word_end_idx else x for idx, x in enumerate(charheights)]))
    charbottoms_w_spaces = list(flatten([(0, x) if idx in word_end_idx else x for idx, x in enumerate(charbottoms)]))
    charbottoms_w_spaces = charbottoms_w_spaces[1:] if charbottoms_w_spaces[0]==0 else charbottoms_w_spaces
    charheights_w_spaces = charheights_w_spaces[1:] if charheights_w_spaces[0]==LARGE_NUMBER else charheights_w_spaces

    line_output = "".join(outchars_w_spaces).strip()

    assert len(charheights_w_spaces) == len(line_output), \
        f"charheights_w_spaces = {len(charheights_w_spaces)}; output = {len(line_output)}; {charheights_w_spaces}; {line_output}"

    output_distinct_lower_idx = [idx for idx, c in enumerate(line_output) if c in create_distinct_lowercase()]

    if len(output_distinct_lower_idx) > 0 and not anchor_margin is None:
        avg_distinct_lower_height = sum(charheights_w_spaces[idx] for idx in output_distinct_lower_idx) / len(output_distinct_lower_idx)
        output_tolower_idx = [idx for idx, c in enumerate(line_output) \
            if abs(charheights_w_spaces[idx] - avg_distinct_lower_height) < anchor_margin * avg_distinct_lower_height]
        output_toupper_idx = [idx for idx, c in enumerate(line_output) \
            if charheights_w_spaces[idx] - avg_distinct_lower_height > anchor_margin * anchor_multiplier * avg_distinct_lower_height]
        avg_distinct_lower_bottom = sum(charbottoms_w_spaces[idx] for idx in output_distinct_lower_idx) / len(output_distinct_lower_idx)
        output_toperiod_idx = [idx for idx, c in enumerate(line_output) \
            if c == "-" and abs(charbottoms_w_spaces[idx] - avg_distinct_lower_bottom) < anchor_margin * avg_distinct_lower_height]

    # if self.spell_check:
    #     line_output = visual_spell_checker(line_output, WORDDICT, SIMDICT, ABBREVSET)

    if len(output_distinct_lower_idx) > 0 and not anchor_margin is None:
        nondistinct_lower = create_nondistinct_lowercase()
        line_output = "".join([c.lower() if idx in output_tolower_idx else c for idx, c in enumerate(line_output)])
        line_output = "".join([c.upper() if idx in output_toupper_idx and c in nondistinct_lower else c for idx, c in enumerate(line_output)])
        line_output = "".join(["." if idx in output_toperiod_idx else c for idx, c in enumerate(line_output)])

    if line_output is None:
        return " "
    else:
        return line_output

def create_batches(data, batch_size = 64):
    """Create batches for inference"""

    batches = []
    batch = []
    count = 0
    for i, d in enumerate(data):
        if d is not None:
            batch.append(d)
        else:
            batch.append(np.zeros((33, 33, 3), dtype=np.float32))
            count += 1
        if (i+1) % batch_size == 0:
            batches.append(batch)
            batch = []
    if len(batch) > 0:
        batches.append(batch)

    return [b for b in batches]

def yolo_to_orig_coords(coords, im_width, im_height, vertical=True):
    x0, y0, x1, y1 = coords            
    if vertical:
        # Math is to convert from YOLO input size (640x640) to source image size
        # Also note we are automatically taking the entire width of the textline
        x0, y0, x1, y1 = 0, int(round(y0.item() * im_height / 640)), im_width, int(round(y1.item() * im_height / 640))
    else:
        # In this case we take the entire height of the textline
        x0, y0, x1, y1 = int(round(x0.item() * im_width / 640)), 0, int(round(x1.item() * im_width / 640)), im_height

    return x0, y0, x1, y1

def iteration(model, input):
    output = model.run(input)
    return output, output

class LocalizerEngineExecutorThread(threading.Thread):
    def __init__(
        self,
        model,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
    ):
        super(LocalizerEngineExecutorThread, self).__init__()
        self._model = model
        self._input_queue = input_queue
        self._output_queue = output_queue

    def run(self):
        while not self._input_queue.empty():
            img_idx, bbox_idx, img = self._input_queue.get()
            output = iteration(self._model, [img])
            self._output_queue.put((img_idx, bbox_idx, output))

class RecognizerEngineExecutorThread(threading.Thread):
    def __init__(
        self,
        model,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
    ):
        super(RecognizerEngineExecutorThread, self).__init__()
        self._model = model
        self._input_queue = input_queue
        self._output_queue = output_queue

    def run(self):
        while not self._input_queue.empty():
            i, batch = self._input_queue.get()
            output = iteration(self._model, batch)
            self._output_queue.put((i, output))

def blank_layout_response():
    return {-1: ''}

def blank_dists_response():
    return {'l_dists': {}, 'r_dists': {}}

def add_paragraph_breaks_to_dict(inference_assembly, side_dists):
    for k, v in inference_assembly.items():
        l_list, r_list = [], []
        im_ids = sorted(list(side_dists[k]['l_dists'].keys()))
        for i in im_ids:
            l_list.append(side_dists[k]['l_dists'][i])
            r_list.append(side_dists[k]['r_dists'][i])
        
        try:
            l_avg = sum(filter(None, l_list)) / (len(l_list) - l_list.count(None))
            r_avg = sum(filter(None, r_list)) / (len(r_list) - r_list.count(None))
        except ZeroDivisionError:
            print("ZeroDivisionError: l_list: {}, r_list: {}".format(l_list, r_list))
            print(f'side_dists: {side_dists[k]}')
            print(f'im_ids: {im_ids}')
            print(f'l_avg: {l_avg}, r_avg: {r_avg}')
            continue   

        l_list = [l_avg if l is None else l for l in l_list]
        r_list = [r_avg if r is None else r for r in r_list]
        r_max = max(r_list)
        r_avg = r_max - r_avg

        l_list = [l / l_avg for l in l_list]
        try:
            r_list = [(r_max - r) / r_avg for r in r_list]
        except ZeroDivisionError:
            r_list = [0] * len(r_list)
            
        for i in range(len(l_list) - 1):
            score = l_list[i + 1] * PARA_WEIGHT_L + r_list[i] * PARA_WEIGHT_R
            if score > PARA_THRESH:
                inference_assembly[k][im_ids[i]] += PARAGRAPH_BREAK

    return inference_assembly


def run_effocr(coco_images, localizer_engine, recognizer_engine, char_transform, lang, num_streams=4, 
                            vertical=False, localizer_output = None, conf_thres=0.5, knn_func = None,
                            candidate_chars = None, iou_thresh=0.5, insert_paragraph_breaks = False, bbox_output=False):

    # Set up eventual output -- inference_results will have the final output, inference_assembly will have the intermediate output
    inference_results, inference_assembly = {}, defaultdict(blank_layout_response)
    # inference_bboxes will save bounding boxes for each detected word and character crop, if requested
    inference_bboxes = defaultdict(dict)
    
    '''''''''''''''
    Localizer Inference
    '''''''''''''''
    input_queue = queue.Queue()
    for im_idx, (bbox_idx, p, coords) in enumerate(coco_images):
        input_queue.put((im_idx, bbox_idx, p))
        if bbox_output:  # Start detections with empty list for each textline image
            inference_bboxes[bbox_idx][im_idx] = {'bbox': coords, 'detections': {'words': [], 'chars': []}}
    output_queue = queue.Queue()
    threads = []

    for thread in range(num_streams):
        threads.append(LocalizerEngineExecutorThread(localizer_engine, input_queue, output_queue))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    '''''''''''''''
    Localizer Postprocessing
    '''''''''''''''

    # Initialize variables
    char_crops, word_end_idxs, n_chars = [], [], []
    charheights, charbottoms, coco_new_order = [], [], []
    side_dists = defaultdict(blank_dists_response)

    while not output_queue.empty():
        # Get all the output off the queue
        im_idx, bbox_idx, result = output_queue.get()
        coco_new_order.append((im_idx, bbox_idx))
        im = coco_images[im_idx][1]
        
        # Get bounding boxes and labels, agnostic to the backend used 
        if localizer_engine._model_backend == 'yolo' or localizer_engine._model_backend == 'yolov8':
            result = result[0][0]
            bboxes, labels = result[:, :4], result[:, -1]
        elif localizer_engine._model_backend == 'detectron2':
            result = result[0][0]
            bboxes, labels = result[0][result[3] > conf_thres], result[1][result[3] > conf_thres]
            bboxes, labels = torch.from_numpy(bboxes), torch.from_numpy(labels)
        elif localizer_engine._model_backend == 'mmdetection':
            result = result[0][0]
            bboxes, labels = result[0][result[0][:, -1] > conf_thres], result[1][result[0][:, -1] > conf_thres]
            bboxes = bboxes[:, :-1]
            bboxes, labels = torch.from_numpy(bboxes), torch.from_numpy(labels)


        if lang == "en":
            # Get all the character and word crops
            char_bboxes, word_bboxes = bboxes[labels == 0], bboxes[labels == 1]

            if len(char_bboxes) != 0: # If there is output, process it into crops and word bounds
                char_bboxes, word_end_idx = en_preprocess(char_bboxes, word_bboxes)
                l_dist, r_dist = char_bboxes[0][0].item(), char_bboxes[-1][-2].item()
                n_chars.append(len(char_bboxes)); word_end_idxs.append(word_end_idx) # Keep track of how many characters are in the line
                side_dists[bbox_idx]['l_dists'][im_idx] = l_dist # Store distances for paragraph detection
                side_dists[bbox_idx]['r_dists'][im_idx] = r_dist

            else:
                n_chars.append(0); word_end_idxs.append([])
                side_dists[bbox_idx]['l_dists'][im_idx] = None; side_dists[bbox_idx]['r_dists'][im_idx] = None
        elif lang == "jp":
            char_bboxes = bboxes[labels == 0] #there should be no other boxes, but have this just in case
            if len(char_bboxes) != 0:
                char_bboxes = jp_preprocess(char_bboxes, vertical=vertical)
                n_chars.append(len(char_bboxes))
            else:
                n_chars.append(0)

        # If requested, save localizer output to file
        if localizer_output:
            img = Image.fromarray(im.astype(np.uint8))
            im_width, im_height = img.size[0], img.size[1]
            draw = ImageDraw.Draw(img)
            os.makedirs(os.path.join(localizer_output, str(bbox_idx)), exist_ok=True)
            for i, bbox in enumerate(char_bboxes):
                x0, y0, x1, y1 = torch.round(bbox)
                if vertical:
                    x0, y0, x1, y1 = 0, int(round(y0.item() * im_height / 640)), im_width, int(round(y1.item() * im_height / 640))
                else:
                    x0, y0, x1, y1 = int(round(x0.item() * im_width / 640)), 0, int(round(x1.item() * im_width / 640)), im_height

                draw.rectangle((x0, y0, x1, y1), outline="red")

            img.save(os.path.join(localizer_output, str(bbox_idx), f'localizer_boxes_{im_idx}.jpg'))

        # Crop out characters and save to list of char crops
        im_height, im_width = im.shape[0], im.shape[1]
        for i, bbox in enumerate(char_bboxes):

            # Round coordinates to integers and convert from YOLO detections to original image coordinates            
            x0, y0, x1, y1 = yolo_to_orig_coords(torch.round(bbox), im_width, im_height, vertical)

            # Add to main list of character crops
            char_crops.append(im[y0:y1, x0:x1, :])
            
            # If requested, add detections to bounding box output
            if bbox_output:
                inference_bboxes[bbox_idx][im_idx]['detections']['chars'].append({
                    'bbox': [x0, y0, x1, y1],
                    'text': '',
                    'id': i
                })
            
            # Add relevant height and bottom information to the lists
            if lang == "en":
                charheights.append(bbox[3]-bbox[1])
                charbottoms.append(bbox[3])

        # If requested, add word bounding boxes to detection list
        for i, word_bbox in enumerate(sorted(word_bboxes, key=lambda x: x[0])):
            x0, y0, x1, y1 = yolo_to_orig_coords(torch.round(word_bbox), im_width, im_height, vertical)
            if bbox_output:
                inference_bboxes[bbox_idx][im_idx]['detections']['words'].append({
                    'bbox': [x0, y0, x1, y1],
                    'text': '',
                    'id': i
                })
    
    '''''''''''''''
    Recognizer Inference
    '''''''''''''''
    
    # Batch character crops
    char_crop_batches = create_batches(char_crops)

    #Now run the crops through the recognizer
    input_queue = queue.Queue()
    for i, batch in enumerate(char_crop_batches):
        input_queue.put((i, batch))
    num_batches = len(char_crop_batches)
    del char_crop_batches

    output_queue = queue.Queue()
    threads = []

    for thread in range(num_streams):
        threads.append(RecognizerEngineExecutorThread(recognizer_engine, input_queue, output_queue))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # Get embeddings off the queue
    embeddings = [None] * num_batches
    while not output_queue.empty():
        i, result = output_queue.get()
        embeddings[i] = result[0][0]

    # Normalize embeddings and get nearest neighbors
    embeddings = [torch.nn.functional.normalize(torch.from_numpy(embedding), p=2, dim=1) for embedding in embeddings]
    indices = [knn_func(embedding, k=1)[1] for embedding in embeddings]
    
    index_list = [index.squeeze(-1).tolist() for index in indices]
    indices = [item for sublist in index_list for item in sublist]
    nn_outputs = [candidate_chars[idx] for idx in indices]

    '''''''''''''''
    Recognizer Postprocessing
    '''''''''''''''
    idx, textline_outputs, textline_bottoms, textline_heights = 0, [], [], []
    # For each textline length (in n_chars), get the corresponding nearest neighbor outputs from the recognizer
    for l in n_chars:
        textline_outputs.append(nn_outputs[idx:idx+l])
        textline_bottoms.append(charbottoms[idx:idx+l])
        textline_heights.append(charheights[idx:idx+l])
        idx += l

    #  Assemble textlines
    outputs = ["".join(x[0] for x in textline).strip() for textline in textline_outputs]

    # Postprocess textlines, saving to inference_assembly
    if lang == "en":
        for i, (im_idx, bbox_idx) in enumerate(coco_new_order):
            inference_assembly[bbox_idx][im_idx] = en_postprocess(outputs[i], word_end_idxs[i], textline_heights[i], textline_bottoms[i])
            if inference_assembly[bbox_idx][im_idx] is None:
                inference_assembly[bbox_idx][im_idx] = " "
            
            # If requested, add text to bounding box output
            if bbox_output:
                words = inference_assembly[bbox_idx][im_idx].split()
                chars = inference_assembly[bbox_idx][im_idx].replace(" ", "")
                if len(chars) == len(inference_bboxes[bbox_idx][im_idx]['detections']['chars']) and len(words) == len(inference_bboxes[bbox_idx][im_idx]['detections']['words']):
                    for i in range(len(inference_bboxes[bbox_idx][im_idx]['detections']['chars'])):
                        inference_bboxes[bbox_idx][im_idx]['detections']['chars'][i]['text'] = chars[i]
                    for i in range(len(inference_bboxes[bbox_idx][im_idx]['detections']['words'])):
                        inference_bboxes[bbox_idx][im_idx]['detections']['words'][i]['text'] = words[i]
                else:
                    for i in range(len(inference_bboxes[bbox_idx][im_idx]['detections']['chars'])):
                        inference_bboxes[bbox_idx][im_idx]['detections']['chars'][i]['text'] = ERROR_TEXT
                    for i in range(len(inference_bboxes[bbox_idx][im_idx]['detections']['words'])):
                        inference_bboxes[bbox_idx][im_idx]['detections']['words'][i]['text'] = ERROR_TEXT
    else:
        for i, idx in enumerate(coco_new_order):
            inference_assembly[bbox_idx][im_idx] = outputs[i]
    
    # Add paragraph breaks if requested
    if insert_paragraph_breaks:
        inference_assembly = add_paragraph_breaks_to_dict(inference_assembly, side_dists)
    
    # Use intermediate data to complete inference_results, keyed by layout box
    try:
        inference_results = {bbox_idx: ' '.join([inference_assembly[bbox_idx][i] for i in sorted([int(x) for x in inference_assembly[bbox_idx].keys()])]) for 
                                                            bbox_idx in inference_assembly.keys()}
    except TypeError as e:
        print(e)
        print(inference_assembly)
        inference_results = {}

    return inference_results, inference_bboxes

