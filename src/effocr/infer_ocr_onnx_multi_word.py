import logging
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
END_PUNCTUATION = '.?!,;:"'

def check_any_overlap(bbox_1, bbox_2):
    """Check if two bboxes overlap, we do this by checking all four corners of bbox_1 against bbox_2"""

    x1, y1, x2, y2 = bbox_1
    x3, y3, x4, y4 = bbox_2

    return x1 < x4 and x2 > x3 and y1 < y4 and y2 > y3

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
                inference_assembly[k][im_ids[i]]['text'] += PARAGRAPH_BREAK

    return inference_assembly

def en_preprocess(bboxes_char, bboxes_word, vertical=False):

    sorted_bboxes_char = sorted(bboxes_char, key=lambda x: x[1] if vertical else x[0])
    sorted_bboxes_word = sorted(bboxes_word, key=lambda x: x[1] if vertical else x[0])

    # For each word, find all chars it overlaps with
    word_char_idx = []
    for word_bbox in sorted_bboxes_word:
        word_char_idx.append([])
        for char_idx, char_bbox in enumerate(sorted_bboxes_char):
            if check_any_overlap(word_bbox, char_bbox):
                word_char_idx[-1].append(char_idx)

    # If there are no overlapping chars for a word, append the word bounding box to the list of chars as a char
    redo_list = False
    for i, word_bbox in enumerate(sorted_bboxes_word):
        if len(word_char_idx[i]) == 0:
            sorted_bboxes_char.append(word_bbox)
            word_char_idx[i].append(len(sorted_bboxes_char) - 1)
            redo_list = True

    # If we found a word with no overlapping chars, we now need to resort the char list and recreate the word_char_idx list
    if redo_list:
        # Resort the sorted_bboxes_char list and adjust the word_char_idx list accordingly
        sorted_bboxes_char = sorted(sorted_bboxes_char, key=lambda x: x[1] if vertical else x[0])
        word_char_idx = []
        for word_bbox in sorted_bboxes_word:
            word_char_idx.append([])
            for char_idx, char_bbox in enumerate(sorted_bboxes_char):
                if check_any_overlap(word_bbox, char_bbox):
                    word_char_idx[-1].append(char_idx)

    # Return the lists of chars, words, and overlaps
    return sorted_bboxes_char, sorted_bboxes_word, word_char_idx

def create_batches(data, batch_size = 64, transform = None):
    """Create batches for inference"""

    batches = []
    batch = []
    for i, d in enumerate(data):
        if d is not None:
            batch.append(d)
        else:
            batch.append(np.zeros((33, 33, 3), dtype=np.int8))
        if (i+1) % batch_size == 0:
            batches.append(batch)
            batch = []
    if len(batch) > 0:
        batches.append(batch)
    return [b for b in batches]

def get_crop_embeddings(recognizer_engine, crops, num_streams=4):
    # Create batches of word crops
    crop_batches = create_batches(crops)

    input_queue = queue.Queue()
    for i, batch in enumerate(crop_batches):
        input_queue.put((i, batch))
    output_queue = queue.Queue()
    threads = []

    for thread in range(num_streams):
        threads.append(RecognizerEngineExecutorThread(recognizer_engine, input_queue, output_queue))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    embeddings = [None] * len(crop_batches)
    while not output_queue.empty():
        i, result = output_queue.get()
        embeddings[i] = result[0][0]

    embeddings = [torch.nn.functional.normalize(torch.from_numpy(embedding), p=2, dim=1) for embedding in embeddings]
    return embeddings

def iteration(model, input):
    output = model.run(input)
    return output, output

''' Threaded Localizer Inference'''
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

'''Threaded Recognizer Inference'''
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

''' Main Function for Running EffOCR on a set of textline images'''
def run_effocr_word(textline_images, localizer_engine, recognizer_engine, char_recognizer_engine, candidate_chars, candidate_words, lang, 
                    word_knn_func, char_knn_func, num_streams=4, vertical=False, localizer_output = None, conf_thres=0.5, recognizer_thresh = 0.5, 
                    bbox_output = False, punc_padding = 0, insert_paragraph_breaks = True):
    
    start_time = time.time()
    inference_results = {}
    inference_assembly = defaultdict(blank_layout_response)
    inference_bboxes = defaultdict(dict)
    image_id, anno_id = 0, 0
    
    print(len(textline_images))
    input_queue = queue.Queue()
    for im_idx, (bbox_idx, p, coords) in enumerate(textline_images):
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

    word_crops, char_crops, n_words, n_chars = [], [], [], []
    all_word_bboxes, word_rec_types, coco_new_order = [None] * len(textline_images), [None] * len(textline_images), []
    word_char_overlaps, last_char_crops = [], []
    side_dists = defaultdict(blank_dists_response)

    while not output_queue.empty():
        im_idx, bbox_idx, result = output_queue.get()
        coco_new_order.append((im_idx, bbox_idx))
        im = textline_images[im_idx][1]
        if localizer_output:
            os.makedirs(os.path.join(localizer_output, str(bbox_idx)), exist_ok=True)
        
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
            char_bboxes, word_bboxes = bboxes[labels == 0], bboxes[labels == 1]

            if len(word_bboxes) != 0:
                char_bboxes, word_bboxes, word_char_overlap = en_preprocess(char_bboxes, word_bboxes)
                word_char_overlaps.append(word_char_overlap)
                n_words.append(len(word_bboxes))
            else:
                n_words.append(0)
                word_char_overlaps.append([])

            if len(char_bboxes) != 0:
                l_dist, r_dist = char_bboxes[0][0].item(), char_bboxes[-1][-2].item()
                side_dists[bbox_idx]['l_dists'][im_idx] = l_dist # Store distances for paragraph detection
                side_dists[bbox_idx]['r_dists'][im_idx] = r_dist
                n_chars.append(len(char_bboxes))
            else:
                n_chars.append(0)
                side_dists[bbox_idx]['l_dists'][im_idx] = None; side_dists[bbox_idx]['r_dists'][im_idx] = None

        if localizer_output:
            img = Image.fromarray((im * -255).astype(np.uint8))
            im_width, im_height = img.size[0], img.size[1]
            draw = ImageDraw.Draw(img)
            for i, bbox in enumerate(char_bboxes):
                x0, y0, x1, y1 = torch.round(bbox)
                if vertical:
                    x0, y0, x1, y1 = 0, int(round(y0.item() * im_height / 640)), im_width, int(round(y1.item() * im_height / 640))
                else:
                    x0, y0, x1, y1 = int(round(x0.item() * im_width / 640)), 0, int(round(x1.item() * im_width / 640)), im_height

                # draw.rectangle([x0, y0, x1, y1], outline='red')
                # char_crop = im.crop((x0, y0, x1, y1))
                # char_crop.save(os.path.join(localizer_output, os.path.basename(path)[:-4], 'char_{}.jpg'.format(i)))
            

            img = Image.fromarray((im * -255).astype(np.uint8))
            draw = ImageDraw.Draw(img)
            all_word_bboxes[im_idx] = []
            for i, bbox in enumerate(word_bboxes):
                x0, y0, x1, y1 = torch.round(bbox)
                if vertical:
                    x0, y0, x1, y1 = 0, int(round(y0.item() * im_height / 640)), im_width, int(round(y1.item() * im_height / 640))
                else:
                    x0, y0, x1, y1 = int(round(x0.item() * im_width / 640)), 0, int(round(x1.item() * im_width / 640)), im_height

                # draw.rectangle([x0, y0, x1, y1], outline='red')
                all_word_bboxes[im_idx].append((x0, y0, x1, y1))

        im_height, im_width = im.shape[0], im.shape[1]
        for i, bbox in enumerate(word_bboxes):
            x0, y0, x1, y1 = torch.round(bbox)
            if vertical:
                x0, y0, x1, y1 = 0, int(round(y0.item() * im_height / 640)), im_width, int(round(y1.item() * im_height / 640))
            else:
                x0, y0, x1, y1 = int(round(x0.item() * im_width / 640)), 0, int(round(x1.item() * im_width / 640)), im_height

            # Verify that the crop is not empty
            if x0 == x1 or y0 == y1 or x0 < 0:
                # If so, eliminate the corresponding entry in the word_char_overlaps list
                word_char_overlaps[-1].pop(i)
                n_words[-1] -= 1
            else:
                word_crops.append(im[y0:y1, x0:x1, :])
        
        last_chars = [overlaps[-1] for overlaps in word_char_overlaps[-1]]
        for i, bbox in enumerate(char_bboxes):
            x0, y0, x1, y1 = torch.round(bbox)
            if vertical:
                x0, y0, x1, y1 = 0, int(round(y0.item() * im_height / 640)), im_width, int(round(y1.item() * im_height / 640))
            else:
                x0, y0, x1, y1 = int(round(x0.item() * im_width / 640)), 0, int(round(x1.item() * im_width / 640)), im_height

            char_crops.append(im[y0:y1, x0:x1, :])

            # if i in last_chars:
            #     last_char_crops.append(im[y0 - punc_padding:y1 + punc_padding, x0-punc_padding:x1+punc_padding, :])

    print('Word crops: ', len(word_crops))
    print('Char crops: ', len(char_crops))
    ''' --- Last Character Recognition ---'''
    # This is an easy way to increase accuracy for punctuation by a lot-- we check the last character of every word
    # (where punctuation is much more likely to appear) to see if it is a punctuation mark. If so, we adjust the word bounding box
    # And save the punctuation mark to be appended to the word later on. 

    # Collect the last character crop from each word
    last_chars = [[overlap[-1] for overlap in word_char_overlap] for word_char_overlap in word_char_overlaps]
    last_char_crops, char_idx = [], 0
    for i, n in enumerate(n_chars):
        for last in last_chars[i]:
            last_char_crops.append(char_crops[char_idx + last])
        char_idx += n
    logging.info('Number of word crops: {}'.format(len(word_crops)))
    logging.info('Number of last characters: {}'.format(len(last_char_crops)))
    
    # Create batches of last character crops
    embeddings = get_crop_embeddings(char_recognizer_engine, last_char_crops, num_streams=num_streams)
    
    # Get the nearest neighbor of each last character crop
    indices = [char_knn_func(embedding, k=1)[1] for embedding in embeddings]
    index_list = [index.squeeze(-1).tolist() for index in indices]
    indices = [item for sublist in index_list for item in sublist]
    nn_outputs_last_chars = [candidate_chars[idx] for idx in indices][:len(word_crops)]

    # If the nearest neighbor is a punctuation mark, we adjust the word bounding box and save the punctuation mark
    found_end_punctuation, cur_line = [], 0
    for i, nn_output in enumerate(nn_outputs_last_chars):
        if nn_output in END_PUNCTUATION:
            found_end_punctuation.append((i, nn_output))
            word_crops[i] = word_crops[i][:, :(-1 * last_char_crops[i].shape[1])]

    ''' Word level recognition '''
    # Get recognizer embeddings of word crops
    embeddings = get_crop_embeddings(recognizer_engine, word_crops, num_streams=num_streams)
    
    # Get the nearest neighbor of each word crop
    distances_and_indices = [word_knn_func(embedding, k=1) for embedding in embeddings]
    distances_index_lists = [(distance.squeeze(-1).tolist(), index.squeeze(-1).tolist()) for (distance, index) in distances_and_indices]
    distances_and_indices = [(distance, index) for (distances, indices) in distances_index_lists for distance, index in zip(distances, indices)]
    nn_outputs, rec_types = [], []
    
    # If the nearest neighbor is closer than the threshold, we recognize the word. Otherwise, we pass the word to char level recognition
    for (distance, idx) in distances_and_indices:
        if distance > recognizer_thresh:
            nn_outputs.append(candidate_words[idx])
            rec_types.append('word')
        else:
            nn_outputs.append("WORD_LEVEL")
            rec_types.append('char')

    # Add punctuation marks to the end of words recognized by the word recognizer
    for (i, punctuation) in found_end_punctuation:
        if nn_outputs[i] != 'WORD_LEVEL':
            nn_outputs[i] += punctuation

    ''' Char level recognition'''
    # Collect char crops from words that are not recognized
    char_crops_to_recognize, word_lens = [], []
    word_idx, char_idx = 0, 0
    for i, (n_c, n_w) in enumerate(zip(n_chars, n_words)):
        for j in range(word_idx, word_idx + n_w):
            if nn_outputs[j] == "WORD_LEVEL":
                for k in word_char_overlaps[i][j - word_idx]:
                    char_crops_to_recognize.append(char_crops[char_idx + k])
                
                word_lens.append(len(word_char_overlaps[i][j - word_idx]))

        word_idx += n_w
        char_idx += n_c 

    # Get char recognizer embeddings of char crops
    embeddings = get_crop_embeddings(char_recognizer_engine, char_crops_to_recognize, num_streams=num_streams)

    indices = [char_knn_func(embedding, k=1)[1] for embedding in embeddings]
    index_list = [index.squeeze(-1).tolist() for index in indices]
    indices = [item for sublist in index_list for item in sublist]
    nn_outputs_chars = [candidate_chars[idx] for idx in indices]
    word_idx, char_idx = 0, 0

    # Summing only up to the total number of words to avoid running into padded examples
    # at the end of the last batch
    for i in range(sum(n_words)):
        if nn_outputs[i] == "WORD_LEVEL":
            textline = nn_outputs_chars[char_idx:char_idx + word_lens[word_idx]]
            nn_outputs[i] = "".join(x[0] for x in textline).strip()
            char_idx += word_lens[word_idx]
            word_idx += 1

    #Now run postprocessing to create full textlines
    idx, textline_outputs, textline_rec_types = 0, [], []
    for l in n_words:
        textline_outputs.append(nn_outputs[idx:idx+l])
        textline_rec_types.append(rec_types[idx:idx+l])
        idx += l

    outputs = [" ".join(x for x in textline).strip() for textline in textline_outputs]
    
    # Postprocess textlines, saving to inference_assembly
    if lang == "en":
        for i, (im_idx, bbox_idx) in enumerate(coco_new_order):
            inference_assembly[bbox_idx][im_idx] = {}
            inference_assembly[bbox_idx][im_idx]['text'] = outputs[i]
            inference_assembly[bbox_idx][im_idx]['rec_types'] = textline_rec_types[i]
            if inference_assembly[bbox_idx][im_idx] is None:
                inference_assembly[bbox_idx][im_idx]['text'] = " "
                inference_assembly[bbox_idx][im_idx]['rec_types'] = []
            
    
    if localizer_output:
        for bbox_idx in inference_assembly.keys():
            for im_idx in inference_assembly[bbox_idx].keys():
                if im_idx == -1:
                    continue
                im = textline_images[im_idx][1]
                predicted_text = inference_assembly[bbox_idx][im_idx]['text'].split(' ')
                rec_types = [w[0] for w in inference_assembly[bbox_idx][im_idx]['rec_types']]

                # Pad the image with 50 pixels of white on the bottom
                im = np.pad(im, ((0, 20), (0, 0), (0, 0)), mode='constant', constant_values=255)
                # Also pad the image with 50 pixels of white on the right
                im = np.pad(im, ((0, 0), (0, 50), (0, 0)), mode='constant', constant_values=255)

                img = Image.fromarray((im * -255).astype(np.uint8))
                im_width, im_height = img.size[0], img.size[1]
                draw = ImageDraw.Draw(img)

                for i, word_bbox in enumerate(all_word_bboxes[im_idx]):
                    draw.rectangle([*word_bbox], outline='red')
                    try:
                        draw.text((word_bbox[0], im_height - 20), f'{rec_types[i]}: {predicted_text[i]}', fill='red')
                    except IndexError:
                        draw.text((word_bbox[0], im_height - 20), 'XXX', fill='red')
                img.save(os.path.join(localizer_output, str(bbox_idx), f'{im_idx}_viz.jpg'))
                logging.info(f'Saved visualization for bbox {bbox_idx} and image {im_idx} to {localizer_output}/{bbox_idx}/{im_idx}_viz.jpg')
    
    # Remove the -1 key from inference_assembly
    for bbox_idx in inference_assembly.keys():
        if -1 in inference_assembly[bbox_idx].keys():
            del inference_assembly[bbox_idx][-1]

    if insert_paragraph_breaks:
        inference_assembly = add_paragraph_breaks_to_dict(inference_assembly, side_dists)

    try:
        inference_results = {bbox_idx: '\n'.join([inference_assembly[bbox_idx][i]['text'] for i in sorted([int(x) for x in inference_assembly[bbox_idx].keys()])]) for 
                                                            bbox_idx in inference_assembly.keys()}
    except TypeError as e:
        print(e)
        print(inference_assembly)
        inference_results = {}

    return inference_results, inference_bboxes