import json
from cv2 import cv2
import os
from pprint import PrettyPrinter
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import sys
from collections import defaultdict, OrderedDict
from multiprocessing import Process, Manager
from functools import partial
import psutil
from contextlib import redirect_stdout
import copy
from PIL import Image
from timeit import default_timer as timer
from multiprocessing.managers import BaseManager, DictProxy
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

sys.path.insert(0, "../")
from utils.spell_check_utils import *


COCO_JSON_SKELETON = {
    "info": {"":""},
    "licenses": [{"":""}],
    "images": [],
    "annotations": [],
    "categories": []
}


def roundint(x):
    return int(round(x))


def create_coco_anno_entry(x, y, w, h, ann_id, image_id, cat_id, text=None, score=None):
    if text is None:
        return {
            "segmentation": [[roundint(x), roundint(y), roundint(x)+roundint(w), roundint(y),
                            roundint(x)+roundint(w), roundint(y)+roundint(h), roundint(x), roundint(y)+roundint(h)]],
            "area": roundint(w)*roundint(h), "iscrowd": 0,
            "image_id": image_id, "bbox": [roundint(x), roundint(y), roundint(w), roundint(h)],
            "category_id": cat_id, "id": ann_id, "score": score
        }
    else:
        return {
            "segmentation": [[roundint(x), roundint(y), roundint(x)+roundint(w), roundint(y),
                            roundint(x)+roundint(w), roundint(y)+roundint(h), roundint(x), roundint(y)+roundint(h)]],
            "area": roundint(w)*roundint(h), "iscrowd": 0,
            "image_id": image_id, "bbox": [roundint(x), roundint(y), roundint(w), roundint(h)],
            "category_id": cat_id, "id": ann_id, "score": score, "text": text
        }


def create_coco_image_entry(path, h, w, image_id, text=None):
    if text is None:
        return {
            "file_name": path,
            "height": h,
            "width": w,
            "id": image_id
        }
    else:
        return {
            "file_name": path,
            "height": h,
            "width": w,
            "id": image_id,
            "text": text
        }


class CustomManager(BaseManager):
    pass


class LayoutObject:

    def __init__(self, full_article_id, label, bbox, ocr_text, object_id, image_file_name, image_path, reading_order_id):

        self.image_file_name = image_file_name
        self.image_path = image_path
        self.object_id = object_id
        self.label = label
        self.bbox = bbox
        self.ocr_text = ocr_text
        self.full_article_id = full_article_id
        self.reading_order_id = reading_order_id


class Layouts:
    """Object class for handling layout info that results from D2 model inference."""


    def __init__(
            self,
            layout_predictions,
            layout_label_map,
            line_model,
            line_label_map,
            effocr_model,
            output_dir,
            spell_checks,
            homoglyphs_path
        ):

        # from prediction
        self.predictions = layout_predictions
        self.label_map = layout_label_map

        # other models
        self.line_model = line_model
        self.line_label_map = line_label_map
        self.effocr_model = effocr_model

        #spell checking
        self.spell_checks = spell_checks
        self.homoglyphs_path = homoglyphs_path

        # save outputs
        self.layout_dir = os.path.join(output_dir, "layout")
        self.line_dir = os.path.join(output_dir, "line")
        os.makedirs(self.layout_dir, exist_ok=True)
        os.makedirs(self.line_dir, exist_ok=True)

        # optionally created
        self.cat_coord_df = None
        self.ocr_text_dict = None


    def create_ocr_text_dict(self, saving=True, tesseract=False, trocr=False):

        start_time = timer()
        output_dict = defaultdict(list)

        if trocr:
            processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
            model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

        image_id, ann_id = 0, 0


        layout_coco = copy.deepcopy(COCO_JSON_SKELETON)
        line_coco = copy.deepcopy(COCO_JSON_SKELETON)
        char_coco = copy.deepcopy(COCO_JSON_SKELETON)

        layout_image_id, layout_ann_id = 0, 0
        line_image_id, line_ann_id = 0, 0

        layout_coco["categories"] = [{'id': k, 'name': v} for k, v in self.label_map.items()] # imgs are scans, annos are layout ele
        line_coco["categories"] = [{'id': 0, 'name': "line"}]                                 # imgs are layout eles, annos are lines
        char_coco["categories"] = [{'id': 0, 'name': "char"}, {'id': 1, 'name': "word"}]      # imgs are lines, annos are chars and words

        # iterating through images
        for prediction in tqdm(self.predictions):

            image_path = prediction["image_path"]
            instance_pred = prediction["instances"].to("cpu")
            boxes = instance_pred.pred_boxes.tensor.tolist()
            labels = instance_pred.pred_classes.tolist()
            scores = instance_pred.scores.tolist()
            image_height, image_width = instance_pred.image_size

            layout_coco["images"].append(create_coco_image_entry(os.path.basename(image_path), image_height, image_width, image_id))

            image = np.array(Image.open(image_path).convert("RGB"))

            layout_element_crops = []

            # getting layout crops
            layout_bboxes = []
            layout_image_ids_covered = []


            for box, label in zip(boxes, labels):

                x1, y1, x2, y2 = map(int, map(round, box))
                layout_bboxes.append((x1, y1, x2, y2))

                layout_element_crop = image[y1:y2,x1:x2,:]
                layout_height, layout_width, _ = layout_element_crop.shape

                layout_file_name = f"{layout_image_id}.png"
                line_coco["images"].append(create_coco_image_entry(layout_file_name, layout_height, layout_width, layout_image_id))
                layout_image_ids_covered.append(layout_image_id) #Having this in the saving block is making the assertion on line 224 fail if saving isn't passed as an argument. Expected?
                layout_image_id += 1

                if saving:
                    Image.fromarray(layout_element_crop).save(os.path.join(self.layout_dir, layout_file_name))


                layout_element_crops.append(layout_element_crop)

            # line prediction for all layout crops
            within_layout_predictions = self.line_model.detect(layout_element_crops)

            # getting line crops
            line_crops_by_layout = []
            line_image_ids_covered_by_layout = []
            for layout_element_crop, within_layout_prediction, _layout_image_id in \
                zip(layout_element_crops, within_layout_predictions, layout_image_ids_covered):

                line_boxes = within_layout_prediction["instances"].to("cpu").pred_boxes.tensor.tolist()
                line_crops = []
                line_image_ids_covered = []

                for line_bbox in sorted(line_boxes, key=lambda x: x[1]):

                    #Create the line crop from predicted bbox
                    lx1, ly1, lx2, ly2 = map(int, map(round, line_bbox))
                    line_crop = layout_element_crop[ly1:ly2,:,:]

                    #Add line crop to various lists
                    line_coco["annotations"].append(create_coco_anno_entry(lx1, ly1, lx2-lx1, ly2-ly1,
                        layout_ann_id, _layout_image_id, 0))
                    layout_ann_id += 1
                    line_height, line_width, _ = line_crop.shape
                    line_file_name = f"{line_image_id}.png"
                    char_coco["images"].append(create_coco_image_entry(line_file_name, line_height, line_width, line_image_id, text=os.path.basename(image_path)))
                    line_image_ids_covered.append(line_image_id)
                    line_image_id += 1
                    line_crops.append(line_crop)


                    if saving:
                        Image.fromarray(line_crop).save(os.path.join(self.line_dir, line_file_name))


                line_image_ids_covered_by_layout.append(line_image_ids_covered)
                line_crops_by_layout.append(line_crops)


            # ocr prediction for all line crops
            assert len(line_crops_by_layout) == len(labels) == len(layout_bboxes) == len(scores), \
                        'Line Crops info problem: n_layout_line_crops: {}, n_layout_labels: {}, n_layout_bboxes: {}, n_layout_conf_scores: {} (should all match exactly)'.format(
                            len(line_crops_by_layout), len(labels), len(layout_bboxes), len(scores))

            for line_crops, label, layout_bbox, _line_image_ids, score in \
                tqdm(zip(line_crops_by_layout, labels, layout_bboxes, line_image_ids_covered_by_layout, scores)):

                class_label = self.label_map.get(label, label)

                if class_label in ("headline", "article", "image_caption") and len(line_crops) != 0:

                    if tesseract:
                        line_ocr = [pytesseract.image_to_string(Image.fromarray(line_crop).convert("RGB"), config="-l eng --psm 7 --oem 1") for line_crop in line_crops]
                        ocr = "\n".join(line_ocr)
                    elif trocr:
                        line_ocr = []
                        for line_crop in line_crops:
                            pixel_values = processor(Image.fromarray(line_crop).convert("RGB"), return_tensors="pt").pixel_values
                            generated_ids = model.generate(pixel_values)
                            line_ocr.append(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
                        ocr = "\n".join(line_ocr)
                    else:
                        line_ocr, char_bboxes_by_line, word_bboxes_by_line = [], [], []
                        for line_crop in line_crops:
                            output, output_nns, char_bboxes, word_bboxes = self.effocr_model.infer(line_crop)
                            line_ocr.append(output)
                            char_bboxes_by_line.append(char_bboxes)
                            word_bboxes_by_line.append(word_bboxes)
                        ocr = "\n".join([x if x is not None else "" for x in line_ocr])

                    # assert len(char_bboxes_by_line) == len(word_bboxes_by_line) == len(_line_image_ids)
                    for charbboxes, wordbboxes, _line_image_id in zip(char_bboxes_by_line, word_bboxes_by_line, _line_image_ids):
                        if charbboxes:
                            for charbbox in charbboxes:
                                cx1, cy1, cx2, cy2 = map(int, map(round, charbbox.tolist()[:4]))
                                char_coco["annotations"].append(create_coco_anno_entry(cx1, cy1, cx2-cx1, cy2-cy1, line_ann_id, _line_image_id, 0))
                                line_ann_id += 1
                            for wordbbox in wordbboxes:
                                wx1, wy1, wx2, wy2 = map(int, map(round, wordbbox.tolist()[:4]))
                                char_coco["annotations"].append(create_coco_anno_entry(wx1, wy1, wx2-wx1, wy2-wy1, line_ann_id, _line_image_id, 1))
                                line_ann_id += 1

                else:
                    ocr = None

                x1, y1, x2, y2 = layout_bbox



                output_dict[os.path.basename(image_path)].append(LayoutObject(
                    image_path=image_path, image_file_name=os.path.basename(image_path),
                    object_id=ann_id, label=self.label_map.get(label, label), bbox=box, ocr_text=ocr,
                    full_article_id=None, reading_order_id=None))

                layout_coco["annotations"].append(create_coco_anno_entry(x1, y1, x2-x1, y2-y1, ann_id, image_id, label, text=ocr, score=score))

                ann_id += 1

            # new id
            image_id += 1


        if self.spell_checks:
            output_dict = self.spell_check_output(output_dict)

        self.ocr_text_dict = output_dict

        return layout_coco, line_coco, char_coco

    def spell_check_output(self, output_dict):
        wordset = None
        abbrevset = None
        homoglyph_dict = None

        for sc in self.spell_checks:
            if sc == 'symspell':
                pass

            elif sc == 'effocr_visual':
                if wordset is None:
                    wordset = create_worddict()
                if abbrevset is None:
                    abbrevset = create_common_abbrev()
                if homoglyph_dict is None:
                    homoglyph_dict = create_homoglyph_dict(homoglyph_fp=homoglyphs_path)

                for scan in output_dict:
                    for bbox in scan:
                        bbox['ocr_text'] = visual_spell_checker(bbox['ocr_text'], wordset, homoglyph_dict, abbrevset)

        return output_dict

    def load_ocr_text_dict(self, ocr_json_path):

        input_dict = defaultdict(list)                                                                                  # Create an empty list for value of any dictionary item you try and access that doesn't exist yet

        with open(ocr_json_path) as jf:
            ocr_text_dict = json.load(jf)
            for imgf in ocr_text_dict.keys():                                                                           # imgf = k
                layout_objs = ocr_text_dict[imgf]                                                                       # layout_oj=bjs = v (list of dictionaries one for each bbox)
                for layout_obj in layout_objs:                                                                          # for each bbox
                    input_dict[imgf].append(                                                                            # Create a key in input_dict, which is the file name
                        LayoutObject(                                                                                   # Value is list of LayoutObjects, each describing a bbox
                            image_path=layout_obj['image_path'],
                            image_file_name=layout_obj['image_file_name'],
                            object_id=layout_obj['object_id'],
                            label=layout_obj['label'],
                            bbox=layout_obj['bbox'],
                            ocr_text=layout_obj['ocr_text'],
                            full_article_id=None,
                            reading_order_id=None
                        )
                    )

        self.ocr_text_dict = input_dict                                                                                 # Store as ocr_text_dict


    def create_ocr_text_dict_parallel(self, tessdata_dir, proc_num=(psutil.cpu_count(logical = False) // 2)):

        print(f'Tesseract version: {tesserocr.tesseract_version()}')

        print(f'Total number of physical cores: {psutil.cpu_count(logical = False)}')

        CustomManager.register('defaultdict', defaultdict, DictProxy)

        with CustomManager as manager:

            output_dict = manager.defaultdict(list)

            partial_f = partial(self._create_ocr_text_dict_parallel,
                output_dict=output_dict,
                tessdata_dir=tessdata_dir)

            with manager.Pool(processes=proc_num) as pool:
                list(tqdm(pool.imap(partial_f, self.predictions)))

            output_dict = dict(output_dict)
            self.ocr_text_dict = output_dict


    def _create_ocr_text_dict_parallel(self, prediction, output_dict, tessdata_dir):

        with tesserocr.PyTessBaseAPI(path=tessdata_dir,
            psm=tesserocr.PSM.SINGLE_COLUMN,
            oem=tesserocr.OEM.LSTM_ONLY) as api:

            image_path = prediction["image_path"]
            instance_pred = prediction["instances"].to("cpu")
            boxes = instance_pred.pred_boxes.tensor.tolist()
            labels = instance_pred.pred_classes.tolist()
            id_from_counter = 0

            api.SetImageFile(image_path)

            for box, label in zip(boxes, labels):

                # ocr
                x1, y1, x2, y2 = box
                left, top, width, height = x1, y1, (x2 - x1), (y2 - y1)
                api.SetRectangle(left, top, width, height)
                ocr_text = api.GetUTF8Text()

                # create output dict entry
                output_dict[os.path.basename(image_path)].append(
                    LayoutObject(
                        image_path=image_path,
                        image_file_name=os.path.basename(image_path),
                        object_id=id_from_counter,
                        label=self.label_map.get(label, label),
                        bbox=box,
                        ocr_text=ocr_text,
                        full_article_id=None,
                        reading_order_id=None
                    )
                )

                # new id
                id_from_counter += 1


    def create_article_headline_layout_info_df(self):

        object_ids = []
        image_file_names = []
        labels = []
        top_left_xs = []
        top_left_ys = []
        bottom_right_xs = []
        bottom_right_ys = []

        for _, layout_objects in self.ocr_text_dict.items():

            for layout_object in layout_objects:

                if layout_object.label not in ('article', 'headline'):
                    continue

                x1, y1, x2, y2 = layout_object.bbox

                object_ids.append(layout_object.object_id)
                image_file_names.append(layout_object.image_file_name)
                labels.append(layout_object.label)
                top_left_xs.append(x1)
                top_left_ys.append(y1)
                bottom_right_xs.append(x2)
                bottom_right_ys.append(y2)

        df = pd.DataFrame({
            'image_file_name': image_file_names,
            'object_id': object_ids,
            'label': labels,
            'top_left_x': top_left_xs,
            'top_left_y': top_left_ys,
            'bottom_right_x': bottom_right_xs,
            'bottom_right_y': bottom_right_ys
        })

        self.article_headline_layout_info_df = df.\
            sort_values(by=['bottom_right_x', 'bottom_right_y'], ascending=True)


    def create_article_headline_layout_info_df_bl(self):

        object_ids = []
        image_file_names = []
        labels = []
        top_left_xs = []
        top_left_ys = []
        bottom_right_xs = []
        bottom_right_ys = []

        for _, layout_objects in self.ocr_text_dict.items():

            for layout_object in layout_objects:

                if layout_object.label not in ('article', 'headline', 'author'):
                    continue

                x1, y1, x2, y2 = layout_object.bbox

                object_ids.append(layout_object.object_id)
                image_file_names.append(layout_object.image_file_name)
                labels.append(layout_object.label)
                top_left_xs.append(x1)
                top_left_ys.append(y1)
                bottom_right_xs.append(x2)
                bottom_right_ys.append(y2)

        df = pd.DataFrame({
            'image_file_name': image_file_names,
            'object_id': object_ids,
            'label': labels,
            'top_left_x': top_left_xs,
            'top_left_y': top_left_ys,
            'bottom_right_x': bottom_right_xs,
            'bottom_right_y': bottom_right_ys
        })

        self.article_headline_layout_info_df = df. \
            sort_values(by=['bottom_right_x', 'bottom_right_y'], ascending=True)


    def create_fa_ids(self):

        # check if object text dict exists; if so, create cat coord dataframe
        if self.ocr_text_dict is None:
            return None
        else:
            self.create_article_headline_layout_info_df()

        # ALGO: associate each headline with article via headline midline intersection
        unique_image_file_names = list(set(self.article_headline_layout_info_df['image_file_name']))

        # perform associations within image
        for image_file_name in unique_image_file_names:

            # subset dfs
            image_df = self.article_headline_layout_info_df.loc[self.article_headline_layout_info_df['image_file_name'] == image_file_name]
            image_article_df = image_df.loc[image_df['label'] == 'article']
            image_headline_df = image_df.loc[image_df['label'] == 'headline']

            # make numpy arrays
            hl_coord_mat = image_headline_df[['top_left_x', 'bottom_right_x', 'bottom_right_y']].to_numpy()
            art_coord_mat = image_article_df[['top_left_x', 'bottom_right_x', 'bottom_right_y']].to_numpy()

            # get object id lists for reference/lookup
            hl_object_ids = image_headline_df['object_id'].tolist()
            art_object_ids = image_article_df['object_id'].tolist()

            # prepare for association
            fullArticle_ids_dict = defaultdict(int)
            fa_id_from_counter = 0

            # keep track of headline y coords
            headline_bottom_seen = defaultdict(int)
            headline_bottom_seen_obj_id = defaultdict(int)

            # associate every headline with an article
            for i in range(hl_coord_mat.shape[0]):

                h_mat = hl_coord_mat[i,:]
                headline_line_seg_l, headline_line_seg_r = h_mat[0], h_mat[1]
                headline_bottom = h_mat[2]
                object_id_of_given_headline = hl_object_ids[i]

                for j in range(art_coord_mat.shape[0]):

                    a_mat = art_coord_mat[j,:]
                    art_top_line_seg_l, art_top_line_seg_r = a_mat[0], a_mat[1]
                    art_bottom = a_mat[2]

                    if headline_bottom > art_bottom:
                        continue

                    x_pad = (a_mat[1] - a_mat[0]) / 10

                    if (art_top_line_seg_l < headline_line_seg_l + x_pad < art_top_line_seg_r) or \
                        (art_top_line_seg_l < headline_line_seg_r - x_pad < art_top_line_seg_r) or \
                        (headline_line_seg_l < art_top_line_seg_l + x_pad < headline_line_seg_r) or \
                        (headline_line_seg_l < art_top_line_seg_r - x_pad < headline_line_seg_r):

                        # create objectID-fullarticleID dict entry for headline
                        fullArticle_ids_dict[object_id_of_given_headline] = [fa_id_from_counter]

                        # create objectID-fullarticleID dict entry for article
                        object_id_of_assoc_article = art_object_ids[j]

                        if object_id_of_assoc_article in fullArticle_ids_dict:
                            fullArticle_ids_dict[object_id_of_assoc_article].append(fa_id_from_counter)
                        else:
                            fullArticle_ids_dict[object_id_of_assoc_article] = [fa_id_from_counter]

                # increment id; keep track of headline y coords
                headline_bottom_seen[fa_id_from_counter] = headline_bottom
                headline_bottom_seen_obj_id[fa_id_from_counter] = object_id_of_given_headline
                fa_id_from_counter += 1

            # prune: only the lowest headline assoc w article is kept
            unpruned_fa_ids_dict = fullArticle_ids_dict.copy()
            for obj_id in fullArticle_ids_dict.keys():

                layout_object_for_obj_id = [layobj for layobj in self.ocr_text_dict[image_file_name] \
                    if layobj.object_id == obj_id][0]

                if layout_object_for_obj_id.label == 'article':

                    fa_ids = fullArticle_ids_dict[obj_id]
                    headline_bottoms_for_fa_ids = np.array([headline_bottom_seen[fa_id] for fa_id in fa_ids])

                    lowest_hl_idx = headline_bottoms_for_fa_ids.argmax()
                    lowest_hl_fa_id = fa_ids[lowest_hl_idx]

                    fullArticle_ids_dict[obj_id] = [lowest_hl_fa_id]

            # retain second lowest headline assoc if second lowest headline is a singleton
            for obj_id in fullArticle_ids_dict.keys():

                layout_object_for_obj_id = [layobj for layobj in self.ocr_text_dict[image_file_name] \
                    if layobj.object_id == obj_id][0]

                if layout_object_for_obj_id.label == 'article':

                    fa_ids = unpruned_fa_ids_dict[obj_id]
                    headline_bottoms_for_fa_ids = np.array([headline_bottom_seen[fa_id] for fa_id in fa_ids])

                    if headline_bottoms_for_fa_ids.size > 1:
                        lowest_hl_idx = headline_bottoms_for_fa_ids.argmax()
                        lowest_hl_fa_id = fa_ids[lowest_hl_idx]
                        second_lowest_hl_idx = headline_bottoms_for_fa_ids.argsort()[-2]
                        second_lowest_hl_fa_id = fa_ids[second_lowest_hl_idx]
                        count_second_lowest_hl_fa_id = sum(1 for sublist in fullArticle_ids_dict.values() \
                            for i in sublist if i == second_lowest_hl_fa_id)
                        is_second_lowest_hl_fa_id_singleton = count_second_lowest_hl_fa_id == 1
                        if is_second_lowest_hl_fa_id_singleton:
                            fullArticle_ids_dict[headline_bottom_seen_obj_id[second_lowest_hl_fa_id]] = [lowest_hl_fa_id]

            # add to new object dict with fullArticle id
            for layout_object in self.ocr_text_dict[image_file_name]:

                retr_fa_id = fullArticle_ids_dict.get(layout_object.object_id, None)
                layout_object.full_article_id = retr_fa_id[0] if isinstance(retr_fa_id, list) else retr_fa_id


    def create_fa_ids_bl(self):

        # check if object text dict exists; if so, create cat coord dataframe
        if self.ocr_text_dict is None:
            return None
        else:
            self.create_article_headline_layout_info_df_bl()

        # ALGO: associate each headline with article via headline midline intersection
        unique_image_file_names = list(set(self.article_headline_layout_info_df['image_file_name']))

        # perform associations within image
        for image_file_name in unique_image_file_names:

            # subset dfs
            image_df = self.article_headline_layout_info_df.loc[self.article_headline_layout_info_df['image_file_name'] == image_file_name]
            image_article_df = image_df.loc[image_df['label'] == 'article']
            image_headline_df = image_df.loc[image_df['label'] == 'headline']
            image_author_df = image_df.loc[image_df['label'] == 'author']

            # make numpy arrays
            hl_coord_mat = image_headline_df[['top_left_x', 'bottom_right_x', 'bottom_right_y', 'top_left_y']].to_numpy()
            art_coord_mat = image_article_df[['top_left_x', 'bottom_right_x', 'bottom_right_y']].to_numpy()
            auth_coord_mat = image_author_df[['top_left_x', 'bottom_right_x', 'bottom_right_y', 'top_left_y']].to_numpy()

            # get object id lists for reference/lookup
            hl_object_ids = image_headline_df['object_id'].tolist()
            art_object_ids = image_article_df['object_id'].tolist()
            auth_object_ids = image_author_df['object_id'].tolist()

            # prepare for association
            fullArticle_ids_dict = defaultdict(int)
            fa_id_from_counter = 0

            # keep track of headline y coords
            headline_bottom_seen = defaultdict(int)
            headline_bottom_seen_obj_id = defaultdict(int)

            # associate every headline with an article
            for i in range(hl_coord_mat.shape[0]):

                h_mat = hl_coord_mat[i,:]
                headline_line_seg_l, headline_line_seg_r = h_mat[0], h_mat[1]
                headline_top, headline_bottom = h_mat[3], h_mat[2]
                object_id_of_given_headline = hl_object_ids[i]

                for j in range(art_coord_mat.shape[0]):

                    a_mat = art_coord_mat[j,:]
                    art_top_line_seg_l, art_top_line_seg_r = a_mat[0], a_mat[1]
                    art_bottom = a_mat[2]

                    if headline_bottom > art_bottom:
                        continue

                    x_pad = (a_mat[1] - a_mat[0]) / 10

                    if (art_top_line_seg_l < headline_line_seg_l + x_pad < art_top_line_seg_r) or \
                            (art_top_line_seg_l < headline_line_seg_r - x_pad < art_top_line_seg_r) or \
                            (headline_line_seg_l < art_top_line_seg_l + x_pad < headline_line_seg_r) or \
                            (headline_line_seg_l < art_top_line_seg_r - x_pad < headline_line_seg_r):

                        # create objectID-fullarticleID dict entry for headline
                        fullArticle_ids_dict[object_id_of_given_headline] = [fa_id_from_counter]

                        # create objectID-fullarticleID dict entry for article
                        object_id_of_assoc_article = art_object_ids[j]

                        if object_id_of_assoc_article in fullArticle_ids_dict:
                            fullArticle_ids_dict[object_id_of_assoc_article].append(fa_id_from_counter)
                        else:
                            fullArticle_ids_dict[object_id_of_assoc_article] = [fa_id_from_counter]

                ####### Headline-author association
                for k in range(auth_coord_mat.shape[0]):

                    auth_mat = auth_coord_mat[k,:]
                    auth_top_line_seg_l, auth_top_line_seg_r = auth_mat[0], auth_mat[1]
                    auth_top, auth_bottom = auth_mat[3], auth_mat[2]

                    x_pad = (auth_mat[1] - auth_mat[0]) / 10
                    y_pad = (auth_mat[2] - auth_mat[3])

                    if ((auth_top_line_seg_l < headline_line_seg_l + x_pad < auth_top_line_seg_r) or \
                            (auth_top_line_seg_l < headline_line_seg_r - x_pad < auth_top_line_seg_r) or \
                            (headline_line_seg_l < auth_top_line_seg_l + x_pad < headline_line_seg_r) or \
                            (headline_line_seg_l < auth_top_line_seg_r - x_pad < headline_line_seg_r)) and \
                        ((auth_top < headline_top < auth_bottom) or \
                            (auth_top < headline_bottom < auth_bottom) or \
                            (auth_top < headline_top - y_pad < auth_bottom) or \
                            (auth_top < headline_bottom + y_pad < auth_bottom)):

                        # create objectID-fullarticleID dict entry for article
                        object_id_of_assoc_article = auth_object_ids[k]

                        if object_id_of_assoc_article in fullArticle_ids_dict:
                            fullArticle_ids_dict[object_id_of_assoc_article].append(fa_id_from_counter)
                        else:
                            fullArticle_ids_dict[object_id_of_assoc_article] = [fa_id_from_counter]
                #######

                # increment id; keep track of headline y coords
                headline_bottom_seen[fa_id_from_counter] = headline_bottom
                headline_bottom_seen_obj_id[fa_id_from_counter] = object_id_of_given_headline
                fa_id_from_counter += 1

            # prune: only the lowest headline assoc w article is kept
            unpruned_fa_ids_dict = fullArticle_ids_dict.copy()
            for obj_id in fullArticle_ids_dict.keys():

                layout_object_for_obj_id = [layobj for layobj in self.ocr_text_dict[image_file_name] \
                                            if layobj.object_id == obj_id][0]

                if layout_object_for_obj_id.label == 'article':

                    fa_ids = fullArticle_ids_dict[obj_id]
                    headline_bottoms_for_fa_ids = np.array([headline_bottom_seen[fa_id] for fa_id in fa_ids])

                    lowest_hl_idx = headline_bottoms_for_fa_ids.argmax()
                    lowest_hl_fa_id = fa_ids[lowest_hl_idx]

                    fullArticle_ids_dict[obj_id] = [lowest_hl_fa_id]

            # retain second lowest headline assoc if second lowest headline is a singleton
            for obj_id in fullArticle_ids_dict.keys():

                layout_object_for_obj_id = [layobj for layobj in self.ocr_text_dict[image_file_name] \
                                            if layobj.object_id == obj_id][0]

                if layout_object_for_obj_id.label == 'article':

                    fa_ids = unpruned_fa_ids_dict[obj_id]
                    headline_bottoms_for_fa_ids = np.array([headline_bottom_seen[fa_id] for fa_id in fa_ids])

                    if headline_bottoms_for_fa_ids.size > 1:
                        lowest_hl_idx = headline_bottoms_for_fa_ids.argmax()
                        lowest_hl_fa_id = fa_ids[lowest_hl_idx]
                        second_lowest_hl_idx = headline_bottoms_for_fa_ids.argsort()[-2]
                        second_lowest_hl_fa_id = fa_ids[second_lowest_hl_idx]
                        count_second_lowest_hl_fa_id = sum(1 for sublist in fullArticle_ids_dict.values() \
                                                           for i in sublist if i == second_lowest_hl_fa_id)
                        is_second_lowest_hl_fa_id_singleton = count_second_lowest_hl_fa_id == 1
                        if is_second_lowest_hl_fa_id_singleton:
                            fullArticle_ids_dict[headline_bottom_seen_obj_id[second_lowest_hl_fa_id]] = [lowest_hl_fa_id]

            # select fa_id if author is associated with > 1
            for obj_id in fullArticle_ids_dict.keys():

                layout_object_for_obj_id = [layobj for layobj in self.ocr_text_dict[image_file_name] \
                                            if layobj.object_id == obj_id][0]

                if layout_object_for_obj_id.label == 'author':

                    fa_ids = fullArticle_ids_dict[obj_id]
                    first_fa_id = fa_ids[0]

                    fullArticle_ids_dict[obj_id] = [first_fa_id]


            # add to new object dict with fullArticle id
            for layout_object in self.ocr_text_dict[image_file_name]:

                retr_fa_id = fullArticle_ids_dict.get(layout_object.object_id, None)
                layout_object.full_article_id = retr_fa_id[0] if isinstance(retr_fa_id, list) else retr_fa_id


    def create_ro_ids(self, vertical_margin=0.10):

        # iterate through images
        unique_image_file_names = list(set(self.article_headline_layout_info_df['image_file_name']))
        for image_file_name in unique_image_file_names:

            # get unique full article IDs in image; iterate through them
            unique_full_article_ids = list(set([layobj.full_article_id for layobj in self.ocr_text_dict[image_file_name]]))
            for unique_fa_id in unique_full_article_ids:

                # vertically sort layout objects for given full article ID
                layobjs_for_unique_fa_id = [layobj for layobj in self.ocr_text_dict[image_file_name] if layobj.full_article_id == unique_fa_id]
                vertical_sorted_layobjs_for_unique_fa_id = sorted(layobjs_for_unique_fa_id, key=lambda x: x.bbox[1])
                ro_sorted_layobjs_for_unique_fa_id = []

                # iterate through sorted layout objects, sorting similarly vertically positioned objects by horizontal position
                for vslayobj in vertical_sorted_layobjs_for_unique_fa_id:

                    if vslayobj.object_id in [o.object_id for o in ro_sorted_layobjs_for_unique_fa_id]:
                        continue

                    height_i = vslayobj.bbox[3] - vslayobj.bbox[1]
                    similar_height_layobjs = [vslayobj]

                    for _vslayobj in vertical_sorted_layobjs_for_unique_fa_id:

                        if vslayobj.object_id == _vslayobj.object_id:
                            continue

                        if _vslayobj.object_id in [o.object_id for o in ro_sorted_layobjs_for_unique_fa_id]:
                            continue

                        height_j = _vslayobj.bbox[3] - _vslayobj.bbox[1]
                        vertical_gap = vertical_margin*min(height_j, height_i)

                        if abs(vslayobj.bbox[1] - _vslayobj.bbox[1]) < vertical_gap:
                            similar_height_layobjs.append(_vslayobj)

                    hori_sorted_similar_height_layobjs = sorted(similar_height_layobjs, key=lambda x: x.bbox[0])

                    for layobj in hori_sorted_similar_height_layobjs:
                        ro_sorted_layobjs_for_unique_fa_id.append(layobj)

                # actually create reading order ids based on list order
                for idx, layobj in enumerate(ro_sorted_layobjs_for_unique_fa_id):
                    layobj.reading_order_id = idx


    def create_ro_ids_alt(self, horizontal_margin=0.40):

        # iterate through images
        unique_image_file_names = list(set(self.article_headline_layout_info_df['image_file_name']))
        for image_file_name in unique_image_file_names:

            # get unique full article IDs in image; iterate through them
            unique_full_article_ids = list(set([layobj.full_article_id for layobj in self.ocr_text_dict[image_file_name]]))
            for unique_fa_id in unique_full_article_ids:

                # horizontally sort layout objects for given full article ID
                layobjs_for_unique_fa_id = [layobj for layobj in self.ocr_text_dict[image_file_name] \
                    if layobj.full_article_id == unique_fa_id and layobj.label == 'article']
                horizontal_sorted_layobjs_for_unique_fa_id = sorted(layobjs_for_unique_fa_id, key=lambda x: x.bbox[0])
                ro_sorted_layobjs_for_unique_fa_id = []

                # iterate through sorted layout objects, sorting similarly vertically positioned objects by horizontal position
                for hslayobj in horizontal_sorted_layobjs_for_unique_fa_id:

                    if hslayobj.object_id in [o.object_id for o in ro_sorted_layobjs_for_unique_fa_id]:
                        continue

                    width_i = hslayobj.bbox[2] - hslayobj.bbox[0]
                    similar_xpos_layobjs = [hslayobj]

                    for _hslayobj in horizontal_sorted_layobjs_for_unique_fa_id:

                        if hslayobj.object_id == _hslayobj.object_id:
                            continue

                        if _hslayobj.object_id in [o.object_id for o in ro_sorted_layobjs_for_unique_fa_id]:
                            continue

                        width_j = _hslayobj.bbox[2] - _hslayobj.bbox[0]
                        horizontal_gap = horizontal_margin*min(width_j, width_i)

                        if abs(hslayobj.bbox[0] - _hslayobj.bbox[0]) < horizontal_gap:
                            similar_xpos_layobjs.append(_hslayobj)

                    vert_sorted_similar_height_layobjs = sorted(similar_xpos_layobjs, key=lambda x: x.bbox[1])

                    for layobj in vert_sorted_similar_height_layobjs:
                        ro_sorted_layobjs_for_unique_fa_id.append(layobj)

                # actually create reading order ids based on list order
                for idx, layobj in enumerate(ro_sorted_layobjs_for_unique_fa_id):
                    layobj.reading_order_id = idx

                # get headlines
                headlines_for_unique_fa_id = [layobj for layobj in self.ocr_text_dict[image_file_name] \
                    if layobj.full_article_id == unique_fa_id and layobj.label == 'headline']

                vert_sorted_headlines_for_unique_fa_id = sorted(headlines_for_unique_fa_id, key=lambda x: x.bbox[1])
                n_hl = len(vert_sorted_headlines_for_unique_fa_id)

                for idx, layobj in enumerate(vert_sorted_headlines_for_unique_fa_id):
                    layobj.reading_order_id = idx - n_hl


    def create_ro_ids_alt_bl(self, horizontal_margin=0.40):

        # iterate through images
        unique_image_file_names = list(set(self.article_headline_layout_info_df['image_file_name']))
        for image_file_name in unique_image_file_names:

            # get unique full article IDs in image; iterate through them
            unique_full_article_ids = list(set([layobj.full_article_id for layobj in self.ocr_text_dict[image_file_name]]))
            for unique_fa_id in unique_full_article_ids:

                # Articles: horizontally sort layout objects for given full article ID
                layobjs_for_unique_fa_id = [layobj for layobj in self.ocr_text_dict[image_file_name] \
                                            if layobj.full_article_id == unique_fa_id and layobj.label == 'article']
                horizontal_sorted_layobjs_for_unique_fa_id = sorted(layobjs_for_unique_fa_id, key=lambda x: x.bbox[0])
                ro_sorted_layobjs_for_unique_fa_id = []

                # iterate through sorted layout objects, sorting similarly vertically positioned objects by horizontal position
                for hslayobj in horizontal_sorted_layobjs_for_unique_fa_id:

                    if hslayobj.object_id in [o.object_id for o in ro_sorted_layobjs_for_unique_fa_id]:
                        continue

                    width_i = hslayobj.bbox[2] - hslayobj.bbox[0]
                    similar_xpos_layobjs = [hslayobj]

                    for _hslayobj in horizontal_sorted_layobjs_for_unique_fa_id:

                        if hslayobj.object_id == _hslayobj.object_id:
                            continue

                        if _hslayobj.object_id in [o.object_id for o in ro_sorted_layobjs_for_unique_fa_id]:
                            continue

                        width_j = _hslayobj.bbox[2] - _hslayobj.bbox[0]
                        horizontal_gap = horizontal_margin*min(width_j, width_i)

                        if abs(hslayobj.bbox[0] - _hslayobj.bbox[0]) < horizontal_gap:
                            similar_xpos_layobjs.append(_hslayobj)

                    vert_sorted_similar_height_layobjs = sorted(similar_xpos_layobjs, key=lambda x: x.bbox[1])

                    for layobj in vert_sorted_similar_height_layobjs:
                        ro_sorted_layobjs_for_unique_fa_id.append(layobj)

                # actually create reading order ids based on list order
                for idx, layobj in enumerate(ro_sorted_layobjs_for_unique_fa_id):
                    layobj.reading_order_id = idx

                # Headlines and author (bylines): sort vertically
                headlines_for_unique_fa_id = [layobj for layobj in self.ocr_text_dict[image_file_name] \
                                              if layobj.full_article_id == unique_fa_id and \
                                              (layobj.label == 'headline' or layobj.label == 'author')]

                vert_sorted_headlines_for_unique_fa_id = sorted(headlines_for_unique_fa_id, key=lambda x: x.bbox[1])
                n_hl = len(vert_sorted_headlines_for_unique_fa_id)

                for idx, layobj in enumerate(vert_sorted_headlines_for_unique_fa_id):
                    layobj.reading_order_id = idx - n_hl


    def create_coco_like_dict(self):

        categories = [{'id': k, 'name': v} for k, v in self.label_map.items()]
        annotations = []
        images = []
        anno_id_from_counter = 0
        image_id_from_counter = 0

        for prediction in self.predictions:

            image_path = prediction["image_path"]
            instance_pred = prediction["instances"].to("cpu")
            scores = instance_pred.scores.tolist()
            boxes = instance_pred.pred_boxes.tensor.tolist()
            labels = instance_pred.pred_classes.tolist()
            areas = instance_pred.pred_boxes.area().tolist()
            images.append({'id': image_id_from_counter, 'file_name': image_path})

            for box, label, score, area in zip(boxes, labels, scores, areas):
                x_1, y_1, x_2, y_2 = box
                annotation = {
                    'image_id': image_id_from_counter,
                    'id': anno_id_from_counter,
                    'category_id': label,
                    'area': area,
                    'bbox': [x_1, y_1, x_2, y_2],
                    'score': score
                }
                anno_id_from_counter += 1
                annotations.append(annotation)

            image_id_from_counter += 1

        self.coco_like_dict = {}
        self.coco_like_dict['annotations'] = annotations
        self.coco_like_dict['categories'] = categories
        self.coco_like_dict['images'] = images


    def create_lp_layouts_from_ocr_text_dict(self):

        if self.ocr_text_dict is None:
            raise Exception("OCR text dict has not been created!")

        self.lp_layouts = []

        for _, layout_objects in self.ocr_text_dict.items():

            lp_layout = Layout()

            for layout_object in layout_objects:

                x1, y1, x2, y2 = layout_object.bbox
                block = TextBlock(
                    text=layout_object.ocr_text,
                    block=Rectangle(x1, y1, x2, y2),
                    type=layout_object.label,
                    id=layout_object.object_id
                )
                lp_layout.append(block)

            lp_layout.page_data = {'image_path': layout_object.image_path}

            self.lp_layouts.append(lp_layout)


    def visualize_ocr_text_dict(self, save_path):

        self.create_lp_layouts_from_ocr_text_dict()

        for lp_layout in self.lp_layouts:

            image_path = lp_layout.page_data['image_path']
            image = cv2.imread(image_path)

            image_with_drawing = draw_text(
                image,
                lp_layout,
                font_size=14,
                with_box_on_text=True,
                text_box_width=1,
                with_layout=True
            )

            image_with_drawing.save(os.path.join(save_path, 'viz-' + os.path.basename(image_path)))


    def full_articles_by_reading_order(self, save_path):

        image_file_names = self.ocr_text_dict.keys()

        with open(os.path.join(save_path, "full_articles_by_reading_order.txt"), "w") as f:

            for image_file_name in image_file_names:

                print(f'======{image_file_name}======', file=f)

                layout_objs = self.ocr_text_dict[image_file_name]

                headline_objs = [layobj for layobj in layout_objs if layobj.label == 'headline']
                article_objs = [layobj for layobj in layout_objs if layobj.label == 'article']

                for headline_obj in headline_objs:

                    print('***\n', file=f)

                    headline_fa_id = headline_obj.full_article_id
                    print(headline_obj.ocr_text, file=f)

                    relevant_article_objs = [article_obj for article_obj in article_objs \
                        if article_obj.full_article_id == headline_fa_id]

                    relevant_article_objs_filtered = [art for art in relevant_article_objs if art.reading_order_id is not None]
                    sorted_relevant_article_objs = sorted(relevant_article_objs_filtered, key=lambda x: x.reading_order_id)

                    for article_obj in sorted_relevant_article_objs:
                        print('-+-\n', file=f)
                        print(article_obj.ocr_text, file=f)
