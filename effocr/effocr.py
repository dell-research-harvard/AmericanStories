import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from pytorch_metric_learning.utils.inference import InferenceModel, FaissKNN
from torchvision import transforms as T
import faiss
from tqdm import tqdm
import json
import argparse
import pytesseract
import numpy as np
from glob import glob
import os
import sys
from google.cloud import vision
import io
import requests
import base64
from timeit import default_timer as timer
import copy
from itertools import chain

from detectron2.engine import DefaultPredictor
from detectron2.config import LazyConfig
from detectron2.config import get_cfg
from mmdet.apis import init_detector, inference_detector
import mmcv
# https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/issues/59
sys.path.insert(0, "../")

from utils.datasets_utils import *
from models.encoders import *
from datasets.effocr_datasets import *
from utils.localizer_utils import *
from models.localizers import *
from utils.coco_utils import *
from utils.spell_check_utils import *
from models.classifiers import *

def run_gcv(
        image_file,
        client,
        lang="ja"
    ):
    """Call to GCV OCR"""

    with io.open(image_file, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image, image_context={"language_hints": [lang]})
    document = response.full_text_annotation
    return document.text


def run_baidu(
        image_path,
        access_token,
        request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic",
        lang="JAP"
    ):
    """Call to Baidu OCR"""

    with open(image_path, 'rb') as f:
        img = base64.b64encode(f.read())
    params = {"image":img,"language_type":lang}
    request_url = f"{request_url}?access_token={access_token}"
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)
    response_json = response.json()
    if response:
        return "".join(x['words'] for x in response_json['words_result'])
    else:
        return None


def create_dataset(image_paths, transform):
    """Create dataset for inference"""

    dataset = EffOCRInferenceDataset(image_paths, transform=transform)
    print(f"Length inference dataset: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    return dataloader


class EffOCR:

    def __init__(self,
            localizer_checkpoint,
            localizer_config,
            recognizer_checkpoint,
            recognizer_index,
            recognizer_chars,
            class_map,
            encoder,
            image_dir,
            vertical,
            char_transform,
            lang,
            device,
            save_chars=True,
            blacklist=None,
            score_thresh=0.5,
            score_thresh_word=0.5,
            knn=10,
            N_classes=None,
            anchor_margin=None,
            d2=False,
            timing=False
        ):
        # load localizer
        if not d2:
            if lang == "en":
                loc_config = {
                    "model.rpn_head.anchor_generator.scales":[2,8,32],
                    "classes":('char','word'), "data.train.classes":('char','word'),
                    "data.val.classes":('char','word'), "data.test.classes":('char','word'),
                    "model.roi_head.bbox_head.0.num_classes": 2,
                    "model.roi_head.bbox_head.1.num_classes": 2,
                    "model.roi_head.bbox_head.2.num_classes": 2,
                    "model.roi_head.mask_head.num_classes": 2,
                }
            elif lang == "jp":
                loc_config = {
                    "model.rpn_head.anchor_generator.scales":[2,8,32],
                    "classes":('char',), "data.train.classes":('char',),
                    "data.val.classes":('char',), "data.test.classes":('char',),
                    "model.roi_head.bbox_head.0.num_classes": 1,
                    "model.roi_head.bbox_head.1.num_classes": 1,
                    "model.roi_head.bbox_head.2.num_classes": 1,
                    "model.roi_head.mask_head.num_classes": 1,
                }
            else:
                raise NotImplementedError

            if device == 'cpu':
                print(localizer_config)
            localizer = init_detector(localizer_config, localizer_checkpoint, device=device, cfg_options=loc_config)
        else:
            cfg = LazyConfig.load(localizer_config).clone()
            exit(1)
            # cfg = LazyConfig.apply_overrides(cfg, args.opts)
            # cfg.merge_from_file()
            # cfg.model.roi_heads.box_predictors.test_score_thresh = score_thresh
            cfg.train.init_checkpoint = localizer_checkpoint
            localizer = DefaultPredictor(cfg)
            if lang == "en":
                cfg.model.roi_heads.num_classes = 2
                cfg.model.roi_heads.mask_head.num_classes = 2
                cfg.model.roi_heads.box_predictor.num_classes = 2
            elif lang == "jp":
                cfg.model.roi_heads.num_classes = 1
                cfg.model.roi_heads.mask_head.num_classes = 1
                cfg.model.roi_heads.box_predictor.num_classes = 1
            localizer = DefaultPredictor(localizer_config)

        # load recognizer encoder

        if N_classes is None:
            with open(recognizer_chars) as f:
                candidate_chars = f.read().split()
                candidate_chars_dict = {c:idx for idx, c in enumerate(candidate_chars)}
                print(f"{len(candidate_chars)} candidate chars!")
            recognizer_encoder = encoder.load(recognizer_checkpoint)
        else:
            recognizer_encoder = encoder.load(recognizer_checkpoint, n_cls=N_classes)
        recognizer_encoder.to(device)
        recognizer_encoder.eval()

        # configure recognizer

        if N_classes is None:
            knn_func = FaissKNN(
                index_init_fn=faiss.IndexFlatIP,
                reset_before=False, reset_after=False
            )
            recognizer = InferenceModel(recognizer_encoder, knn_func=knn_func)
            recognizer.load_knn_func(recognizer_index)
            if not blacklist is None:
                blacklist_ids = np.array([candidate_chars_dict[blc] for blc in blacklist])
                recognizer.knn_func.index.remove_ids(blacklist_ids)
                candidate_chars = [c for c in candidate_chars if not c in blacklist]
            class_map_dict = None
        else:
            with open(class_map) as f:
                class_map_dict = json.load(f)
            recognizer = recognizer_encoder
            candidate_chars = None


        # set default args

        self.localizer = localizer
        self.recognizer = recognizer
        self.recongizer_encoder = recognizer_encoder
        self.vertical = vertical
        self.double_clipped = True
        self.candidate_chars = candidate_chars
        self.char_transform = char_transform
        self.save_chars = save_chars
        self.image_dir = image_dir
        self.score_thresh = score_thresh
        self.score_thresh_word = score_thresh_word
        self.N_classes = N_classes
        self.class_map_dict = class_map_dict
        self.anchor_margin = anchor_margin
        self.lang = lang
        self.device = device
        self.LARGE_NUM = 1_000_000
        self.anchor_multiplier = 4
        self.knn = knn
        self.d2 = d2
        self.timing=timing

    def infer(self, im):

        # localizer inference
        if not self.d2:
            result = inference_detector(self.localizer, im)
        else:
            result = self.localizer(cv2.imread(im))
            print(result["instances"].pred_classes)
            print(result["instances"].pred_boxes)
            exit(1)


        # organize results of localizer inference

        if self.lang == "en":
            char_bboxes, word_bboxes = result if isinstance(result[0], np.ndarray) else result[0]
            char_bboxes, word_end_idx = self.en_preprocess(result)
        elif self.lang == "jp":
            char_bboxes, word_bboxes = self.jp_preprocess(result), None

        # get char crops for coordinates, store metadata about coordinates

        # im = np.array(Image.open(im).convert("RGB"))
        im = np.array(Image.fromarray(im).convert('RGB'))
        im_height, im_width = im.shape[0], im.shape[1]
        char_crops = []
        charheights, charbottoms = [], []

        for bbox in char_bboxes:
            try:
                x0, y0, x1, y1 = map(int, map(round, bbox))
                if self.double_clipped:
                    if self.vertical:
                        x0, y0, x1, y1 = 0, y0, im_width, y1
                    else:
                        x0, y0, x1, y1 = x0, 0, x1, im_height
                char_crops.append(self.char_transform(im[y0:y1,x0:x1,:]))
                if self.lang == "en":
                    charheights.append(bbox[3]-bbox[1])
                    charbottoms.append(bbox[3])
            except (RuntimeError, IndexError):
                continue

        if len(char_crops) == 0:
            # print("No content detected!")
            return None, None, None, None
        # else:
        #     print('Content Detected!')

        # perform batched recognizer inference


        if self.N_classes is None: # kNN

            with torch.no_grad():
                concat_char_dets = torch.stack(char_crops).to(self.device).squeeze(1)
                char_det_square_emb = self.recongizer_encoder(concat_char_dets)

            char_det_all_concat = torch.nn.functional.normalize(char_det_square_emb, p=2, dim=1)
            _, indices = self.recognizer.knn_func(char_det_all_concat, k=self.knn)
            index_list = indices.squeeze(-1).cpu().tolist()
            nearest_chars = [[self.candidate_chars[nn] for nn in nns] for nns in index_list]

            if self.lang == "en":
                assert len(nearest_chars) == len(charheights) == len(charbottoms), \
                    f"{len(nearest_chars)} == {len(charheights)} == {len(charbottoms)}; {nearest_chars}"

        else: # FFNN

            with torch.no_grad():
                outputs = self.recognizer(torch.stack(char_crops, dim=0))
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                predictions = logits.argmax(-1)
                predlist = predictions.detach().cpu().tolist()
            nearest_chars = [[self.class_map_dict[str(x)]] for x in predlist]

        # postprocessing (mostly for English)

        output_nns = ["".join(chars).strip() for chars in nearest_chars]
        output = "".join(x[0] for x in nearest_chars).strip()

        if self.lang == "en":
            output = self.en_postprocess(output, word_end_idx, charheights, charbottoms)

        return output, output_nns, char_bboxes, word_bboxes


    def en_preprocess(self, result):

        bboxes_char, bboxes_word = result if isinstance(result[0], np.ndarray) else result[0]
        sorted_bboxes_char = sorted(bboxes_char, key=lambda x: x[1] if self.vertical else x[0])
        sorted_bboxes_char = [x[:4] for x in sorted_bboxes_char if x[4] > self.score_thresh]
        sorted_bboxes_word = sorted(bboxes_word, key=lambda x: x[1] if self.vertical else x[0])
        sorted_bboxes_word = [x[:4] for x in sorted_bboxes_word if x[4] > self.score_thresh_word]

        word_end_idx = []
        closest_idx = 0
        sorted_bboxes_char_rights = [x[2] for x in sorted_bboxes_char]
        sorted_bboxes_word_lefts = [x[0] for x in sorted_bboxes_word]
        for wordleft in sorted_bboxes_word_lefts:
            prev_dist = self.LARGE_NUM
            for idx, charright in enumerate(sorted_bboxes_char_rights):
                dist = abs(wordleft-charright)
                if dist < prev_dist and charright > wordleft:
                    prev_dist = dist
                    closest_idx = idx
            word_end_idx.append(closest_idx)
        assert len(word_end_idx) == len(sorted_bboxes_word)

        return sorted_bboxes_char, word_end_idx


    def en_postprocess(self, line_output, word_end_idx, charheights, charbottoms):

        assert len(line_output) == len(charheights) == len(charbottoms), f"{len(line_output)} == {len(charheights)} == {len(charbottoms)}; {line_output}; {charbottoms}; {charheights}"

        if any(map(lambda x: len(x)==0, (line_output, word_end_idx, charheights, charbottoms))):
            return None

        outchars_w_spaces = [" " + x if idx in word_end_idx else x for idx, x in enumerate(line_output)]
        charheights_w_spaces = list(flatten([(self.LARGE_NUM, x) if idx in word_end_idx else x for idx, x in enumerate(charheights)]))
        charbottoms_w_spaces = list(flatten([(0, x) if idx in word_end_idx else x for idx, x in enumerate(charbottoms)]))
        charbottoms_w_spaces = charbottoms_w_spaces[1:] if charbottoms_w_spaces[0]==0 else charbottoms_w_spaces
        charheights_w_spaces = charheights_w_spaces[1:] if charheights_w_spaces[0]==self.LARGE_NUM else charheights_w_spaces

        line_output = "".join(outchars_w_spaces).strip()

        assert len(charheights_w_spaces) == len(line_output), \
            f"charheights_w_spaces = {len(charheights_w_spaces)}; output = {len(line_output)}; {charheights_w_spaces}; {line_output}"

        output_distinct_lower_idx = [idx for idx, c in enumerate(line_output) if c in create_distinct_lowercase()]

        if len(output_distinct_lower_idx) > 0 and not self.anchor_margin is None:
            avg_distinct_lower_height = sum(charheights_w_spaces[idx] for idx in output_distinct_lower_idx) / len(output_distinct_lower_idx)
            output_tolower_idx = [idx for idx, c in enumerate(line_output) \
                if abs(charheights_w_spaces[idx] - avg_distinct_lower_height) < self.anchor_margin * avg_distinct_lower_height]
            output_toupper_idx = [idx for idx, c in enumerate(line_output) \
                if charheights_w_spaces[idx] - avg_distinct_lower_height > self.anchor_margin * self.anchor_multiplier * avg_distinct_lower_height]
            avg_distinct_lower_bottom = sum(charbottoms_w_spaces[idx] for idx in output_distinct_lower_idx) / len(output_distinct_lower_idx)
            output_toperiod_idx = [idx for idx, c in enumerate(line_output) \
                if c == "-" and abs(charbottoms_w_spaces[idx] - avg_distinct_lower_bottom) < self.anchor_margin * avg_distinct_lower_height]

        if len(output_distinct_lower_idx) > 0 and not self.anchor_margin is None:
            nondistinct_lower = create_nondistinct_lowercase()
            line_output = "".join([c.lower() if idx in output_tolower_idx else c for idx, c in enumerate(line_output)])
            line_output = "".join([c.upper() if idx in output_toupper_idx and c in nondistinct_lower else c for idx, c in enumerate(line_output)])
            line_output = "".join(["." if idx in output_toperiod_idx else c for idx, c in enumerate(line_output)])

        return line_output


    def jp_preprocess(self, result):

        bboxes_char = result[0][0]
        sorted_bboxes_char = sorted(bboxes_char, key=lambda x: x[1] if self.vertical else x[0])
        sorted_bboxes_char = [x[:4] for x in sorted_bboxes_char if x[4] > self.score_thresh]

        return sorted_bboxes_char