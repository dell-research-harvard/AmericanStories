"""
Amended version of rule-based script to deal with CA outputs 
"""

import os
import sys
from glob import glob
from tqdm import tqdm
import json

from transformers import RobertaTokenizerFast
import logging

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
sys.path.append(grandparentdir)

from images_to_embeddings_pipeline.stages.layouts_to_text import Layouts
from scripts.quality_checking_functions import *


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


def predict_faro(path):

    # Reformat into previous format 
    with open(path) as f:
        new_format = json.load(f)

    reformatted_bboxes = []
    legibility_dict = {}
    for bbox in new_format["bboxes"]:
        reformatted_bboxes.append({
            "object_id": bbox["id"],
            "image_path": new_format["scan"]["ocr_text_url"],
            "image_file_name": "bboxes",
            "label": bbox["class"],
            "bbox": list(bbox["bbox"].values()),
            "ocr_text": bbox["raw_text"],
            "full_article_id": 0,
            "reading_order_id": 0
        }) 

        legibility_dict[bbox["id"]] = bbox["legibility"]



    old_format = {"bboxes": reformatted_bboxes} 

    with open('temp.json', 'w') as f:
        json.dump(old_format, f, indent=4)

    # Predict full articles and reading order 
    layouts = Layouts(predictions=None, label_map=label_map)
    layouts.load_ocr_text_dict('temp.json')
    layouts.create_fa_ids_bl()
    layouts.create_ro_ids_alt_bl()

    del new_format["bboxes"]

    return layouts.ocr_text_dict, new_format, legibility_dict


def create_full_articles(bbox_dict, spell_dict, legibility_dict, file_name, min_token_length=10):

    # instantiate tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    set_global_logging_level(logging.ERROR, ["transformers", "RobertaTokenizerFast"])

    # Keep article, headline and author lobjs with full_article_ids and non-word rate < 100%
    clean_bboxes = []
    fa_ids = []
    for bb_lo in bbox_dict["bboxes"]:
        bb = bb_lo.__dict__
        nwr = get_prop_non_words(bb["ocr_text"], spell_dict)
        if (((bb["label"] in ["article", "headline"]) and nwr < 1) or \
            bb["label"] == "author") and \
                bb["full_article_id"] is not None:
            clean_bboxes.append(bb)
            fa_ids.append(bb["full_article_id"])

    # Group by full_article_id
    fa_file = []

    for fa_id in set(fa_ids):
        fa_list = [bb for bb in clean_bboxes if bb["full_article_id"] == fa_id]
        ro_id = [bb["reading_order_id"] for bb in fa_list]

        # Merge into full article
        fa = {
            "image_file_name": file_name,  # Should be the same for everything within FA
            "image_path": "",  # Should be the same for everything within FA
            "object_id": [],  # List of IDs for each object
            "headline": "",  # From label and ocr_text
            "article": "",  # From label and ocr_text
            "byline": "",    # From label and ocr_text
            "bbox_list": [],  # List of individual bounding boxes
            "bbox": [],     # Grouped bbox
            "full_article_id": "",  # Should be the same for everything within FA
            "id": "",  # Combination of full_article_id and image_file_name to create a unique id
            "legibility": []
        }

        for ro_id in set(ro_id):
            for m in range(len(fa_list)):
                if fa_list[m]["reading_order_id"] == ro_id:

                    if fa["image_path"] == "":
                        fa["image_path"] = fa_list[m]["image_path"]
                    else:
                        assert fa["image_path"] == fa_list[m]["image_path"]

                    fa["object_id"].append(fa_list[m]["object_id"])

                    if fa_list[m]["label"] == "headline":
                        fa["headline"] = fa["headline"] + fa_list[m]["ocr_text"] + " "

                    if fa_list[m]["label"] == "article":
                        fa["article"] = fa["article"] + fa_list[m]["ocr_text"] + " "

                    if fa_list[m]["label"] == "author":
                        fa["byline"] = fa["byline"] + fa_list[m]["ocr_text"] + " "

                    fa["bbox_list"].append(fa_list[m]["bbox"])

                    if fa["full_article_id"] == "":
                        fa["full_article_id"] = fa_list[m]["full_article_id"]
                    else:
                        assert fa["full_article_id"] == fa_list[m]["full_article_id"]

                    if fa["id"] == "":
                        fa["id"] = str(fa_list[m]['full_article_id']) + "_" + file_name
                    else:
                        assert fa["id"] == str(fa_list[m]['full_article_id']) + '_' + file_name

                    fa["legibility"].append(legibility_dict[fa_list[m]["object_id"]])


        # Create merged bbox
        all_x = []
        all_y = []
        for bbox in fa["bbox_list"]:
            all_x.append(bbox[0])
            all_x.append(bbox[2])
            all_y.append(bbox[1])
            all_y.append(bbox[3])

        fa["bbox"] = [min(all_x), min(all_y), max(all_x), max(all_y)]

        # Reject articles with empty or very short length
        tokens = tokenizer(fa["article"], truncation=False)['input_ids']

        if len(tokens) > min_token_length:
            fa_file.append(fa)

    return fa_file


if __name__ == '__main__':

    save_dir = '/mnt/data01/faro/rule_based_new_first_of_months/'
    os.makedirs(save_dir, exist_ok=True)

    ocr_json_pattern = '/mnt/data01/faro/new_first_of_months/**/*.json'

    label_map = {0: 'article',
                 1: 'author',
                 2: 'cartoon_or_advertisement',
                 3: 'headline',
                 4: 'image_caption',
                 5: 'masthead',
                 6: 'newspaper_header',
                 7: 'page_number',
                 8: 'photograph',
                 9: 'table'}
    
    spell_dict = load_lowercase_spell_dict('/mnt/data01/hunspell_word_dict_lower/hunspell_word_dict_lower')
    # spell_dict = load_lowercase_spell_dict('/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/hunspell_word_dict_lower')

    for ocr_json_path in tqdm(glob(ocr_json_pattern)):

        try:
            # Generate full article and reading order IDs
            text_dict, file_with_metadata, legibility = predict_faro(ocr_json_path)

            # Sort metadata 
            file_with_metadata["org_bbox_path"] = ocr_json_path
            file_with_metadata["paper_date"] = file_with_metadata["edition"]["date"]

            file_name = "_".join([
                file_with_metadata["lccn"]["lccn"],
                file_with_metadata["edition"]["date"],
                "e" + str(file_with_metadata["edition"]["edition"]),
                "p" + file_with_metadata["page_number"]])

            # Merge into full articles
            fa_dict = create_full_articles(text_dict, spell_dict, legibility, file_name, min_token_length=10)
            file_with_metadata["articles"] = fa_dict

            # Save (fails if file already exists)
            year = file_with_metadata['paper_date'][:4]

            os.makedirs(f'{save_dir}/faro_{year}/', exist_ok=True)

            try:
                with open(f'{save_dir}/faro_{year}/{file_name}.json', 'x') as f:
                    json.dump(file_with_metadata, f, indent=4)
            except:
                print(f'{save_dir}/faro_{year}/{file_name}.json')
                print(ocr_json_path)
                print("**")

            # save_end = ocr_json_path.split('/')[-2:]
            # save_batch_dir = os.path.join(save_dir, save_end[0])
            # save_path = os.path.join(save_dir, save_end[0], save_end[1])
            # with open(save_path, 'w') as f:
            #     json.dump(layouts.ocr_text_dict, f, default=lambda x: x.__dict__, indent=2)

        except:
            print("ERROR:", ocr_json_path)
