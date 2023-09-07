import pickle
import json
import os
from datetime import datetime

from tqdm import tqdm
from glob import glob
import pandas as pd
import numpy as np

import multiprocessing
from functools import partial

from data_fns import clusters_from_edges, edges_from_clusters, import_single_scan_labelled_data
from data.clean_labelled_sample import get_prop_non_words, load_lowercase_spell_dict, clean


def is_banner_headline(bbox, page_size):

    scan_width = page_size[0]
    scan_height = page_size[1]

    # Count lines
    split = bbox["ocr_text"].split("\n")
    split = [t for t in split if len(t.strip()) > 1]
    n_lines = len(split)

    bb_width = bbox["bbox"][2] - bbox["bbox"][0]
    bb_line_height = (bbox["bbox"][3] - bbox["bbox"][1])/n_lines

    if bb_width > ((2*scan_width)/3) and bb_line_height > (scan_height/50):
        return True
    else:
        return False


def directly_above_cartoon_or_ad(hl_bbox, cads, dims):

    # Overlapping horizontally, by more than 1% of the page width
    possible_cads = [b for b in cads if
        max(0, min(b["bbox"]["y0"], hl_bbox["bbox"]["y0"]) - max(b["bbox"]["x0"], hl_bbox["bbox"]["x0"])) >
        dims[0]/100]

    # Directly above or slightly overlapping vertically
    possible_cads = [b for b in possible_cads if abs(hl_bbox["bbox"]["y1"] - b["bbox"]["y0"]) < (dims[1]/50)]

    if len(possible_cads) > 0:
        return True
    else:
        return False


def create_fa_ids(bbox_list, dims):

    # Articles
    articles_with_fa_ids = []
    headlines = []
    authors = []
    cad_list = []

    for obj in bbox_list:

        # Articles
        if obj["class"] == "article":
            obj["full_article_id"] = obj["id"] + 1
            articles_with_fa_ids.append(obj)

        # Headlines and bylines
        elif obj["class"] == "headline":
            obj["full_article_id"] = None
            headlines.append(obj)
        elif obj["class"] == "author":
            obj["full_article_id"] = None
            authors.append(obj)

        # Pictue type objects 
        elif obj["class"] in ["cartoon_or_advertisement", "photograph"]:
            cad_list.append(obj)

    # Headline-article association
    headlines_with_fa_ids = []
    for hl in headlines:

        # if is_banner_headline(hl, dims):
        #     headlines_with_fa_ids.append(hl)
        #     continue

        # if directly_above_cartoon_or_ad(hl, cad_list, dims):
        #     headlines_with_fa_ids.append(hl)
        #     continue

        headline_left = hl['bbox']["x0"]
        headline_right = hl['bbox']["x1"]
        headline_bottom = hl['bbox']["y1"]

        # Overlapping horizontally, by more than 1% of the page width
        selected_arts = [b for b in articles_with_fa_ids if
            max(0, min(b["bbox"]["x1"], headline_right) - max(b["bbox"]["x0"], headline_left)) >
            dims[0]/100]
        
        # Above or slightly overlapping vertically
        selected_arts = [b for b in selected_arts if headline_bottom < b["bbox"]["y0"] + (dims[1]/50)]

        # Not too far above
        selected_arts = [b for b in selected_arts if b["bbox"]["y0"] - headline_bottom < (dims[1]/10)]

        # Sort by closest below
        sorted_arts = sorted(selected_arts, key=lambda x: x["bbox"]["y0"])

        # Take full article ID
        if len(sorted_arts) > 0:

            hl["full_article_id"] = sorted_arts[0]["full_article_id"]
        
        headlines_with_fa_ids.append(hl)

    # Byline-article association 
    authors_with_fa_ids = []

    for bl in authors:

        author_left = bl['bbox']["x0"]
        author_right = bl['bbox']["x1"]
        author_bottom = bl['bbox']["y1"]

        # Overlapping horizontally, by more than 1% of the page width
        selected_arts = [b for b in articles_with_fa_ids if
            max(0, min(b["bbox"]["x1"], author_right) - max(b["bbox"]["x0"], author_left)) >
            dims[0]/100]

        # Above or slightly overlapping vertically
        selected_arts = [b for b in selected_arts if author_bottom < b["bbox"]["y0"] + (dims[1]/50)]

        # Sort by closest below
        sorted_arts = sorted(selected_arts, key=lambda x: x["bbox"]["y0"])

        # Take full article ID
        if len(sorted_arts) > 0:

            bl["full_article_id"] = sorted_arts[0]["full_article_id"]
        
        authors_with_fa_ids.append(bl)

    return articles_with_fa_ids + headlines_with_fa_ids + authors_with_fa_ids


def create_ro_ids(bbox_list, horizontal_margin=0.40):

    bboxes_with_ro_ids = []

    # get unique full article IDs in image; iterate through them
    unique_full_article_ids = list(set([bbox["full_article_id"] for bbox in bbox_list if bbox["full_article_id"]]))
    for unique_fa_id in unique_full_article_ids:

        # Articles
        # all article bboxes in this full article
        art_bboxs_in_this_fa = [bbox for bbox in bbox_list if bbox["full_article_id"] == unique_fa_id and \
                                bbox["class"] == "article"]
                
        # sort horizontally
        horizontal_sorted_art_bboxs_in_this_fa = sorted(art_bboxs_in_this_fa, key=lambda x: x['bbox']["x0"])
        
        # iterate through sorted layout objects, sorting similarly vertically positioned objects by horizontal position
        ro_sorted_layobjs_for_unique_fa_id = []

        for hslayobj in horizontal_sorted_art_bboxs_in_this_fa:

            if hslayobj["id"] in [o["id"] for o in ro_sorted_layobjs_for_unique_fa_id]:
                continue

            width_i = hslayobj["bbox"]["y1"] - hslayobj["bbox"]["x0"]
            similar_xpos_layobjs = [hslayobj]

            for _hslayobj in horizontal_sorted_art_bboxs_in_this_fa:

                if hslayobj["id"] == _hslayobj["id"]:
                    continue

                if _hslayobj["id"] in [o["id"] for o in ro_sorted_layobjs_for_unique_fa_id]:
                    continue

                width_j = _hslayobj["bbox"]["x1"] - _hslayobj["bbox"]["x0"]
                horizontal_gap = horizontal_margin*min(width_j, width_i)

                if abs(hslayobj["bbox"]["x0"] - _hslayobj["bbox"]["x0"]) < horizontal_gap:
                    similar_xpos_layobjs.append(_hslayobj)

            vert_sorted_similar_height_layobjs = sorted(similar_xpos_layobjs, key=lambda x: x["bbox"]["y0"])

            for layobj in vert_sorted_similar_height_layobjs:
                ro_sorted_layobjs_for_unique_fa_id.append(layobj)

        # actually create reading order ids based on list order
        for idx, layobj in enumerate(ro_sorted_layobjs_for_unique_fa_id):
            layobj["reading_order_id"] = idx

        bboxes_with_ro_ids.extend(ro_sorted_layobjs_for_unique_fa_id)

        # Headlines and author (bylines): sort vertically
        headlines_for_unique_fa_id = [bbox for bbox in bbox_list \
                                        if bbox["full_article_id"] == unique_fa_id and \
                                        (bbox["class"] in ['headline', 'author'])]
        
        vert_sorted_headlines_for_unique_fa_id = sorted(headlines_for_unique_fa_id, key=lambda x: x["bbox"]["y0"])
        n_hl = len(vert_sorted_headlines_for_unique_fa_id)

        for idx, layobj in enumerate(vert_sorted_headlines_for_unique_fa_id):
            layobj["reading_order_id"] = idx - n_hl

        bboxes_with_ro_ids.extend(vert_sorted_headlines_for_unique_fa_id)

    return bboxes_with_ro_ids


def group_articles(bbox_list, name):

    # Group by full_article_id
    all_fas = []

    fa_ids = list(set([b["full_article_id"] for b in bbox_list]))

    for fa_id in fa_ids:
        fa_list = [bb for bb in bbox_list if bb["full_article_id"] == fa_id]
        sorted_fa_list = sorted(fa_list, key=lambda x: x['reading_order_id'])

        # Merge into full article
        fa = {
            "object_ids": [b["id"] for b in sorted_fa_list],  # List of IDs for each object
            "headline": "\n\n".join([b["raw_text"] for b in sorted_fa_list if b["class"] == "headline"]),
            "article": "\n\n".join([b["raw_text"] for b in sorted_fa_list if b["class"] == "article"]),
            "byline": "\n\n".join([b["raw_text"] for b in sorted_fa_list if b["class"] == "author"]),
            "bbox_list": [b['bbox'] for b in sorted_fa_list],  # List of individual bounding boxes
            "bbox": [],     # Grouped bbox
            "full_article_id": fa_id,  # Should be the same for everything within FA
            "id": str(fa_id) + "_" + name,  # Combination of full_article_id and image_file_name to create a unique id
        }

        # Create merged bbox
        fa["bbox"] = [min([b["x0"] for b in fa["bbox_list"]]),
                      min([b["y0"] for b in fa["bbox_list"]]),
                      max([b["x1"] for b in fa["bbox_list"]]),
                      max([b["y1"] for b in fa["bbox_list"]])]

        if fa['article'].replace("\n", "").strip() != "":
            all_fas.append(fa)

    return all_fas


def open_labels():

    # # with open('/mnt/data01/faro/labelled_data/ca_multi_page_fa.json') as f:
    # with open('/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/faro/labelled_data/ca_multi_page_fa.json') as f:
    with open('/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/faro/labelled_data/gold_fa_labelled.json') as f:
        raw_labels = json.load(f)

    print(len(raw_labels))
    # pages = sum([(len([d for d in list(edition['data'].values()) if d !='/data/local-files/?d=images/white_box.png'])) for edition in raw_labels])
    # print(pages)
    # print(pages/13)

    article_count = 0

    label_dict = {}

    for edition in raw_labels:

        annots = edition['annotations'][0]['result']

        unique_images = list(set([annot['to_name'] for annot in annots]))

        labels_seen = []

        for img in unique_images:

            img_path = edition['data'][img]

            image_annots = [annot for annot in annots if annot['to_name'] == img and 'rectanglelabels' in annot['value'] and len(annot['value']['rectanglelabels']) > 0]

            label_dict[img_path.split("/")[-1]] = image_annots

            labels = list(set([annot['value']['rectanglelabels'][0] for annot in image_annots]))

            for lab in labels:
                if lab in labels_seen:
                    print("multi-page:", lab)
                labels_seen.append(lab)

        article_count += len(labels_seen)

    print(article_count)


    gt_dict = {}
    gt_clusters = {}
    for scan, labels in label_dict.items():

        unique_labs = list(set([annot['value']['rectanglelabels'][0] for annot in labels]))

        cluster_dict = {}
        for lab in unique_labs:

            art_labs = [l for l in labels if l['value']['rectanglelabels'][0] == lab]
            
            cluster_dict[lab] = [int(a["id"].split("_")[0]) for a in art_labs if a['id'] != 'imXp3BgwkZ']
        
        edges = edges_from_clusters(cluster_dict)
        gt_dict[scan] = edges

        clus = clusters_from_edges(edges)

        clu_list = [sorted(clu) for clu in list(clus.values())]

        gt_clusters[scan] = clu_list

    return gt_dict, gt_clusters


def get_inf_data(gt_edges):

    with open('/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/faro/labelled_data/preds.json') as f:
    # with open('/mnt/data01/faro/labelled_data/preds.json') as f:
        data = json.load(f)

    reformatted_data = []

    for dat in data:

        if dat["data"]["pcr"].split("/")[-1] in gt_edges:

            bbox_dict = {}

            for bbox in dat["annotations"][0]["result"]:

                if bbox['id'] not in bbox_dict:
                    bbox_dict[bbox['id']] = {
                            "id": bbox['id'],
                            "bbox": {
                                "x0": bbox["value"]["x"],
                                "y0": bbox["value"]["y"],
                                "x1": bbox["value"]["x"] + bbox["value"]["width"],
                                "y1": bbox["value"]["y"] + bbox["value"]["height"]
                            },
                            "legibility": "NA"
                        }

                if bbox["type"] == "labels":
                    bbox_dict[bbox['id']]["class"] = bbox["value"]["labels"][0]
                elif bbox["type"] == "textarea":
                    bbox_dict[bbox['id']]["raw_text"] = bbox["value"]["text"][0]


            reformatted_data.append({
                    'scan': {'jp2_url': dat["data"]["pcr"].split("/")[-1]},
                    'bboxes': list(bbox_dict.values())
                }
    )
    
    for scan in reformatted_data:
        widths = [b["bbox"]["x1"] for b in scan["bboxes"]]
        heights = [b["bbox"]["y1"] for b in scan["bboxes"]]

        scan["scan"]["height"] = max(heights) 
        scan["scan"]["width"] = max(widths) 

    text_dict = {}
    for scan in reformatted_data:

        scan_id = scan['scan']['jp2_url']
        text_dict[scan_id] = {}

        for bbox in scan["bboxes"]:
            if "raw_text" in bbox:
                text_dict[scan_id][bbox["id"]] = bbox["raw_text"]

    return reformatted_data, text_dict


def eval_headlines():

    size_dict = {
    "0004": [5390, 6974],
    "0015": [3517, 4889],
    "0042": [4664, 6688],
    "0075": [6908, 9600],
    "0102": [6952, 9576],
    "0131": [6477, 8510],
    "0272": [4976, 6416],
    "0424": [3496, 6151],
    "0810": [4940, 7025],
    "1230": [4732, 7061]
    }

    gt_edges, _ = open_labels()

    print(gt_edges.keys())

    tps = 0
    fps = 0
    fns = 0

    list_of_file_paths = glob('/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/faro/labelled_data/gold/**')

    multi_bbox_count = 0

    for scan_path in list_of_file_paths:

        with open(scan_path) as f:
            scan_dict = json.load(f)

        # Name
        name = scan_path.split("/")[-1][:-5].split("_")[-1]

        hl_list = [b['id'] for b in scan_dict["bboxes"] if b['class'] == 'headline']
        bl_list = [b['id'] for b in scan_dict["bboxes"] if b['class'] == 'author']
        art_list = [b['id'] for b in scan_dict["bboxes"] if b['class'] == 'article']

        print('Outer', len(bl_list))

        # Count multi-art bboxes 
        gts = gt_edges[name+".jpg"]
        gt_art_edges = [e for e in gts if e[0] in art_list and e[1] in art_list]
        multi_bbox_count += len(clusters_from_edges(gt_art_edges))
        
        # Full article IDs
        dimensions = [size_dict[name][0], size_dict[name][1]]
        bbox_list_fa_ids = create_fa_ids(scan_dict["bboxes"], dims=dimensions)

        # Reading orders
        bbox_list_faro_ids = create_ro_ids(bbox_list_fa_ids)

        # Merge
        full_articles = group_articles(bbox_list_faro_ids, name)

        scan_dict["full articles"] = full_articles

        # Extract edges
        pred_clusters = [art["object_ids"] for art in scan_dict["full articles"]]

        cluster_dict = {}
        for i, clu in enumerate(pred_clusters):
            cluster_dict[i] = clu        
        pred_edges = edges_from_clusters(cluster_dict)

        # Reduce to headline edge
        # pred_headline_edges = [e for e in pred_edges if (e[0] in hl_list and e[1] in art_list) or (e[1] in hl_list and e[0] in art_list)]
        pred_headline_edges = [e for e in pred_edges if (e[0] in bl_list and e[1] in art_list) or (e[1] in bl_list and e[0] in art_list)]

        gts = gt_edges[name+".jpg"]
        # gt_headline_edges = [e for e in gts if (e[0] in hl_list and e[1] in art_list) or (e[1] in hl_list and e[0] in art_list)]
        gt_headline_edges = [e for e in gts if (e[0] in bl_list and e[1] in art_list) or (e[1] in bl_list and e[0] in art_list)]

        # print(pred_headline_edges)
        # print(gt_headline_edges)

        tps += len([x for x in pred_headline_edges if x in gt_headline_edges or [x[1], x[0]] in gt_headline_edges])
        fp_list = [x for x in pred_headline_edges if x not in gt_headline_edges and [x[1], x[0]] not in gt_headline_edges]
        fps += len(fp_list)
        fn_list = [x for x in gt_headline_edges if x not in pred_headline_edges and [x[1], x[0]] not in pred_headline_edges]

        single_fn_list = []
        for f in fn_list:
            if f[0] in hl_list:
                if len([p for p in pred_headline_edges if f[0] in p]) == 0:
                    single_fn_list.append(f)
            elif f[1] in hl_list:
                if len([p for p in pred_headline_edges if f[1] in p]) == 0:
                    single_fn_list.append(f)
        fns += len(single_fn_list)

    #     # if len(single_fn_list) > 0:
    #     #     print(name)
    #     #     print(pred_edges)
    #     #     print(pred_clusters)
    #     #     print(single_fn_list)
    #     #     print("**")
    #     #     for p in single_fn_list:
    #     #         print(p)
    #     #         print("**")
    #     #         print(text_dict[name][p[0]])
    #     #         print("**")
    #     #         print(text_dict[name][p[1]])
    #     #         print("**")
    #     #     print("##############################")

    #     if len(fp_list) > 0:
    #         print(name)
    #         print(pred_edges)
    #         print(pred_clusters)
    #         print(fp_list)
    #         print("**")
    #         for p in fp_list:
    #             print(p)
    #             print("**")
    #             print(text_dict[name][p[0]])
    #             print("**")
    #             print(text_dict[name][p[1]])
    #             print("**")
    #         print("##############################")

    print(tps, fps, fns)
    recall = tps/(tps+fns)
    precision = tps/(tps + fps)
    F1 = 2 * (precision * recall)/(precision + recall)
    print("Recall:", round(recall*100, 1), "Precision", round(precision*100, 1), "F1:", round(F1*100, 1))

    print('Multi bbox articles', multi_bbox_count)



def main(scan_path):

    try:
        with open(scan_path) as f:
            scan_dict = json.load(f)

        # Name
        file_name = scan_path.split("/")[-1]

        if scan_dict["page_number"] == "na":
            scan_dict["page_number"] = 1

        if "edition" not in scan_dict:
            scan_dict["edition"] = {}
            date = file_name.split("_")[-2]
            formatted_date = date[:4] + "-" + date[4:6] + "-" + date[6:8]
            scan_dict["edition"]["date"] = formatted_date

        if "width" not in scan_dict["scan"]:
            scan_dict["scan"]["width"] = max([b["bbox"]["x1"] for b in scan_dict["bboxes"]])
            scan_dict["scan"]["dimensions_approx"] = True
        if "height" not in scan_dict["scan"]:
            scan_dict["scan"]["height"] = max([b["bbox"]["y1"] for b in scan_dict["bboxes"]])
            scan_dict["scan"]["dimensions_approx"] = True


        name = "_".join([
                    scan_dict["edition"]["date"],
                    "p" + str(scan_dict["page_number"]),
                    file_name
                ])

        # Full article IDs
        dimensions = [scan_dict["scan"]["width"], scan_dict["scan"]["height"]]
        bbox_list_fa_ids = create_fa_ids(scan_dict["bboxes"], dims=dimensions)

        # Reading orders 
        bbox_list_faro_ids = create_ro_ids(bbox_list_fa_ids)
        scan_dict["bboxes"] = bbox_list_faro_ids

        # Merge
        full_articles = group_articles(bbox_list_faro_ids, name)

        scan_dict["full articles"] = full_articles

        # Save
        year = scan_dict["edition"]["date"][:4]

        os.makedirs(f'/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/ca_rule_based_fa_clean/faro_{year}/', exist_ok=True)

        try:
            with open(f'/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/ca_rule_based_fa_clean/faro_{year}/{name}', 'x') as f:
                json.dump(scan_dict, f, indent=4)
        except Exception as e:
            t=1
            print(e)
            raise e
            # print("Done")
            # print(f'/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/ca_rule_based_fa_clean/faro_{year}/{name} already exists')

    except Exception as e:
        print(f'ERROR in filepath {scan_path}')
        print(e)
        # with open('error_log.json') as f:
        #     error_log = json.load(f)
        # error_log.append(scan_path)
        # with open('error_log.json', 'w') as f:
        #     json.dump(error_log, f)



if __name__ == '__main__':
    
    # eval_headlines()

    list_of_file_paths = []
    for root, dirs, files in os.walk('/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/e2e2e/end-to-end-pipeline/pipeline_egress/ca_pipeline_2'):
    # # for root, dirs, files in os.walk('/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/e2e2e/end-to-end-pipeline/pipeline_egress/dbx_pipeline'):
        for file in files:
            if file.endswith(".json"):
                list_of_file_paths.append(os.path.join(root, file))

    with open(f'files_processed{datetime.now()}.json', 'w') as f:
        json.dump(list_of_file_paths, f)

    for scan_path in tqdm(list_of_file_paths):
        main(scan_path)

    print(len(list_of_file_paths))

    num_processes = multiprocessing.cpu_count() - 5
    pool = multiprocessing.Pool(processes=num_processes)

    full_arts_partial = partial(main)

    i = 0
    with tqdm(total=len(list_of_file_paths), desc="Getting full articles") as pbar:
        for _ in pool.imap_unordered(full_arts_partial, list_of_file_paths):
            pbar.update()
            i += 1

            if i > 9:
                break
