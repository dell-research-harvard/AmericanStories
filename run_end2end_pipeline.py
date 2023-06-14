from stages.images_to_layouts import LayoutModel, Predictor
from stages.layouts_to_text import Layouts
from stages.text_to_embeddings import Embeddings

import json
import argparse

if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--parallel", help="", action="store_true")
    # parser.add_argument("--processes", help="", type=int)
    # args = parser.parse_args()

    #========== images-to-layouts ==========================================

    images = ['inputs/images/1589749-new-oxford-item-Jul-25-1957-p-1.jpg']

    with open('inputs/label_maps/label_map.json') as jf:
        label_map = {int(k): v for k, v in  json.load(jf).items()}

    layout_model = LayoutModel(
        config_path='inputs/model_files/config.yaml', 
        model_path='inputs/model_files/model_final.pth',
        label_map=label_map
    )

    layout_predictions = layout_model.detect(
        original_images=images, 
        prediction_type='default',
        output_type='d2'
    )

    #========== layouts-to-text ============================================

    layouts = Layouts(
        predictions=layout_predictions,
        original_images=images,
        label_map=label_map, 
        input_type='d2'
    )
    
    layouts.create_ocr_text_dict_parallel()
    layouts.create_full_article_ids(algo='mid')

    article_and_headline_text_dict = layouts.ocr_text_dict

    #========== text-to-embeddings =========================================
    
    newspaper_embeddings = Embeddings(model_name='roberta')

    newspaper_embeddings.embed(
        name='headline_embeddings', 
        data=[v for k, v in article_and_headline_text_dict.items() if 'headline' in k],
        pooling='mean', 
        layer=-2
    )

    newspaper_embeddings.embed(
        name='article_embeddings', 
        data=[v for k, v in article_and_headline_text_dict.items() if 'article' in k],
        pooling='mean', 
        layer=-2
    )

    newspaper_embeddings.umap_plot(
        names=['article_embeddings', 'headline_embeddings'],
        n_neighbors=15, 
        min_dist=0.1
    )