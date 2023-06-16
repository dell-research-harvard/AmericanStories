import os
import json
import argparse
from timeit import default_timer as timer
import sys
import torch
from glob import glob
import logging

from stages.images_to_layouts import LayoutModel, LineModel, BackboneClassifierHead
from stages.layouts_to_text import Layouts
from stages.pdfs_to_images import pdfs_to_images
from effocr.effocr import EffOCR
from effocr.dataset_utils import create_paired_transform
from effocr.encoders import VitEncoder
from models.encoders import AutoEncoderFactory

logging.disable(logging.WARNING)


if __name__ == '__main__':

    print("Start!")
    _start_time = timer()

    #========== inputs =============================================

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_source_path",
        help="Path to PDF source files; PDFs may be nested in sub-directories")
    parser.add_argument("--output_save_path",
        help="Path to directory for saving outputs of pipeline inference")
    parser.add_argument("--config_path_layout",
        help="Path to Detectron2 config file")
    parser.add_argument("--config_path_line",
        help="Path to Detectron2 config file")
    parser.add_argument("--checkpoint_path_layout",
        help="Path to Detectron2 compatible checkpoint file (model weights file)")
    parser.add_argument("--checkpoint_path_line",
        help="Path to Detectron2 compatible checkpoint file (model weights file)")
    parser.add_argument("--label_map_path_layout",
        help="Path to JSON file mapping numeric object classes to their labels")
    parser.add_argument("--label_map_path_line",
        help="Path to JSON file mapping numeric object classes to their labels")
    parser.add_argument("--data_source",
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
    parser.add_argument("--classifier_head_checkpoint_path",
        help="Path to backbone classifier head checkpoint; if supplied, classifier head is deployed")
    parser.add_argument("--language",
        help="")
    parser.add_argument("--effocr_recognizer_dir",
        help="")
    parser.add_argument("--effocr_localizer_dir",
        help="")
    parser.add_argument('--effocr_timm_auto_model', default=None,
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
    parser.add_argument("--line_model_onnx_export", default=None,
        help="")
    # parser.add_argument("--effocr_batch_size", type=int, default=250,
    #     help="")
    args = parser.parse_args()
    print('Running!')

    pdf_source_path = args.pdf_source_path
    output_save_path = args.output_save_path
    img_save_path = os.path.join(output_save_path, "images")
    nested = args.nonnested
    resize = args.resize

    config_path_layout = args.config_path_layout
    config_path_line = args.config_path_line

    checkpoint_path_layout = args.checkpoint_path_layout
    checkpoint_path_line = args.checkpoint_path_line

    label_map_path_layout = args.label_map_path_layout
    label_map_path_line = args.label_map_path_line

    effocr_recognizer_dir = args.effocr_recognizer_dir
    effocr_localizer_dir = args.effocr_localizer_dir
    effocr_timm_auto_model = args.effocr_timm_auto_model
   # effocr_batch_size = args.effocr_batch_size

    filter_dup = args.filter_duplicates
    visualize_ocr_text = args.viz_ocr_texts
    output_full_articles = args.full_articles_out

    data_source = args.data_source
    device = args.device

    line_model_onnx_export = args.line_model_onnx_export

    classifier_head_checkpoint_path = args.classifier_head_checkpoint_path

    os.makedirs(output_save_path, exist_ok=True)

    #========== pdfs-to-images =============================================

    start_time = timer()

    pdfs_to_images(
        source_path=pdf_source_path,
        save_path=img_save_path,
        data_source=data_source,
        nested=nested,
        resize=resize,
        deskew=False
    )

    images = glob(os.path.join(img_save_path, "*.png")) + glob(os.path.join(img_save_path, "*.jpg"))

    pdf_to_images_time = timer() - start_time
    print(f'pdfs-to-images: {pdf_to_images_time}')

    #========== images-to-layouts ==========================================

    start_time = timer()

    with open(label_map_path_layout) as jf:
        label_map_layout = {int(k): v for k, v in  json.load(jf).items()}

    if classifier_head_checkpoint_path:
        classifier_head = BackboneClassifierHead()
        classifier_head.classifier.load_state_dict(torch.load(classifier_head_checkpoint_path, map_location=torch.device('cpu')))
        classifier_head.classifier.to(device)
        classifier_head.classifier.eval()
    else:
        classifier_head = None

    #finetuned detectron2
    layout_model = LayoutModel(
        config_path=config_path_layout,
        model_path=checkpoint_path_layout,
        device=device,
        filter_duplicates=filter_dup,
        classifier_head=classifier_head
    )

    layout_predictions = layout_model.detect(
        image_paths=images,
        data_source=data_source,
        prediction_type='default',
        out_path=output_save_path
    )

    images_to_layout_time = timer() - start_time
    print(f'images-to-layouts: {images_to_layout_time}')

    #========== layouts-to-text ============================================

    start_time = timer()

    with open(label_map_path_line) as jf:
        label_map_line = {int(k): v for k, v in  json.load(jf).items()}

    #GUPPY - mnt/data02/e2e2e/line_det....
    line_model = LineModel(
        config_path=config_path_line,
        model_path=checkpoint_path_line,
        device=device,
        onnx_export=line_model_onnx_export
    )

    if not effocr_timm_auto_model is None:
        encoder = AutoEncoderFactory("timm", effocr_timm_auto_model)
    else:
        encoder = VitEncoder

    # As before in standard EffOCR
    effocr_model = EffOCR(
        localizer_checkpoint=os.path.join(args.effocr_localizer_dir, "best_bbox_mAP.pth"),
        localizer_config=glob(os.path.join(args.effocr_localizer_dir, "*.py"))[0],
        recognizer_checkpoint=os.path.join(args.effocr_recognizer_dir, "enc_best.pth"),
        recognizer_index=os.path.join(args.effocr_recognizer_dir, "ref.index"),
        recognizer_chars=os.path.join(args.effocr_recognizer_dir, "ref.txt"),
        class_map=os.path.join(args.effocr_recognizer_dir, "class_map.json"),
        encoder=encoder, image_dir = None, vertical=False, device=device,
        #batch_size=effocr_batch_size,
        char_transform=create_paired_transform(lang=args.language),
        lang=args.language, score_thresh=0.5,
        score_thresh_word=0.5, spell_check=True,
        N_classes=None, anchor_margin=None,
    )

    layouts = Layouts(
        layout_predictions=layout_predictions,
        layout_label_map=label_map_layout,
        line_model=line_model,
        line_label_map=label_map_line,
        effocr_model=effocr_model,
        output_dir=output_save_path,
    )

    layout_coco, line_coco, char_coco = \
        layouts.create_ocr_text_dict(saving=args.saving, tesseract=args.tesseract, trocr=args.trocr)
    # layouts.create_fa_ids()
    # layouts.create_ro_ids()

    layouts_to_text_time = timer() - start_time
    print(f'layouts-to-text: {layouts_to_text_time}')

    #========== outputs =============================================

    with open(os.path.join(output_save_path, 'layout_coco.json'), 'w') as jf:
        json.dump(layout_coco, jf, indent=2)
    with open(os.path.join(output_save_path, 'line_coco.json'), 'w') as jf:
        json.dump(line_coco, jf, indent=2)
    with open(os.path.join(output_save_path, 'char_coco.json'), 'w') as jf:
        json.dump(char_coco, jf, indent=2)

    article_and_headline_text_dict = layouts.ocr_text_dict
    with open(os.path.join(output_save_path, 'ocr_text.json'), 'w') as jf:
        json.dump(article_and_headline_text_dict, jf, default=lambda x: x.__dict__, indent=2)

    print(f'Total time elapsed: {timer() - _start_time}', file=sys.stdout)

    if visualize_ocr_text:
        layouts.visualize_ocr_text_dict(save_path=output_save_path)

    if output_full_articles:
        layouts.full_articles_by_reading_order(save_path=output_save_path)