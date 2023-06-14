

COCO_JSON_SKELETON = {
    "info": {"":""}, 
    "licenses": [{"":""}], 
    "images": [], 
    "annotations": [], 
    "categories": [{"id": 0, "name": "char"}]
}


def create_coco_anno_entry(x, y, w, h, ann_id, image_id, cat_id=0, text=None):
    if text is None:
        return {
            "segmentation": [[int(x), int(y), int(x)+int(w), int(y), int(x)+int(w), int(y)+int(h), int(x), int(y)+int(h)]], 
            "area": w*h, "iscrowd": 0, 
            "image_id": image_id, "bbox": [int(x), int(y), int(w), int(h)], 
            "category_id": cat_id, "id": ann_id, "score": 1.0
        }
    else:
        return {
            "segmentation": [[int(x), int(y), int(x)+int(w), int(y), int(x)+int(w), int(y)+int(h), int(x), int(y)+int(h)]], 
            "area": w*h, "iscrowd": 0, 
            "image_id": image_id, "bbox": [int(x), int(y), int(w), int(h)], 
            "category_id": cat_id, "id": ann_id, "score": 1.0,
            "text": text
        }



def create_coco_image_entry(path, height, width, image_id, text=None):
    if text is None:
        return {
            "file_name": path, 
            "height": height, 
            "width": width, 
            "id": image_id
        }
    else:
        return {
            "file_name": path, 
            "height": height, 
            "width": width, 
            "id": image_id,
            "text": text
        }