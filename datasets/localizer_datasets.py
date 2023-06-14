from PIL import Image
from torchvision.datasets import CocoDetection
from torch.utils.data import Dataset

from utils.localizer_utils import *


class CocoSegmentation(CocoDetection):

    def __init__(self, contract=0.2, vertical=False, category_id=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contract = contract
        self.vertical = vertical
        self.category_id = category_id

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB"), path

    def _load_target(self, id: int):
        anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        assert len(anns) > 0, f"No annotations for id {id}"
        h, w = self.coco.annToMask(anns[0]).shape
        mask = np.zeros((h,w))
        bboxes = []
        bboxes_pct = []
        for ann in anns:
            if ann['category_id'] != self.category_id:
                continue
            bbox = find_bbox(self.coco.annToMask(ann))
            if bbox is not None and abs(bbox[3]-bbox[2]) > 0 and abs(bbox[1]-bbox[0]) > 0:
                u, d, l, r = bbox
                bboxes.append([l,u,r,d])
                bboxes_pct.append([l/w,u/h,r/w,d/h])
                vmarg, hmarg = int(abs(u-d) * self.contract), int(abs(l-r) * self.contract)
                if self.vertical:
                    mask[u+vmarg:d-vmarg,l:r] = 1
                else:
                    mask[u:d,l+hmarg:r-hmarg] = 1
        return mask, bboxes, bboxes_pct

    def __getitem__(self, index: int):
        id = self.ids[index]
        image, path = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return (image, path), target


class CocoBorderSegmentation(CocoDetection):

    def __init__(self, contract=0.2, vertical=False, category_id=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contract = contract
        self.vertical = vertical
        self.category_id = category_id

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB"), path

    def _load_target(self, id: int):
        anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        assert len(anns) > 0, f"No annotations for id {id}"
        h, w = self.coco.annToMask(anns[0]).shape
        mask = np.zeros((h,w))
        bboxes = []
        bboxes_pct = []
        for ann in anns:
            if ann['category_id'] != self.category_id:
                continue
            bbox = find_bbox(self.coco.annToMask(ann))
            if bbox is not None and abs(bbox[3]-bbox[2]) > 0 and abs(bbox[1]-bbox[0]) > 0:
                u, d, l, r = bbox
                bboxes.append([l,u,r,d])
                bboxes_pct.append([l/w,u/h,r/w,d/h])
                vmarg, hmarg = int(abs(u-d) * self.contract), int(abs(l-r) * self.contract)
                if self.vertical:
                    mask[u:u+vmarg,l:r] = 1
                    mask[d-vmarg:d,l:r] = 1
                else:
                    mask[u:d,l:l+hmarg] = 1
                    mask[u:d,r-hmarg:r] = 1
        return mask, bboxes, bboxes_pct

    def __getitem__(self, index: int):
        id = self.ids[index]
        image, path = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return (image, path), target


class TwoCatCocoSegmentation(CocoDetection):

    def __init__(self, 
            contract_0=0.2, 
            contract_1=0.0, 
            vertical=False, 
            category_id_0=0, 
            category_id_1=1, 
            *args, **kwargs
        ):

        super().__init__(*args, **kwargs)
        self.contract_0 = contract_0
        self.contract_1 = contract_1
        self.vertical = vertical
        self.category_id_0 = category_id_0
        self.category_id_1 = category_id_1

    def _load_image(self, id: int) -> Image.Image:

        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB"), path

    def _load_target(self, id: int):

        anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        assert len(anns) > 0, f"No annotations for id {id}"
        h, w = self.coco.annToMask(anns[0]).shape

        mask_0, mask_1 = np.zeros((h,w)), np.zeros((h,w))
        bboxes_0, bboxes_1 = [], []
        bboxes_pct_0, bboxes_pct_1 = [], []

        for ann in anns:

            _cat_id = ann['category_id']
            if not _cat_id in (self.category_id_0, self.category_id_1):
                continue

            bbox = find_bbox(self.coco.annToMask(ann))

            if bbox is not None and abs(bbox[3]-bbox[2]) > 0 and abs(bbox[1]-bbox[0]) > 0:

                u, d, l, r = bbox

                if _cat_id == self.category_id_0:
                    bboxes_0.append([l,u,r,d])
                    bboxes_pct_0.append([l/w,u/h,r/w,d/h])
                    vmarg, hmarg = int(abs(u-d) * self.contract_0), int(abs(l-r) * self.contract_0)
                    if self.vertical:
                        mask_0[u+vmarg:d-vmarg,l:r] = 1
                    else:
                        mask_0[u:d,l+hmarg:r-hmarg] = 1

                elif _cat_id == self.category_id_1:
                    bboxes_1.append([l,u,r,d])
                    bboxes_pct_1.append([l/w,u/h,r/w,d/h])
                    vmarg, hmarg = int(abs(u-d) * self.contract_1), int(abs(l-r) * self.contract_1)
                    if self.vertical:
                        mask_1[u+vmarg:d-vmarg,l:r] = 1
                    else:
                        mask_1[u:d,l+hmarg:r-hmarg] = 1

                else:
                    raise NotImplementedError

        return mask_0, mask_1, bboxes_0, bboxes_1, bboxes_pct_0, bboxes_pct_1

    def __getitem__(self, index: int):

        id = self.ids[index]
        image, path = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return (image, path), target


class LocalizerInferenceDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, x) for x in os.listdir(root_dir) if not x.startswith(".")]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(image_path)
