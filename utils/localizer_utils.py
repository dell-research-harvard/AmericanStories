import numpy as np
from math import exp
import cv2
import os
import torchvision
import torch
import torchvision.transforms as T

from utils.datasets_utils import INV_NORMALIZE


def find_bbox(img):
    a = np.where(img != 0)
    try:
        bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
        return bbox
    except ValueError:
        return None


def create_gaussian_2d(h, w):
    scaledGaussian = lambda x : exp(-(1/2)*(x**2))
    isotropicGrayscaleImage = np.zeros((h,w))
    center = np.array([(h-1)/2, (w-1)/2])
    denom = min(h/2, w/2)
    for i in range(h):
        for j in range(w):
            distanceFromCenter = 1.5*np.linalg.norm(np.array([i, j]) - center)/denom
            isotropicGrayscaleImage[i,j] = np.clip(scaledGaussian(distanceFromCenter)*1, 0, 1)
    return isotropicGrayscaleImage


def create_gaussian_1d(h, w):
    scaledGaussian = lambda x : exp(-(1/2)*(x**2))
    isotropicGrayscaleImage = np.zeros((h,w))
    center = (w-1)/2
    denom = min(h/2, w/2)
    for j in range(w):
        distanceFromCenter = 1.5*abs(j-center)/denom
        for i in range(h):
            isotropicGrayscaleImage[i,j] = np.clip(scaledGaussian(distanceFromCenter), 0, 1)
    return isotropicGrayscaleImage


def perspective_warp_bbox(src, ow, oh):
    ih, iw = src.shape
    input = np.float32([[0,0], [iw-1,0], [iw-1,ih-1], [0,ih-1]])
    output = np.float32([[0,0], [ow-1,0], [ow-1,oh-1], [0,oh-1]])
    matrix = cv2.getPerspectiveTransform(input, output)
    imgOutput = cv2.warpPerspective(src, matrix, (ow,oh), cv2.INTER_LINEAR)
    return imgOutput


def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


def save_img_twohead(input, output_0, output_1, bboxes_0, bboxes_1, savedir, idx, thr=0.5):
    image = INV_NORMALIZE(input.squeeze(0)).cpu()
    mimg = torch.tensor(image*255, dtype=torch.uint8)
    mask_0 = output_0.sigmoid() > thr
    mask_1 = output_1.sigmoid() > thr
    masked_image = torchvision.utils.draw_segmentation_masks(mimg, mask_1, alpha=0.25, colors=["green"])
    masked_image = torchvision.utils.draw_segmentation_masks(masked_image, mask_0, alpha=0.5, colors=["blue"])
    boxed_image = torchvision.utils.draw_bounding_boxes(masked_image, torch.tensor(bboxes_0), colors="red")
    boxed_image = torchvision.utils.draw_bounding_boxes(boxed_image, torch.tensor(bboxes_1), colors="purple")
    torchvision.utils.save_image(T.ToTensor()(T.ToPILImage()(boxed_image)), os.path.join(savedir, f"{idx}_trainseg.png"))


def save_img(input, output, bboxes, savedir, idx, thr=0.5):
    image = INV_NORMALIZE(input.squeeze(0)).cpu()
    mimg = torch.tensor(image*255, dtype=torch.uint8)
    mask = output.sigmoid() > thr
    masked_image = torchvision.utils.draw_segmentation_masks(mimg, mask, alpha=0.5, colors=["blue"])
    boxed_image = torchvision.utils.draw_bounding_boxes(masked_image, torch.tensor(bboxes), colors="red")
    torchvision.utils.save_image(T.ToTensor()(T.ToPILImage()(boxed_image)), os.path.join(savedir, f"{idx}_trainseg.png"))


def save_img_bbox(input, bboxes, savedir, idx, thr=0.5):
    image = INV_NORMALIZE(input.squeeze(0)).cpu() # image = T.ToTensor()(input).cpu()
    mimg = torch.tensor(image*255, dtype=torch.uint8)
    boxed_image = torchvision.utils.draw_bounding_boxes(mimg, torch.tensor(bboxes), colors="red")
    torchvision.utils.save_image(T.ToTensor()(T.ToPILImage()(boxed_image)), os.path.join(savedir, f"{idx}_trainseg.png"))


def save_heat(input, output, savedir, idx):
    input = INV_NORMALIZE(input.squeeze(0)).permute(1,2,0).cpu().numpy()
    input = convert(input, 0, 255, np.uint8)
    output = output.permute(1,2,0).cpu().numpy()
    output = convert(output, 0, 255, np.uint8)
    output = cv2.applyColorMap(output, cv2.COLORMAP_JET)
    fin = cv2.addWeighted(output, 0.5, input, 0.5, 0)
    cv2.imwrite(os.path.join(savedir, f"{idx}_trainseg.png"), fin)


def bboxes_to_preds(bboxes):
    preds = [
        dict(
            boxes=torch.Tensor(bboxes),
            scores=torch.Tensor([1]*len(bboxes)),
            labels=torch.IntTensor([0]*len(bboxes)),
        )
    ]
    return preds


def box_area(arr):
    # arr: np.array([[x1, y1, x2, y2]])
    width = arr[:, 2] - arr[:, 0]
    height = arr[:, 3] - arr[:, 1]
    return width * height


def _box_inter_union(arr1, arr2):
    # arr1 of [N, 4]
    # arr2 of [N, 4]
    area1 = box_area(arr1)
    area2 = box_area(arr2)

    # Intersection
    top_left = np.maximum(arr1[:, :2], arr2[:, :2]) # [[x, y]]
    bottom_right = np.minimum(arr1[:, 2:], arr2[:, 2:]) # [[x, y]]
    wh = bottom_right - top_left
    # clip: if boxes not overlap then make it zero
    intersection = wh[:, 0].clip(0) * wh[:, 1].clip(0)

    #union 
    union = area1 + area2 - intersection
    return intersection, union


def _box_inter_min(arr1, arr2):
    # arr1 of [N, 4]
    # arr2 of [N, 4]
    area1 = box_area(arr1)
    area2 = box_area(arr2)

    # Intersection
    top_left = np.maximum(arr1[:, :2], arr2[:, :2]) # [[x, y]]
    bottom_right = np.minimum(arr1[:, 2:], arr2[:, 2:]) # [[x, y]]
    wh = bottom_right - top_left
    # clip: if boxes not overlap then make it zero
    intersection = wh[:, 0].clip(0) * wh[:, 1].clip(0)

    #union 
    mini = min(area1, area2)
    return intersection, mini


def box_iou(arr1, arr2):
    # arr1[N, 4]
    # arr2[N, 4]
    # N = number of bounding boxes
    assert(arr1[:, 2:] > arr1[:, :2]).all()
    assert(arr2[:, 2:] > arr2[:, :2]).all()
    inter, union = _box_inter_union(arr1, arr2)
    iou = inter / union
    print(iou)


def box_iom(arr1, arr2):
    # arr1[N, 4]
    # arr2[N, 4]
    # N = number of bounding boxes
    assert(arr1[:, 2:] > arr1[:, :2]).all()
    assert(arr2[:, 2:] > arr2[:, :2]).all()
    inter, mini = _box_inter_min(arr1, arr2)
    iom = inter / mini
    return iom