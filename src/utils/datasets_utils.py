import os
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from torchvision import transforms as T
import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import albumentations as A
import kornia


BASE_TRANSFORM = T.Compose([
    T.ToTensor(), 
    T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])


GRAY_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Grayscale(num_output_channels=3),
    T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])


INV_NORMALIZE = T.Normalize(
   mean= [-m/s for m, s in zip(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)],
   std= [1/s for s in IMAGENET_DEFAULT_STD]
)


def random_erode_dilate(x):
    erode = np.random.choice([True, False])
    if erode:
        return kornia.morphology.dilation(x.unsqueeze(0), 
            kernel=torch.ones(np.random.choice([3,4]), np.random.choice([2,3]))).squeeze(0)
    else:
        return kornia.morphology.erosion(x.unsqueeze(0), 
            kernel=torch.ones(np.random.choice([3,4]), np.random.choice([2,3]))).squeeze(0)


def patch_resize(pil_img, patchsize=8, targetsize=224):

    w, h = pil_img.size
    larger_side = max([w, h])
    height_larger = larger_side == h 
    aspect_ratio = w / h if height_larger else h / w
    
    if height_larger:
        patch_resizer = T.Resize((targetsize, (int(aspect_ratio*targetsize) // patchsize) * patchsize))
    else:
        patch_resizer = T.Resize(((int(aspect_ratio*targetsize) // patchsize) * patchsize, targetsize))

    return patch_resizer(pil_img)


def color_shift(im):
    color = list(np.random.random(size=3))
    im[0, :, :][im[0, :, :] >= 0.8] = color[0]
    im[1, :, :][im[1, :, :] >= 0.8] = color[1]
    im[2, :, :][im[2, :, :] >= 0.8] = color[2]
    return im


def blur_transform(high):
    if high:
        return T.RandomApply([T.GaussianBlur(15, sigma=(1, 4))], p=0.3)
    else:
        return  T.RandomApply([T.GaussianBlur(11, sigma=(0.1, 2.0))], p=0.3)


class MedianPad:

    def __init__(self, override=None):

        self.override = override

    def __call__(self, image):


        image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        max_side = max(image.size)
        pad_x, pad_y = [max_side - s for s in image.size]
        padding = (0, 0, pad_x, pad_y)

        imgarray = np.array(image)
        h, w, c = imgarray.shape
        rightb, leftb = imgarray[:,w-1,:], imgarray[:,0,:]
        topb, bottomb = imgarray[0,:,:], imgarray[h-1,:,:]
        bordervals = np.concatenate([rightb, leftb, topb, bottomb], axis=0)
        medval = tuple([int(v) for v in np.median(bordervals, axis=0)])

        return T.Pad(padding, fill=medval if self.override is None else self.override)(image)


class AddAdjacentChars:
    def __init__(self, font, fontsize=224):
        self.fontsize = fontsize
        self.font = ImageFont.truetype(font, fontsize)
        self.chars = list("HOXELI代西岡光夫締西岡雪")
        
    def __call__(self, pil):
        n_sides = np.random.choice(range(5), size=1, replace=False, p=[0.7, 0.2, 0.05, 0.05, 0.0])[0]
        if n_sides == 0: return pil
        d = ImageDraw.Draw(pil)
        w, h = pil.size
        cx, cy = w // 2, h // 2
        offx = min(w, h) // 10
        offy = offx // 2
        sides = np.random.choice(list("lrtb"), size=n_sides, replace=False).tolist()
        for side in sides:
            if side == "l":
                d.text((offx, cy), 
                      np.random.choice(self.chars), font=self.font, anchor="rm", fill=(0, 0, 0))
            elif side == "r":
                d.text((w - offx, cy), 
                      np.random.choice(self.chars), font=self.font, anchor="lm", fill=(0, 0, 0))
            elif side == "b":
                d.text((cx, h - offy), 
                      np.random.choice(self.chars), font=self.font, anchor="mt", fill=(0, 0, 0))
            elif side == "t":
                d.text((cx, offy), 
                      np.random.choice(self.chars), font=self.font, anchor="mb", fill=(0, 0, 0))
        return pil


class AddAdjacentCharsEng:
    def __init__(self, font, fontsize=224):
        self.fontsize = fontsize
        self.font = ImageFont.truetype(font, fontsize)
        self.chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,")
        
    def __call__(self, pil):
        n_sides = np.random.choice(range(3), size=1, replace=False, p=[0.5, 0.25, 0.25])[0]
        if n_sides == 0: return pil
        d = ImageDraw.Draw(pil)
        w, h = pil.size
        offx = min(w, h) // 8
        sides = np.random.choice(list("lr"), size=n_sides, replace=False).tolist()
        for side in sides:
            if side == "l":
                d.text((0 + offx, h), 
                      np.random.choice(self.chars), font=self.font, anchor="rb", fill=(0, 0, 0))
            elif side == "r":
                d.text((w - offx, h), 
                      np.random.choice(self.chars), font=self.font, anchor="lb", fill=(0, 0, 0))
        return pil


def create_render_transform(lang, high_blur, size=224):
    return T.Compose([
        # AddAdjacentChars(font="./japan_font_files/NotoSerifCJKjp-Regular.otf") if lang=="jp" else lambda x: x,
        T.ToTensor(),
        T.RandomApply([T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=1)], p=0.7) if lang=="en" \
            else T.RandomApply([T.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1), fill=1)], p=0.7),
        T.RandomApply([color_shift], p=0.25),
        T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.3)], p=0.5),
        T.RandomApply([random_erode_dilate], p=0.5) if lang=="en" else lambda x: x,
        T.ToPILImage(),
        lambda x: Image.fromarray(A.GaussNoise(var_limit=(10.0, 150.0), mean=0, p=0.25)(image=np.array(x))["image"]),
        blur_transform(high_blur),
        T.RandomGrayscale(p=0.2),
        MedianPad(override=(255,255,255)),
        T.ToTensor(),
        T.Resize((size, size)),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])


def create_paired_transform(lang, size=224):
    return T.Compose([
        # SquarePad(),
        MedianPad(override=(255,255,255)),
        # T.Resize(size=(224,224)),
        # patch_resize,
        T.ToTensor(),
        T.Resize((size, size)),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        # featx_transform,
    ])


def create_inference_transform(lang, size=224):
    return T.Compose([
        MedianPad(override=(255,255,255)),
        T.Resize((size, size)),
    ])


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
