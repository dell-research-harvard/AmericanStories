import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
import numpy as np
import os
import math
import json
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from datasets.recognizer_samplers import *
from utils.datasets_utils import *


def diff_size_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


class CustomSubset(Dataset):

    def __init__(self, dataset, indices):
        self.super_dataset = dataset
        self.indices = indices
        self.class_to_idx = dataset.class_to_idx
        self.data = [x for idx, x in enumerate(dataset.data) if idx in indices]
        self.targets = [x for idx, x in enumerate(dataset.targets) if idx in indices]

    def __getitem__(self, idx):
        image = self.super_dataset[self.indices[idx]][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.indices)


class FontImageFolder(ImageFolder):

    def __init__(self, root, render_transform=None, paired_transform=None, patch_resize=False,
                 loader=default_loader, is_valid_file=None):

        super(ImageFolder, self).__init__(root, loader, 
            IMG_EXTENSIONS if is_valid_file is None else None, is_valid_file=is_valid_file)
        self.data = self.samples
        self.render_transform = render_transform
        self.paired_transform = paired_transform
        self.patch_resize = patch_resize

    def __getitem__(self, index):

        path, target = self.data[index]
        sample = self.loader(path)
        
        if os.path.basename(path).startswith("PAIRED"):
            sample = self.paired_transform(sample)
        else:
            sample = self.render_transform(sample)

        return sample, target


def create_dataset(
        root_dir, 
        train_ann_path,
        val_ann_path,
        test_ann_path, 
        batch_size,
        hardmined_txt=None, 
        m=4,
        finetune=False,
        pretrain=False,
        high_blur=False,
        lang="jp",
        knn=True,
        diff_sizes=False,
        imsize=224,
        num_passes=1
    ):

    if finetune and pretrain:
        raise NotImplementedError
    if finetune:
        print("Finetuning mode!")
    if pretrain:
        print("Pretraining model!")

    dataset = FontImageFolder(
        root_dir, 
        render_transform=create_render_transform(lang, high_blur, size=imsize), 
        paired_transform=create_paired_transform(lang, size=imsize),
        patch_resize=diff_sizes
    )

    with open(train_ann_path) as f: 
        train_ann = json.load(f)
        train_stems = [os.path.splitext(x['file_name'])[0] for x in train_ann['images']]
    with open(val_ann_path) as f: 
        val_ann = json.load(f)
        val_stems = [os.path.splitext(x['file_name'])[0] for x in val_ann['images']]
    with open(test_ann_path) as f: 
        test_ann = json.load(f)
        test_stems = [os.path.splitext(x['file_name'])[0] for x in test_ann['images']]

    assert len(set(test_stems).intersection(set(train_stems))) == 0
    assert len(set(val_stems).intersection(set(train_stems))) == 0
    if test_ann_path != val_ann_path:
        assert len(set(val_stems).intersection(set(test_stems))) == 0
    print(f"textline train len: {len(train_stems)}\ntextline val len: {len(val_stems)}\ntextline test len: {len(test_stems)}")
    
    paired_train_idx = [idx for idx, (p, t) in enumerate(dataset.data) if \
        any(os.path.basename(p).startswith(f'PAIRED_{imf}_') for imf in train_stems)]
    paired_val_idx = [idx for idx, (p, t) in enumerate(dataset.data) if \
        any(os.path.basename(p).startswith(f'PAIRED_{imf}_') for imf in val_stems)]
    paired_test_idx = [idx for idx, (p, t) in enumerate(dataset.data) if \
        any(os.path.basename(p).startswith(f'PAIRED_{imf}_') for imf in test_stems)]
    render_idx = [idx for idx, (p, t) in enumerate(dataset.data) if \
        not os.path.basename(p).startswith("PAIRED")]

    assert len(set(paired_train_idx).intersection(set(paired_val_idx))) == 0
    if test_ann_path != val_ann_path:
        assert len(set(paired_val_idx).intersection(set(paired_test_idx))) == 0
    assert len(set(paired_test_idx).intersection(set(paired_train_idx))) == 0 
    print(f"train len: {len(paired_train_idx)}\nval len: {len(paired_val_idx)}\ntest len: {len(paired_test_idx)}")
    
    if finetune:
        idx_train = sorted(paired_train_idx)
    elif pretrain:
        idx_train = sorted(render_idx)
    else:
        idx_train = sorted(render_idx + paired_train_idx)
    idx_val = sorted(paired_val_idx)
    idx_test = sorted(paired_test_idx)

    if finetune:
        assert len(idx_train) + len(idx_val) + len(render_idx) + len(idx_test) == \
            len(dataset), f"{len(idx_train)} + {len(idx_val)} + {len(idx_test)} != {len(dataset)}"
    elif pretrain:
        assert len(idx_train) + len(idx_val) + len(paired_train_idx) + len(idx_test) == \
            len(dataset), f"{len(idx_train)} + {len(idx_val)} + {len(idx_test)} != {len(dataset)}"
    else:
        if test_ann_path != val_ann_path:
            assert len(idx_train) + len(idx_val) + len(idx_test) == \
                len(dataset), f"{len(idx_train)} + {len(idx_val)} + {len(idx_test)} != {len(dataset)}"
        else:
            assert len(idx_train) + len(idx_val) == \
                len(dataset), f"{len(idx_train)} + {len(idx_val)} != {len(dataset)}"        

    train_dataset = CustomSubset(dataset, idx_train)
    val_dataset = CustomSubset(dataset, idx_val)
    test_dataset = CustomSubset(dataset, idx_test)
    print(f"Len train dataset: {len(train_dataset)}")
    print(f"Len val dataset: {len(val_dataset)}")
    print(f"Len test dataset: {len(test_dataset)}")

    if hardmined_txt is None:
        train_sampler = NoReplacementMPerClassSampler(
            train_dataset, m=m, batch_size=batch_size, num_passes=num_passes
        )
    else:
        with open(hardmined_txt) as f:
            hard_negatives = f.read().split()
            print(f"Len hard negatives: {len(hard_negatives)}")
        train_sampler = HardNegativeClassSampler(train_dataset, 
            train_dataset.class_to_idx, hard_negatives, m=m, batch_size=batch_size, 
            num_passes=num_passes
        )

    if knn:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False,
            num_workers=32, pin_memory=True, drop_last=True, 
            sampler=train_sampler, collate_fn=diff_size_collate if diff_sizes else None)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=32, pin_memory=True, drop_last=False,
            collate_fn=diff_size_collate if diff_sizes else None)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=32, pin_memory=True, drop_last=False,
            collate_fn=diff_size_collate if diff_sizes else None)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=32, pin_memory=True, drop_last=True, 
            collate_fn=diff_size_collate if diff_sizes else None)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True,
            num_workers=32, pin_memory=True, drop_last=False, 
            collate_fn=diff_size_collate if diff_sizes else None)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True,
            num_workers=32, pin_memory=True, drop_last=False, 
            collate_fn=diff_size_collate if diff_sizes else None)
    
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def create_paired_dataset(root_dir, lang, imsize=224):

    paired_transform = render_transform = create_paired_transform(lang, imsize)

    dataset = FontImageFolder(root_dir, render_transform=render_transform, paired_transform=paired_transform)
    idx_paired = [idx for idx, (p, t) in enumerate(dataset.data) if os.path.basename(p).startswith("PAIRED")]

    paired_dataset = CustomSubset(dataset, idx_paired)
    print(f"Len paired dataset: {len(paired_dataset)}")
    
    return paired_dataset


def create_render_dataset(root_dir, lang, imsize=224, font_name=""):

    paired_transform = render_transform = create_paired_transform(lang, imsize)

    dataset = FontImageFolder(root_dir, render_transform=render_transform, paired_transform=paired_transform)
    idx_render = [idx for idx, (p, t) in enumerate(dataset.data) if font_name in p and not os.path.basename(p).startswith("PAIRED")]

    render_dataset = CustomSubset(dataset, idx_render)
    print(f"Len render dataset: {len(render_dataset)}")
    
    return render_dataset
