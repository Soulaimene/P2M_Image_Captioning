import os
import random
import pathlib
from PIL import Image
from typing import Tuple, Dict, List
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision import datasets
from pathlib import Path
from pycocotools.coco import COCO
random.seed(0)


def Create_DATA(annotationFile):
    keys_to_remove = []
    coco = COCO(annotationFile)
    img_idx=None
    for i in  ((coco.anns.keys())):
        height=coco.loadImgs(coco.anns[i]['image_id'])[0]['height']
        width=coco.loadImgs(coco.anns[i]['image_id'])[0]['width']

        if  (min(width,height) < 224) :
            keys_to_remove.append(i) 

    for key in keys_to_remove:
        if key in coco.anns:
            del coco.anns[key]
    return coco.anns



class P2MDataset(Dataset):
    def __init__(self,img_folder,annotationFile,data_coco,feature_extractor,tokenizer, NumData, max_len=75, mode='train'):
        self.crop_size=224
        self.annotatioFile = annotationFile
        self.max_len=max_len
        self.feature_extractor=feature_extractor
        self.tokenizer=tokenizer
        self.mode=mode
        self.coco= COCO(annotationFile)
        self.data = data_coco #still missing for the testing , need to add it 
        self.img_folder = img_folder

        self.ids = list(self.data.keys())[:NumData]

        if self.mode == "train":
            self.transform = transforms.Compose([transforms.Resize((224,224)),
                              #transforms.RandomHorizontalFlip(),
                              #transforms.RandomCrop(224),
                              #transforms.ToTensor(),
                             # transforms.Normalize(0.5, 0.5),
                              ])
            
            
        elif self.mode == 'test':
            self.transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(self.feature_extractor.image_mean, self.feature_extractor.image_std)])

    def __len__(self):
        if self.mode == 'train':
            return len(self.ids)

    def __getitem__(self, idx):            
        ann_id = self.ids[idx] 
        img_id = self.data[ann_id]['image_id']

        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')

        image = self.transform(image)
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values

        caption = self.data[ann_id]['caption']
        labels = self.tokenizer(caption, padding='max_length', max_length=self.max_len, truncation=True).input_ids
        labels = [label if label != self.tokenizer.pad_token_id else -100 for label in labels]

        return {"pixel_values": pixel_values.squeeze(),
                "labels": torch.tensor(labels)}
