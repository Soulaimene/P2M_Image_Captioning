from pycocotools.coco import COCO
import argparse
import os
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, ViTFeatureExtractor
import data_setup
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('--image_folder', type=str, default="/content/train2017",
                      help="specify the folder containing the images you want to visualize")
parser.add_argument('--annotation_folder', type=str, default="/content/annotations/captions_train2017.json",
                      help="specify the folder containing the annotation files of the images")
parser.add_argument('--id', type=int, default=1,
                      help="the id of the image you want to visualize")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.unk_token
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
NumData= args.id + 1
data_fixed = data_setup.Create_DATA(args.annotation_folder)
ds_loader = data_setup.P2MDataset(img_folder= args.image_folder, annotationFile=args.annotation_folder ,data_coco=data_fixed, feature_extractor= feature_extractor, tokenizer=tokenizer, NumData=NumData)


coco=COCO(args.annotation_folder)

ids = list(data_fixed.keys())
item=ds_loader[args.id]
ann_id = ids[args.id] 
img_id = data_fixed[ann_id]['image_id']
path = coco.loadImgs(img_id)[0]['file_name']
image = Image.open(os.path.join(args.image_folder, path)).convert('RGB')

labels=item['labels']
labels[labels == -100] = tokenizer.pad_token_id
label_str = tokenizer.decode(labels, skip_special_tokens=True)
print(label_str)




plt.axis("off")
image.show()
