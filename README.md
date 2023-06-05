# P2M_Image_Captioning
This project aims to generate captions for images using a pre-trained image captioning model. The model used in this project is based on the VisionEncoderDecoderModel from the Hugging Face transformers library and specifically uses the ViT-GPT2 architecture.

## Features
- Utilizes the GPT-2 language model and ViT for image caption generation.
- Pretrained weights for GPT-2 and ViT are used to initialize the model.
- Integration with the COCO dataset for training and evaluation.
- Option to set the number of data for training.
- Evaluation metric includes the ROUGE score for caption quality assessment.
- Detailed examples and usage instructions are provided in the documentation.

## Model Deployment
The trained model has been deployed using Gradio and Hugging Face's model hosting service. You can access the deployed model by following this link. The model checkpoint and associated files can be found on the Hugging Face model repository.

## Requirements
Python 3.7 or higher
PyTorch
Transformers library
torchvision
pycocotools

## Installation
- Clone the repository:

```bash
Copy code
git clone [https://github.com/your-username/image-captioning.git](https://github.com/Soulaimene/P2M_Image_Captioning.git)
```

- Install the required dependencies:

```bash
Copy code
pip install -r content/going_modular/requirements.txt
```

## Usage
1. Prepare the COCO dataset for training and evaluation. The model uses the COCO dataset for training, which provides image-caption pairs for a wide range of images. The dataset can be downloaded from official-coco-website:
- If you are on Linux :
```bash
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip
```
-If you are on Windows :
```bash
curl -O http://images.cocodataset.org/zips/train2017.zip
curl -O http://images.cocodataset.org/zips/val2017.zip
curl -O http://images.cocodataset.org/zips/test2017.zip
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
curl -O http://images.cocodataset.org/annotations/image_info_test2017.zip
```
2. To train the image captioning model on a specific number of data points, you can modify the training script or command to include the desired parameters. Here's an example of how you can specify the number of data points for training and validation:
```bash
python training5.py --number_train_data 5000 --number_val_data 3000
```
* You can set the image folder and the annotation folder paths:
```bash
python training5.py --number_train_data 5000 --number_val_data 3000
                    --train_images_folder "insert_image_folder"
                    --train_annotations_folder "insert_annotation_path"
                    --val_images_folder "insert_image_validation_folder"
                    --val_annotations_folder "insert_annotation_validation_path"
```
* You can custiomize your batch_size, learning rate and the number of epochs just by :
```bash
python training5.py --bs 24 --lr 5e-5  --epoch 12
```
3. To visualize the training dataset and it's captions:
```bash
python visualize.py --id 50 --image_folder "insert img_folder here" --annotation_folder "insert_annotation_here"
```
