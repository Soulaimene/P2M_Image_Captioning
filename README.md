# P2M_Image_Captioning
This project aims to generate captions for images using a pre-trained image captioning model. The model used in this project is based on the VisionEncoderDecoderModel from the Hugging Face transformers library and specifically uses the ViT-GPT2 architecture.

##Features
- Utilizes the GPT-2 language model and ViT for image caption generation.
- Pretrained weights for GPT-2 and ViT are used to initialize the model.
- Integration with the COCO dataset for training and evaluation.
- Option to set the number of data for training.
- Evaluation metric includes the ROUGE score for caption quality assessment.
- Detailed examples and usage instructions are provided in the documentation.

## Model Deployment
The trained model has been deployed using Gradio and Hugging Face's model hosting service. You can access the deployed model by following this link. The model checkpoint and associated files can be found on the Hugging Face model repository.
