import argparse
from transformers import Seq2SeqTrainer ,Seq2SeqTrainingArguments, default_data_collator, ViTFeatureExtractor, AutoTokenizer
import model_build, rouge2, data_setup


parser = argparse.ArgumentParser()

parser.add_argument('--decoder', type=str, default="gpt2",
                      help="specify the decoder model")

parser.add_argument('--encoder', type=str, default="google/vit-base-patch16-224",
                      help="specify the encoder model")

parser.add_argument('--train_images_folder', type=str, default="/content/train2017",
                      help="specify the train images folder")

parser.add_argument('--train_annotations_folder', type=str, default="/content/annotations/captions_train2017.json",
                      help=" specify the train anotation folder")

parser.add_argument('--val_images_folder', type=str, default="/content/val2017",
                      help="specify the validation images folder")

parser.add_argument('--val_annotations_folder', type=str, default="/content/annotations/captions_val2017.json",
                      help=" specify the validation anotation folder")

parser.add_argument('--number_train_data', type=int, default=8000,
                      help=" specify the train data amount - default is 8000")

parser.add_argument('--number_val_data', type=int, default=3000,
                      help=" specify the val data amount - default is 3000")

parser.add_argument('--output_folder', type=str, default="model8k3k",
                      help=" specify the model's output folder name")

parser.add_argument('--bs', type=int, default=12,
                      help="specify the batch size")
parser.add_argument('--lr', type=float, default=5e-5,
                      help="specify the learning rate")

parser.add_argument('--epoch', type=int, default=12,
                      help="specify the number of epochs")


args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained(args.decoder)
tokenizer.pad_token = tokenizer.unk_token
feature_extractor = ViTFeatureExtractor.from_pretrained(args.encoder)

#coco = COCO("/content/annotations/captions_train2017.json")
img_folder=args.train_images_folder
annotationFile=args.train_annotations_folder
img_dir_val= args.val_images_folder
annotation_val= args.val_annotations_folder


data_train=data_setup.Create_DATA(annotationFile)
data_val=data_setup.Create_DATA(annotation_val)
train_ds = data_setup.P2MDataset(img_folder= img_folder, annotationFile=annotationFile ,data_coco=data_train, feature_extractor= feature_extractor, tokenizer=tokenizer, NumData=args.number_train_data)
val_ds   = data_setup.P2MDataset(img_dir_val,annotationFile=annotation_val ,data_coco=data_val, feature_extractor= feature_extractor ,tokenizer= tokenizer,NumData=args.number_val_data)


training_args = Seq2SeqTrainingArguments(
    output_dir=args.output_folder,
    per_device_train_batch_size=args.bs,
    per_device_eval_batch_size=args.bs,
    predict_with_generate=True,
    evaluation_strategy="epoch",
    do_train=True,
    do_eval=True,
    logging_steps=1024,  
    save_steps=2048, 
    warmup_steps=1024,  
    learning_rate = args.lr,
    #max_steps=1500, # delete for full training
    num_train_epochs = args.epoch, #TRAIN_EPOCHS
    overwrite_output_dir=True,
    save_total_limit=1)


trainer = Seq2SeqTrainer(
        tokenizer=feature_extractor,
        model=model_build.define_model(args.encoder, args.decoder),
        args=training_args,
        compute_metrics=rouge2.compute_metrics,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator)

trainer.train()
trainer.save_model(args.output_folder)
