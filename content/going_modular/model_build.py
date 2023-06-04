from transformers import VisionEncoderDecoderModel
from transformers import AutoTokenizer
def define_model(encoder ,decoder):
  tokenizer = AutoTokenizer.from_pretrained(decoder)  
  tokenizer.pad_token = tokenizer.unk_token

  model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder, decoder)

  model.config.decoder_start_token_id = tokenizer.cls_token_id
  model.config.pad_token_id = tokenizer.pad_token_id

  # make sure vocab size is set correctly

  model.config.vocab_size = model.config.decoder.vocab_size

  # set beam search parameters

  model.config.eos_token_id = tokenizer.sep_token_id
  model.config.decoder_start_token_id = tokenizer.bos_token_id
  model.config.max_length = 20
  model.config.no_repeat_ngram_size = 3
  model.config.length_penalty = 2.0
  model.config.num_beams = 4

  return model
