import nlp
import logging
from transformers import BertTokenizer, EncoderDecoderModel, Trainer, TrainingArguments
import torch
from transformers.tokenization_bert_japanese import BertJapaneseTokenizer
from transformers.modeling_bert import BertForMaskedLM
import csv
import pandas as pd
from datasets import load_dataset


logging.basicConfig(level=logging.INFO)

bert_jp_model = 'cl-tohoku/bert-base-japanese-whole-word-masking'
mecab_opts = {"mecab_option": "-r /dev/null -d /usr/local/lib/mecab/dic/ipadic"}

tokenizer = BertJapaneseTokenizer.from_pretrained(bert_jp_model, mecab_kwargs=mecab_opts)
model = EncoderDecoderModel.from_encoder_decoder_pretrained(bert_jp_model, bert_jp_model)

# CLS token will work as BOS token
tokenizer.bos_token = tokenizer.cls_token

# SEP token will work as EOS token
tokenizer.eos_token = tokenizer.sep_token

train_dataset = nlp.load_dataset('csv', data_files='/home/ats432/projects/Matsuzaki_Lab/Transformers_EncoderDecoder_Sum/notebook/dataset.csv', split = 'train[1000:]')
val_dataset = nlp.load_dataset('csv', data_files='/home/ats432/projects/Matsuzaki_Lab/Transformers_EncoderDecoder_Sum/notebook/dataset.csv', split = 'train[:1000]')

# load rouge for validation
rouge = nlp.load_metric("rouge")

# set decoding params
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.max_length = 142
model.config.min_length = 56
model.config.no_repeat_ngram_size = 3
model.early_stopping = True
model.length_penalty = 2.0
model.num_beams = 4

# map data correctly
def map_to_encoder_decoder_inputs(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=512)
    # force summarization <= 128
    outputs = tokenizer(batch["highlights"], padding="max_length", truncation=True, max_length=128)
    
    
    batch["input_ids"] = inputs.input_ids # inputsのID
    batch["attention_mask"] = inputs.attention_mask # 　encoderの重要部分を測る

    batch["decoder_input_ids"] = outputs.input_ids # outputsのID
    batch["labels"] = outputs.input_ids.copy() # outputsのIDをコピーしラベルとして使用
    # mask loss for padding
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
    ] # attentionの値を際立たせてる？　＜＝　聞こう
    
    batch["decoder_attention_mask"] = outputs.attention_mask # decoderの重要部分を測る

    assert all([len(x) == 512 for x in inputs.input_ids])  # "assert 条件式, 条件式がFalseの場合に出力するメッセージ"
    assert all([len(x) == 128 for x in outputs.input_ids])  # "assert 条件式, 条件式がFalseの場合に出力するメッセージ"

    return batch


def compute_metrics(pred):
    
    labels_ids = pred.label_ids # 参照データのID
    # pred_ids = pred.predictions #  予測結果のID
    pred_ids = pred.predictions.argmax(-1)

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True) #予測結果の不要トークンの削除
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True) #参照データの不要トークンの削除

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid # ←具体的には何やってるかわからん

    return {
        "rouge2_precision": round(rouge_output.precision, 4), #精度
        "rouge2_recall": round(rouge_output.recall, 4),              #再現性
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),#F値
    }


# set batch size here
batch_size = 16

# make train dataset ready
train_dataset = train_dataset.map(
    map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["highlights", "article"],
)
train_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

# same for validation dataset
val_dataset = val_dataset.map(
    map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["article", "highlights"],
)
val_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)


# set training arguments - these params are not really tuned, feel free to change
training_args = TrainingArguments(
    output_dir="./",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    # predict_from_generate=True,
    evaluate_during_training=True,
    do_train=True,
    do_eval=True,
    logging_steps=1000,
    save_steps=1000,
    overwrite_output_dir=True,
    warmup_steps=2000,
    save_total_limit=10,
)


# instantiate trainer
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)


# start training
trainer.train()

model.save_pretrained("bert2bert")

model.from_pretrained("bert2bert")

