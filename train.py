from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate
import numpy as np
import nltk
nltk.download("punkt")


# 1. Choose model & dataset

model_name = "facebook/bart-base"  # or "t5-base", "google/pegasus-xsum"
dataset_name = "cnn_dailymail"     # or "xsum"
dataset_config = "3.0.0"           # required for cnn_dailymail


# 2. Load dataset

raw_datasets = load_dataset(dataset_name, dataset_config)


# 3. Load tokenizer & model

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

max_input_length = 512
max_target_length = 128

def preprocess_function(examples):
    if dataset_name == "cnn_dailymail":
        inputs = examples["article"]
        targets = examples["highlights"]
    else:  # xsum
        inputs = examples["document"]
        targets = examples["summary"]

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names)


# 4. Evaluation (ROUGE & BLEU)

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    bleu_result = bleu.compute(predictions=[p.split() for p in decoded_preds],
                               references=[[r.split()] for r in decoded_labels])

    result = {**rouge_result, "bleu": bleu_result["bleu"]}
    return {k: round(v, 4) for k, v in result.items()}


# 5. Training setup

args = Seq2SeqTrainingArguments(
    output_dir="./results",
    do_eval=True,
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=1,  # Can increase later
    predict_with_generate=True,
    fp16=True
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"].select(range(1000)),  # train on smaller 1st
    eval_dataset=tokenized_datasets["validation"].select(range(200)),
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# 6. Train & Save

trainer.train()
trainer.save_model("./summarizer_model")
tokenizer.save_pretrained("./summarizer_model")
