import wandb
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
import pandas as pd
import math
import random
import re
import numpy as np
from scipy.special import softmax
from sklearn.metrics import log_loss


def read_swissgerman_split(filename):
    texts = []
    labels = []
    for line in open(filename, "r"):
        texts.append(line.strip().split("\t")[0])
        if line.strip().split("\t")[1] == "BS":
            labels.append(0)
        elif line.strip().split("\t")[1] == "LU":
            labels.append(1)
        elif line.strip().split("\t")[1] == "ZH":
            labels.append(2)
        elif line.strip().split("\t")[1] == "BE":
            labels.append(3)
        else:
            print(line.strip().split("\t")[1])
    return texts, labels

def preprocess(data):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    processed = []
    for sent in data:
        sent = sent.lower()  # lower-casing
        sent = emoji_pattern.sub(r'', sent)  # remove emojis
        for i in sent:
            if i in punc:
                sent = sent.replace(i, "")  # remove punctuations
        processed.append(sent)
    return processed

def tokenize_function(examples):
    return tokenizer(examples["sentences"])

def loss_to_perplexity(loss):
    perplexity = []
    for l in loss:
        p = math.exp(l)
        perplexity.append(p)
    return perplexity

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probabilities = softmax(logits, axis=-1)
    loss = log_loss(labels.flatten(), probabilities.reshape(-1, probabilities.shape[-1]), labels=[i for i in range(logits.shape[-1])])
    perplexity = np.exp(loss)
    return {"perplexity": perplexity}

def train_model(model_name, train_dataset, test_dataset, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, padding=True, return_tensors='pt')
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-4,
        num_train_epochs=3,
        weight_decay=0.01,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=128,
        save_total_limit=1,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        greater_is_better=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model(output_dir)
    eval_results = trainer.evaluate()
    perplexity = math.exp(eval_results['eval_loss'])
    return perplexity

    def train_and_evaluate(models, model_type):
        for model_name, (corpus, output_dir) in models.items():
            corpus_dataset = Dataset.from_dict({"sentences": corpus})
            corpus_dataset = corpus_dataset.shuffle(seed=42)
            tokenized_datasets = corpus_dataset.map(tokenize_function, batched=True, remove_columns=["sentences"])
            tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.2)
            
            perplexity = train_model(model_type, tokenized_datasets["train"], tokenized_datasets["test"], output_dir)
            
            print(f"Perplexity for {model_name} with {model_type}: {perplexity:.2f}")

def main():
    train_texts, train_labels = read_swissgerman_split("train.txt")
    val_texts, val_labels = read_swissgerman_split("dev.txt")
    test_texts, test_labels = read_swissgerman_split("gold.txt")
    whole_corpus = []
    for line in open("archimob_corpus.txt", "r"):
        whole_corpus.append(line.strip())
    whole_corpus = [sent for sent in whole_corpus if sent not in train_texts and sent not in val_texts and sent not in test_texts]
    processed_SwissCrawl = preprocess(swiss_crawl_dataset)


    gbert_models = {
        "GBert/MLM_archimob_subset": (whole_corpus, "GBert/MLM_archimob_subset"),
        "GBert/MLM_swiss_crawl_64k": (random.sample(processed_SwissCrawl, 64000), "GBert/MLM_swiss_crawl_64k"),
        "GBert/MLM_swiss_crawl_full_new": (processed_SwissCrawl, "GBert/MLM_swiss_crawl_full_new")}
    xlm_r_models={
        "XLM_R/MLM_archimob": (whole_corpus, "XLM_R/MLM_archimob"),
        "XLM_R/MLM_swiss_crawl_64k": (random.sample(processed_SwissCrawl, 64000), "XLM_R/MLM_swiss_crawl_64k"),
        "XLM_R/MLM_swiss_crawl_full_new": (processed_SwissCrawl, XLM_R/MLM_swiss_crawl_full_new")
    }]
    train_and_evaluate(gbert_models, "deepset/gbert-base")
    train_and_evaluate(xlm_r_models, "xlm-roberta-base")


if __name__ == "__main__":
    main()
