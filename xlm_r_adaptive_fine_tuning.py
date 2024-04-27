import os
import torch
import numpy as np
import evaluate
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import (
    XLMRobertaConfig, 
    XLMRobertaModelWithHeads,
    TrainingArguments,
    AdapterTrainer,
    AutoTokenizer,
    EarlyStoppingCallback,
)
from transformers.adapters import HoulsbyConfig
from datasets import load_dataset


# Read Swiss German dataset with error handling for unknown labels
def read_swissgerman_split(filename):
    texts = []
    labels = []
    with open(filename, "r") as file:
        for line in file:
            text, label = line.strip().split("\t")
            texts.append(text)
            if label in ["BS", "LU", "ZH", "BE"]:
                labels.append({"BS": 0, "LU": 1, "ZH": 2, "BE": 3}[label])
            else:
                raise ValueError(f"Unknown label '{label}' in dataset.")
    return texts, labels



class SwissGermanDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    results = {
        "precision": precision.compute(predictions=predictions, references=labels, average="macro")["precision"],
        "recall": recall.compute(predictions=predictions, references=labels, average="macro")["recall"],
        "f1": f1.compute(predictions=predictions, references=labels, average="macro")["f1"],
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
    }

    return results



def plot_confusion_matrix(predictions, references, class_names):
    cm = confusion_matrix(references, predictions)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()



def main():
    # Load datasets
    train_texts, train_labels = read_swissgerman_split("train.txt")
    val_texts, val_labels = read_swissgerman_split("dev.txt")
    test_texts, test_labels = read_swissgerman_split("gold.txt")

    # Tokenize datasets
    tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-base")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    # Create datasets
    train_dataset = SwissGermanDataset(train_encodings, train_labels)
    val_dataset = SwissGermanDataset(val_encodings, val_labels)
    test_dataset = SwissGermanDataset(test_encodings, test_labels)

    # Define models and adapter configurations to experiment with
    XLM_R_models = [
        "xlm-roberta-base",
        "XLM_R/MLM_archimob",
        "XLM_R/MLM_swiss_crawl_64k",
        "XLM_R/MLM_swiss_crawl_full_new",
    ]

    adapter_configs = [
        HoulsbyConfig(),
        HoulsbyConfig(reduction_factor=3),
        HoulsbyConfig(reduction_factor=96),
    ]

    # Loop through each model and adapter configuration
    for model_name in XLM_R_models:
        for adapter_config in adapter_configs:
            config = XLMRobertaConfig.from_pretrained(model_name, num_labels=4)
            model = XLMRobertaModelWithHeads.from_pretrained(model_name, config=config)

            # Add a unique adapter for each configuration
            adapter_name = f"swiss_base_{adapter_config.__class__.__name__}"
            model.add_adapter("swiss_xlm", config=adapter_config)
            model.add_classification_head(
                "swiss_dialects",
                num_labels=4,
                id2label={0: "BS", 1: "LU", 2: "ZH", 3: "BE"},
            )

            model.train_adapter("swiss_xlm")

            # Training arguments
            training_args = TrainingArguments(
                output_dir=f"./xlm_r_adapter/{model_name}_{adapter_name}",
                do_train=True,
                do_eval=True,
                num_train_epochs=8,
                learning_rate=1e-4,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=128,
                save_total_limit=1,
                evaluation_strategy="steps",
                eval_steps=300,
                save_steps=300,
                load_best_model_at_end=True,
                greater_is_better=False,
                metric_for_best_model="eval_loss",
                report_to="wandb",
            )

            # Train the model with AdapterTrainer
            trainer = AdapterTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_threshold=0.01, early_stopping_patience=8)],
            )

            # Train and evaluate
            trainer.train()

            # Evaluate on the test set
            model_predictions = trainer.predict(test_dataset).predictions
            test_results = compute_metrics((model_predictions, test_labels))
            print(f"Results for {model_name} with {adapter_name}: {test_results}")

            # Plot confusion matrix
            plot_confusion_matrix(np.argmax(model_predictions, axis=-1), test_labels, ["BS", "LU", "ZH", "BE"])


if __name__ == "__main__":
    main()
