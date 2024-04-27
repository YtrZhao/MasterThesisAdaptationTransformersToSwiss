import torch
import evaluate
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    AutoTokenizer
)
import matplotlib.pyplot as plt


def read_swissgerman_split(filename):
    texts = []
    labels = []
    with open(filename, "r") as file:
        for line in file:
            text, label = line.strip().split("\t")
            texts.append(text)
            labels.append(
                {
                    "BS": 0,
                    "LU": 1,
                    "ZH": 2,
                    "BE": 3,
                }.get(label, -1)
            )
    return texts, labels

# Swiss German dataset class
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

# Define metrics computation
def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision = precision_metric.compute(predictions=predictions, references=labels, average="macro")["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels, average="macro")["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

# Define model initialization
def model_init():
    return AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=4)

# Define hyperparameter space
def my_hp_space(trial):
    return {
        "learning_rate": trial.suggest_categorical("learning_rate", [2e-4, 3e-4, 5e-4]),
        "weight_decay": trial.suggest_categorical("weight_decay", [1e-4, 1e-3, 1e-2]),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
    }

def my_objective(metrics):
    return metrics["eval_loss"]

def main():
    # Load datasets
    train_texts, train_labels = read_swissgerman_split("train.txt")
    val_texts, val_labels = read_swissgerman_split("dev.txt")
    test_texts, test_labels = read_swissgerman_split("gold.txt")

    # Tokenize datasets
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = SwissGermanDataset(train_encodings, train_labels)
    val_dataset = SwissGermanDataset(val_encodings, val_labels)
    test_dataset = SwissGermanDataset(test_encodings, test_labels)

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir="./xlm_r",
        do_train=True,
        do_eval=True,
        num_train_epochs=3,
        evaluation_strategy="steps",
        eval_steps=300,
        save_steps=300,
        save_total_limit=1,
        per_device_eval_batch_size=64,
        load_best_model_at_end=True,
        greater_is_better=False,
        metric_for_best_model="eval_loss",
        report_to="wandb",
    )

    # Define trainer
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_threshold=0.01, early_stopping_patience=8)],
    )

    # Run hyperparameter search
    best_run = trainer.hyperparameter_search(
        direction="minimize",
        backend="optuna",
        compute_objective=my_objective,
        n_trials=10,
        hp_space=my_hp_space,
    )

    model_dir=["xlm-roberta-base","XLM_R/MLM_archimob","XLM_R/MLM_swiss_crawl_64k","XLM_R/MLM_swiss_crawl_full_new"]
    for i in range(4):
        # Train model with best hyperparameters
        model=AutoModelForSequenceClassification.from_pretrained(model_dir[i], num_labels=4)
        training_args = TrainingArguments(
        output_dir=f"{model_dir[i]}_fine_tuned",
        do_train=True,
        do_eval=True,
        num_train_epochs=3,
        evaluation_strategy="steps",
        eval_steps=300,
        save_steps=300,
        save_total_limit=1,
        per_device_eval_batch_size=64,
        load_best_model_at_end=True,
        greater_is_better=False,
        metric_for_best_model="eval_loss",
        report_to="wandb",
    )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_threshold=0.01, early_stopping_patience=8)],
        )
        for n, v in best_run.hyperparameters.items():
            setattr(trainer.args, n, v)
        
        trainer.train()

        # Test the model and evaluate
        model_predictions = trainer.predict(test_dataset).predictions
        test_metrics = compute_metrics((model_predictions, test_labels))
        print(test_metrics)

        conf_matrix = confusion_matrix(test_labels, np.argmax(model_predictions, axis=-1))
        disp = ConfusionMatrixDisplay(conf_matrix, display_labels=["BS", "LU", "ZH", "BE"])
        disp.plot()
        plt.show()

if __name__ == "__main__":
    main()
