from pathlib import Path
import json
import numpy as np
import torch

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
import evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_DIR = PROJECT_ROOT / "data" / "ner" / "splits" / "hf_rubert_dataset_no_docnum_v1"
LABELS_PATH = PROJECT_ROOT / "data" / "ner" / "splits" / "label_list_no_docnum_v1.json"
OUTPUT_DIR = PROJECT_ROOT / "models" / "ner" / "rubert_ner_manual_v3_full_finetune"
MODEL_NAME = "ai-forever/ruBert-base"


def load_labels(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        label_list = json.load(f)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    return label_list, label2id, id2label


def main():
    dataset = load_from_disk(str(DATASET_DIR))
    print(dataset)

    label_list, label2id, id2label = load_labels(LABELS_PATH)
    print("Labels:", label_list)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    TRAIN_HEAD_ONLY = False

    if TRAIN_HEAD_ONLY:
        for param in model.base_model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True
        

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = []
        true_labels = []

        for prediction, label in zip(predictions, labels):
            current_preds = []
            current_labels = []

            for pred_id, label_id in zip(prediction, label):
                if label_id == -100:
                    continue
                current_preds.append(label_list[pred_id])
                current_labels.append(label_list[label_id])

            true_predictions.append(current_preds)
            true_labels.append(current_labels)

        results = seqeval.compute(
            predictions=true_predictions,
            references=true_labels,
        )

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=1,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=8,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    metrics = trainer.evaluate()
    print("Final eval metrics:", metrics)

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    metrics_path = OUTPUT_DIR / "final_eval_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Model saved to:", OUTPUT_DIR)
    print("Metrics saved to:", metrics_path)


if __name__ == "__main__":
    main()