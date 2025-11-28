import numpy as np
import torch
import argparse
import os
import gdown
from evaluate import load
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    BertTokenizer,
    RobertaTokenizer,
    DataCollatorWithPadding,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# download_ids = [train_set, test_set, adversarial_train_set, adversarial_test_set]
download_ids = ['1QJ38pa4EZkpkW7XykVfnacnORrwuZ1gb',
                '1JW06Jy-4rI0x452N8v0WNOmdV8J_wYCU',
                '1GTNn5K89ASxDixB0CowSLj4vzhy395hP',
                '1AGgznq-oKUAxcQr3shGNeZUm4coZ8gN6']

path_main = "E:/Universita/2 Magistrale/18 CFU/3. Multilingual Natural Language Processing/Homework 2/Project/main"
path_adv = "E:/Universita/2 Magistrale/18 CFU/3. Multilingual Natural Language Processing/Homework 2/Project/" \
           "adversarial"

main_language_model_name = "bert-base-cased"
adv_language_model_name = "roberta-base"

main_tokenizer = BertTokenizer.from_pretrained(main_language_model_name)
main_data_collator = DataCollatorWithPadding(tokenizer=main_tokenizer)
adv_tokenizer = RobertaTokenizer.from_pretrained(adv_language_model_name)
adv_data_collator = DataCollatorWithPadding(tokenizer=adv_tokenizer)


def main_tokenize_function(examples):
    return main_tokenizer(examples['premise'], examples['hypothesis'], truncation=True)


def adv_tokenize_function(examples):
    return adv_tokenizer(examples['premise'], examples['hypothesis'], truncation=True)


def modify_labels_list(labels_list):
    modified_list = []
    for item in labels_list:
        if item == 'ENTAILMENT':
            modified_list.append(0)
        elif item == 'NEUTRAL':
            modified_list.append(1)
        else:
            modified_list.append(2)
    return modified_list


def compute_metrics(eval_pred):
    load_accuracy = load("accuracy")
    load_precision = load("precision")
    load_recall = load("recall")
    load_f1 = load("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    precision = load_precision.compute(predictions=predictions, references=labels, labels=[0, 1, 2],
                                       average=None)["precision"]
    recall = load_recall.compute(predictions=predictions, references=labels, labels=[0, 1, 2], average=None)["recall"]
    f1 = load_f1.compute(predictions=predictions, references=labels, labels=[0, 1, 2], average=None)["f1"]
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def download(name, download_id):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    gdown.download(id=download_id, output=os.path.join(current_directory, f"{name}.jsonl"))
    new_dataset = Dataset.from_parquet(f'{name}.parquet')
    return new_dataset


def process_data(data_name, download_id, tokenize_function, name='dataset'):
    dataset = download(data_name, download_id)
    # print(f"Tokenize {name}")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    new_labels_dataset = modify_labels_list(tokenized_dataset['label'])
    no_labels_dataset = tokenized_dataset.remove_columns(['label'])
    tokenized_dataset = no_labels_dataset.add_column('label', new_labels_dataset)
    return tokenized_dataset


def print_eval(result):
    print("Loss : ", result['eval_loss'])
    print("Accuracy : ", result['eval_accuracy'])
    print("Precision : ", result['eval_precision'])
    print("Recall : ", result['eval_recall'])
    print("F1-Score : ", result['eval_f1'])
    print("Runtime : ", result['eval_runtime'])
    print("Samples per Second : ", result['eval_samples_per_second'])
    print("Steps per Second : ", result['eval_steps_per_second'])


main_training_args = TrainingArguments(
    output_dir=path_main,
    num_train_epochs=5,
    per_device_train_batch_size=32,
    warmup_steps=500,
    weight_decay=0.005,
    save_strategy="no",
    learning_rate=1e-4
    )
adv_training_args = TrainingArguments(
    output_dir=path_adv,
    num_train_epochs=5,
    per_device_train_batch_size=32,
    warmup_steps=500,
    weight_decay=0.005,
    save_strategy="no",
    learning_rate=1e-4
    )


# -------------------------------------------------------------------------------------------------------------------- #


def train():
    tokenized_train = process_data('train_set', download_ids[0], main_tokenize_function)
    std_model = AutoModelForSequenceClassification.from_pretrained(main_language_model_name,
                                                                   ignore_mismatched_sizes=True,
                                                                   output_attentions=False,
                                                                   output_hidden_states=False,
                                                                   num_labels=3)
    std_model.to(device)
    trainer = Trainer(
        model=std_model,
        args=main_training_args,
        train_dataset=tokenized_train,
        tokenizer=main_tokenizer,
        data_collator=main_data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model()


def evaluate():
    tokenized_train = process_data('train_set', download_ids[0], main_tokenize_function)
    tokenized_test = process_data('test_set', download_ids[1], main_tokenize_function)
    tokenized_adv_test = process_data('adversarial_test_set', download_ids[3], main_tokenize_function)
    std_model = AutoModelForSequenceClassification.from_pretrained(path_main,
                                                                   ignore_mismatched_sizes=True,
                                                                   output_attentions=False,
                                                                   output_hidden_states=False,
                                                                   num_labels=3)
    std_model.to(device)
    trainer = Trainer(
        model=std_model,
        args=main_training_args,
        train_dataset=tokenized_train,
        tokenizer=main_tokenizer,
        data_collator=main_data_collator,
        compute_metrics=compute_metrics,
    )
    result = trainer.evaluate(tokenized_test)
    print("Evaluation Using the Advanced Test Set")
    print_eval(result)
    adv_result = trainer.evaluate(tokenized_adv_test)
    print("Evaluation Using the Advanced Test Set")
    print_eval(adv_result)


def train_advanced():
    tokenized_train = process_data('train_set', download_ids[0], adv_tokenize_function)
    tokenized_adv_train = process_data('adversarial_dataset', download_ids[2], adv_tokenize_function)
    train_dataset = concatenate_datasets([tokenized_train, tokenized_adv_train])
    train_dataset = train_dataset.shuffle(seed=42)
    adv_model = AutoModelForSequenceClassification.from_pretrained(adv_language_model_name,
                                                                   ignore_mismatched_sizes=True,
                                                                   output_attentions=False,
                                                                   output_hidden_states=False,
                                                                   num_labels=3)
    adv_model.to(device)
    trainer = Trainer(
        model=adv_model,
        args=adv_training_args,
        train_dataset=train_dataset,
        tokenizer=adv_tokenizer,
        data_collator=adv_data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model()


def evaluate_advanced():
    tokenized_train = process_data('train_set', download_ids[0], adv_tokenize_function)
    tokenized_test = process_data('test_set', download_ids[1], adv_tokenize_function)
    tokenized_adv_test = process_data('adversarial_test_set', download_ids[3], adv_tokenize_function)
    adv_model = AutoModelForSequenceClassification.from_pretrained(path_adv,
                                                                   ignore_mismatched_sizes=True,
                                                                   output_attentions=False,
                                                                   output_hidden_states=False,
                                                                   num_labels=3)
    adv_model.to(device)
    trainer = Trainer(
        model=adv_model,
        args=adv_training_args,
        train_dataset=tokenized_train,
        tokenizer=adv_tokenizer,
        data_collator=adv_data_collator,
        compute_metrics=compute_metrics,
    )
    result = trainer.evaluate(tokenized_test)
    print("Evaluation Using the Advanced Test Set")
    print_eval(result)
    adv_result = trainer.evaluate(tokenized_adv_test)
    print("Evaluation Using the Advanced Test Set")
    print_eval(adv_result)


def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('-t_b', '--train_b', action='store_true')
    parser.add_argument('-e_b', '--evaluate_b', action='store_true')
    parser.add_argument('-t_a', '--train_a', action='store_true')
    parser.add_argument('-e_a', '--evaluate_a', action='store_true')
    args = parser.parse_args()

    if args.train_b:
        train()

    if args.evaluate_b:
        evaluate()

    if args.train_a:
        train_advanced()

    if args.evaluate_a:
        evaluate_advanced()


if __name__ == '__main__':
    main()
