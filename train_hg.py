from typing import *
import argparse
import sys
import math

import numpy as np
import torch
from torch.utils.data import random_split

# Huggingface
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification

# Our own stuff
from tagger import TokenTagger, TokenTag
from lang8_dataset import Lang8Dataset
from token_tag_dataset import TokenTagDataset


class App:

    def __init__(
            self,
            device: str = 'cpu',
            model_name: str = 'distilbert-base-uncased',
            train_percent: float = 0.99,
            tags_file: Optional[str] = None,
            filter_indices_file: Optional[str] = None) -> None:

        self.device = device

        # hugging face tokenizer
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

        def tokenize_func(sentence: str):
            # tokenize the sentence using BertTokenizer
            tokens = self.hf_tokenizer(sentence)
            # get the list of token strings
            return self.hf_tokenizer.convert_ids_to_tokens(tokens['input_ids'])

        # initialize our token tagger
        self.token_tagger = TokenTagger(tokenizer=tokenize_func)

        # laod dataset
        self.dataset = Lang8Dataset("datasets/lang-8-en-1.0")

        if tags_file is not None:
            print("Loading tags...")
            self.token_tagger.load_tags(tags_file)
            print("Done.")
        else:
            print("Generating tags...")
            self.prepare_token_tagger()
            self.token_tagger.save_tags("all_tags.json")
            print("Done.")

        # add token tag to the dataset
        self.token_tag_dataset = TokenTagDataset(
            dataset=self.dataset,
            token_tagger=self.token_tagger,
            tokenizer=self.hf_tokenizer,
            filtered_indices_file=filter_indices_file,
            return_huggingface_format=True)

        # split our dataset into trian and eval part
        n_samples = len(self.token_tag_dataset)
        n_train = math.floor(n_samples * train_percent)
        n_eval = n_samples - n_train

        self.train_dataset, self.eval_dataset = \
            random_split(self.token_tag_dataset, (n_train, n_eval))

        # Note: Anything prefixed with hf comes from Huggingface
        self.hf_model = \
            AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=self.token_tagger.get_num_tags())

        self.hf_collator = DataCollatorForTokenClassification(
            self.hf_tokenizer)
        self.hf_training_args = TrainingArguments(
            output_dir="model_output",
            learning_rate=0.1,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=1,
            num_train_epochs=100,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            n_total = 0
            n_correct = 0

            for i in range(len(labels)):
                for j in range(len(labels[i])):
                    n_total += 1

                    if predictions[i][j] == labels[i][j]:
                        n_correct += 1

            return {
                "accuracy": n_correct / n_total,
            }

        self.hf_trainer = Trainer(
            model=self.hf_model,
            args=self.hf_training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.hf_tokenizer,
            data_collator=self.hf_collator,
            compute_metrics=compute_metrics
        )

    def prepare_token_tagger(self):

        for i in range(len(self.dataset)):
            entry = self.dataset[i]
            self.token_tagger.tag_original_sentence(
                entry.original_sentence, entry.corrected_sentence)

        self.token_tagger.generate_indices(drop_tag_thresh=10)

    def train(self):
        self.hf_trainer.train()

    def save_weights(self, directory: str):
        self.hf_model.save_pretrained(directory)

    def load_weights(self, directory: str):
        self.hf_model.from_pretrained(directory)

    def interactive_eval(self, input_sentence: str):

        tokenized = self.hf_tokenizer(
            [input_sentence,], return_tensors='pt').to(self.device)
        print(tokenized)
        preds = self.hf_model(**tokenized)[0][0]
        print(preds.shape)
        pred_tag_indices = torch.argmax(preds, dim=1)
        print("pred tag indices")
        print(pred_tag_indices)

        tags: List[TokenTag] = []

        for idx in pred_tag_indices.tolist():
            tags.append(self.token_tagger.get_tag(idx))

        print(len(tags))

        return self.token_tagger.apply_tag(input_sentence, tags)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action='store_true')
    parser.add_argument("--ckpt", type=str,
                        help='Path to the saved model checkpoint.')
    parser.add_argument("--tags", type=str,
                        help='Path to the saved tags json file.')
    parser.add_argument("--indices", type=str,
                        help='Path to the saved filtered indices json file.')

    args = parser.parse_args()

    if args.interactive:

        if not args.tags:
            print("Saved tags file is required to perform evaluation.")
            sys.exit()

        app = App(device='cuda', tags_file=args.tags, filter_indices_file=args.indices)
        app.load_weights(args.ckpt)

        while True:
            print(">", end='')
            sentence = input()
            print(app.interactive_eval(sentence))

    else:
        app = App(device='cuda', tags_file=args.tags, filter_indices_file=args.indices)

        if args.ckpt:
            app.load_weights(args.ckpt)

        app.train()
