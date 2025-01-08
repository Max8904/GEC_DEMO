from typing import *
import argparse
import sys
import math

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# Huggingface
from transformers import BertTokenizer, BertModel

from tqdm import tqdm

from tagger import TokenTagger, TokenTag
from lang8_dataset import Lang8Dataset
from token_tag_dataset import TokenTagDataset
from text_dataset import TextGECDataset


class Net(nn.Module):

    def __init__(self, num_tags: int, tokenizer, device: str, backbone_name: str) -> None:
        super().__init__()

        self.device = device
        self.tokenizer = tokenizer

        # pre-trained backbone
        self.bert_model = BertModel.from_pretrained(backbone_name)

        # freeze the weights of BERT
        for param in self.bert_model.parameters():
            param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(True),
            nn.Linear(768, 768),
            nn.ReLU(True),
            nn.Linear(768, num_tags)
        )

    def forward(self, sentences: List[str]):

        tokens = self.tokenizer(
            sentences, return_tensors='pt', padding=True).to(self.device)
        out0, out1 = self.bert_model(**tokens, return_dict=False)

        return self.decoder(out0)


class Trainer:

    def __init__(
        self,
        device: str = 'cpu',
        tags_file: Optional[str] = None,
        filter_indices_file: Optional[str] = None,
        model_name: str = 'bert-base-uncased',
        train_percent: float = 0.95
    ) -> None:

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.device = device

        def tokenize_func(sentence: str):
            # tokenize the sentence using BertTokenizer
            tokens = self.tokenizer(sentence)
            # get the list of token strings
            return self.tokenizer.convert_ids_to_tokens(tokens['input_ids'])

        self.token_tagger = TokenTagger(tokenizer=tokenize_func)

        # laod dataset
        # self.dataset = Lang8Dataset("datasets/lang-8-en-1.0")
        self.dataset = TextGECDataset(
            original_sentences_file="datasets/a1_train_incorr_sentences.txt",
            corrected_sentences_file="datasets/a1_train_corr_sentences.txt",
            max_sentences=300000
        )

        if tags_file is not None:
            print("Loading tags...")
            self.token_tagger.load_tags(tags_file)
            print("Done.")
        else:
            print("Generating tags...")
            self.prepare_token_tagger()
            self.token_tagger.save_tags("all_tags.json")
            print("Done.")
            print("Tags saved to all_tags.json.")

        self.token_tag_dataset = TokenTagDataset(
            dataset=self.dataset,
            token_tagger=self.token_tagger,
            tokenizer=None,
            filtered_indices_file=filter_indices_file)

        self.NUM_TAGS = self.token_tagger.get_num_tags() + 1
        self.PADDING_TOKEN_IDX = self.token_tagger.get_num_tags()

        # split our dataset into trian and eval part
        n_samples = len(self.token_tag_dataset)
        n_train = math.floor(n_samples * train_percent)
        n_eval = n_samples - n_train

        print(f"n_samples: {n_samples}")
        print(f"n_train: {n_train}")
        print(f"n_eval: {n_eval}")

        self.train_dataset, self.eval_dataset = \
            random_split(self.token_tag_dataset, (n_train, n_eval))

        # network
        self.network = Net(
            num_tags=self.NUM_TAGS,
            tokenizer=self.tokenizer,
            device=device,
            backbone_name=model_name
        ).to(self.device)

    def prepare_token_tagger(self):

        for i in range(len(self.dataset)):
            entry = self.dataset[i]
            self.token_tagger.tag_original_sentence(
                entry.original_sentence, entry.corrected_sentence)

        self.token_tagger.generate_indices(drop_tag_thresh=10)

    def train(self, num_epochs=100):

        criterion = nn.CrossEntropyLoss(ignore_index=self.PADDING_TOKEN_IDX)
        optimizer = optim.Adam(self.network.parameters())
        writer = SummaryWriter()
        global_steps = 0

        def collate_fn(data):

            sentences: List[str] = []
            indices: List[int] = []

            # find the maximum length of the tag indices
            max_length = -1
            for sentence, tag_indices in data:
                if len(tag_indices) > max_length:
                    max_length = len(tag_indices)

            for sentence, tag_indices in data:
                sentences.append(sentence)

                # add padding if needed
                if len(tag_indices) < max_length:
                    for _ in range(max_length - len(tag_indices)):
                        # pad using pad token
                        tag_indices.append(self.PADDING_TOKEN_IDX)

                indices.append(tag_indices)

            return sentences, torch.tensor(indices, dtype=torch.long)

        loader = DataLoader(
            self.train_dataset,
            batch_size=64,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True)

        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=64,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )

        for current_epoch in range(num_epochs):

            n_total = 0
            n_correct = 0

            for batch_sentence, batch_tags in tqdm(loader):

                batch_size = batch_tags.size(0)
                seq_length = batch_tags.size(1)

                optimizer.zero_grad()

                preds = self.network(batch_sentence)
                preds_flat = preds.view(batch_size * seq_length, -1)

                batch_tags_flat_dev = batch_tags.view(
                    batch_size * seq_length).to(self.device)
                # print(f"size of pred: {preds_flat.size()}")
                # print(f"size of g: {batch_tags_flat_dev.size()}")
                loss = criterion(preds_flat, batch_tags_flat_dev)
                # calculate accuracy
                n_correct += torch.sum(torch.argmax(preds_flat, dim=1)
                                       == batch_tags_flat_dev).item()
                writer.add_scalar("train/loss", loss, global_step=global_steps)
                loss.backward()
                optimizer.step()

                n_total += batch_size * seq_length
                global_steps += 1

            writer.add_scalar("train/accuracy", n_correct /
                              n_total, global_step=global_steps)

            self.save_weights(f"epoch_{current_epoch + 1}.ckpt")

            # eval

            with torch.no_grad():

                n_total = 0
                n_correct = 0
                print("Evaluating...")
                for batch_sentence, batch_tags in tqdm(eval_loader):

                    batch_size = batch_tags.size(0)
                    seq_length = batch_tags.size(1)

                    batch_tags_flat_dev = batch_tags.view(
                        batch_size * seq_length).to(self.device)

                    preds = self.network(batch_sentence)
                    preds_flat = preds.view(batch_size * seq_length, -1)

                    # calculate accuracy
                    n_correct += torch.sum(torch.argmax(preds_flat, dim=1)
                                           == batch_tags_flat_dev).item()

                    n_total += batch_size * seq_length
                    global_steps += 1

                writer.add_scalar("eval/accuracy", n_correct /
                                  n_total, global_step=global_steps)

    def save_weights(self, filename: str):
        torch.save(self.network.state_dict(), filename)

    def load_weights(self, filename: str):
        self.network.load_state_dict(torch.load(filename))

    def interactive_eval(self, input_sentence: str):

        preds = self.network([input_sentence])[0]
        pred_tag_indices = torch.argmax(preds, dim=1)

        tags: List[TokenTag] = []

        for idx in pred_tag_indices.tolist():
            tags.append(self.token_tagger.get_tag(idx))

        return self.token_tagger.apply_tag(input_sentence, tags)

    def batch_eval(self, input_file: str, output_file: str = "result.txt"):

        print(f"Evaling sentences from {input_file}")

        with open(output_file, 'w', encoding='utf-8') as out_f:

            # read sentences from file
            sentences = []

            with open(input_file, 'r', encoding='utf-8') as f:
                for sentence in f:
                    sentences.append(sentence)

            # create a data loader
            loader = DataLoader(
                dataset=sentences,
                batch_size=512,
                shuffle=False,
                pin_memory=True,
                num_workers=2
            )

            with torch.no_grad():
                for sentence_batch in tqdm(loader):

                    b_size = len(sentence_batch)

                    preds = self.network(sentence_batch)
                    pred_tag_indices = torch.argmax(preds, dim=2)

                    for i in range(b_size):

                        tags: List[TokenTag] = []

                        for idx in pred_tag_indices[i].tolist():
                                if idx == self.PADDING_TOKEN_IDX:
                                    tags.append(TokenTag(type='keep'))
                                else:
                                    tags.append(self.token_tagger.get_tag(idx))

                        out_f.write(self.token_tagger.apply_tag(
                            sentence_batch[i], tags, tag_possibly_longer=True) + "\n")
        print(f"Results saved into {output_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action='store_true')
    parser.add_argument("--eval_file", type=str,
                        help='The sentences to be corrected.')
    parser.add_argument("--ckpt", type=str,
                        help='Path to the saved model checkpoint.')
    parser.add_argument("--tags", type=str,
                        help='Path to the saved tags json file.')
    parser.add_argument("--indices", type=str,
                        help='Path to the saved filtered indices json file.')

    args = parser.parse_args()

    if args.interactive or args.eval_file is not None:

        if not args.tags:
            print("Saved tags file is required to perform evaluation.")
            sys.exit()

        trainer = Trainer(device='cuda', tags_file=args.tags,
                          filter_indices_file=args.indices)
        trainer.load_weights(args.ckpt)

        if args.interactive:
            while True:
                print(">", end='')
                sentence = input()
                print(trainer.interactive_eval(sentence))
        else:
            trainer.batch_eval(args.eval_file)

    else:
        trainer = Trainer(device='cuda', tags_file=args.tags,
                          filter_indices_file=args.indices)

        if args.ckpt:
            trainer.load_weights(args.ckpt)

        trainer.train()
