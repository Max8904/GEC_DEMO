from typing import *
from gec_dataset import IGECDataset, Entry


class TextGECDataset(IGECDataset):

    def __init__(self,
                 original_sentences_file: str,
                 corrected_sentences_file: str,
                 max_sentences: int = -1
                 ) -> None:

        self.orig_sentences = self.read_lines(
            original_sentences_file, max_sentences)
        self.corr_sentences = self.read_lines(
            corrected_sentences_file, max_sentences)

    def read_lines(self, filename: str, n_lines: -1):

        result = []

        with open(filename, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                result.append(line)
                if idx - 1 >= n_lines:
                    break
        
        return result

    def __len__(self) -> int:
        return len(self.orig_sentences)

    def __getitem__(self, i: int) -> Entry:
        return Entry(original_sentence=self.orig_sentences[i], corrected_sentence=self.corr_sentences[i])

if __name__ == "__main__":

    import os

    ds = TextGECDataset(
        original_sentences_file="datasets/a1_train_incorr_sentences.txt",
        corrected_sentences_file="datasets/a1_train_corr_sentences.txt",
        max_sentences=10
        )
    
    for i in range(len(ds)):
        print(ds[i])
    