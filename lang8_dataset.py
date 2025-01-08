from typing import *
import os
from gec_dataset import IGECDataset, Entry


class Lang8Dataset(IGECDataset):

    def __init__(self, lang8_dir: str) -> None:

        self._entries: List[Entry] = []

        with open(os.path.join(lang8_dir, "entries.train")) as f:
            # read the file line by line
            for line in f:
                # split the line by tab
                splited = line.split('\t')

                # check if this entry has any corrected sentence
                if len(splited) >= 6:
                    original = splited[4].replace('\n', '')
                    corrected = splited[5].replace('\n', '')

                    if original != corrected:
                        self._entries.append(
                            Entry(
                                original_sentence=original,
                                corrected_sentence=corrected
                            ))

    def __len__(self):
        """
        Return total number of entries of the dataset
        """
        return len(self._entries)

    def __getitem__(self, i: int) -> Entry:
        """
        Return i-th entry of the dataset
        """
        return self._entries[i]


if __name__ == "__main__":

    dataset = Lang8Dataset("datasets/lang-8-en-1.0")

    print(f"Number of entries: {len(dataset)}")
    for i in range(3):
        print(f"Entry[{i}]: {dataset[i]}")
