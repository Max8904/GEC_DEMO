from typing import *
import json

from gec_dataset import IGECDataset
from tagger import TokenTagger


class TokenTagDataset:
    """
    Adds Token tags on top of IGECDataset
    """

    def __init__(self,
                 dataset: IGECDataset,
                 token_tagger: TokenTagger,
                 tokenizer,
                 filtered_indices_file: Optional[str] = None,
                 return_huggingface_format: bool = False
                 ) -> None:

        super().__init__()
        self.dataset = dataset
        self.token_tagger = token_tagger
        self.tokenizer = tokenizer
        self.return_huggingface_format = return_huggingface_format

        print(f"Number of samples before filtering: {len(self.dataset)}")
        if filtered_indices_file is None:
            # go through the whole dataset and drop samples with unknown tag
            print("Filtering dataset...")
            self.filtered_indices: List[int] = []

            for i in range(len(self.dataset)):
                entry = self.dataset[i]
                tags = self.token_tagger.tag_original_sentence(
                    entry.original_sentence, entry.corrected_sentence)
                
                if len(tags) >= 512:
                    continue

                has_unknown_tag = False

                for tag in tags:
                    try:
                        self.token_tagger.get_tag_index(tag)
                    except KeyError:
                        has_unknown_tag = True
                        break

                if not has_unknown_tag:
                    self.filtered_indices.append(i)
            print("Done")
            print("Indices saved to indices.json.")
            self.save_indices("indices.json")
        else:
            with open(filtered_indices_file, 'r', encoding='utf-8') as f:
                self.filtered_indices = json.load(f)
            

        print(
            f"Number of samples after filtering: {len(self)}")

    def __getitem__(self, index):

        entry_idx = self.filtered_indices[index]

        entry = self.dataset[entry_idx]
        tags = self.token_tagger.tag_original_sentence(
            entry.original_sentence, entry.corrected_sentence)

        indices: List[int] = []

        for tag in tags:
            indices.append(self.token_tagger.get_tag_index(tag))

        if self.return_huggingface_format:
            tokenized = self.tokenizer(entry.original_sentence)
            indices[0] = -100
            indices[-1] = -100
            tokenized["labels"] = indices

            return tokenized
        else:
            return entry.original_sentence, indices

    def __len__(self):
        return len(self.filtered_indices)

    def save_indices(self, filename: str):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.filtered_indices, f)
