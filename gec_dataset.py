from dataclasses import dataclass


@dataclass
class Entry:
    original_sentence: str
    corrected_sentence: str


class IGECDataset:

    def __len__(self) -> int:
        """
        Return total number of entries of the dataset
        """
        raise NotImplementedError()

    def __getitem__(self, i: int) -> Entry:
        """
        Return i-th entry of the dataset
        """
        raise NotImplementedError()
