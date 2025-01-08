from typing import *
import os
import json
from gec_dataset import IGECDataset, Entry


class wi_locness_dataset(IGECDataset):

    def __init__(self, wi_locness_dir: str) -> None:

        self._entries: List[Entry] = []

        allFileList = os.listdir(wi_locness_dir)

        for file in allFileList:
            with open(os.path.join(wi_locness_dir, file)) as f:
                # read the file line by line
                for line in f:
                    data = json.loads(line)
                    # text , edits list
                    # print(data["text"], data["edits"][0][1])

                    i = 0
                    j = 0
                    corrected_sentence = ""
                    if(data["edits"][0][1] != []):
                        for edit_list in data["edits"][0][1]:
                            # print(edit_list)
                            # i is first index
                            i = edit_list[0]
                            if(edit_list[2] != None):
                                corrected_sentence += data["text"][j:i] + edit_list[2]
                            else:
                                edit_list[2] = ""
                                corrected_sentence += data["text"][j:i] + edit_list[2]
                            # j is second index
                            j = edit_list[1]
                    else:
                        corrected_sentence = data["text"]

                    # print(corrected_sentence)
                    # print("=====================================================================================")

                    self._entries.append(
                        Entry(
                            original_sentence = data["text"],
                            corrected_sentence = corrected_sentence
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

    dataset = wi_locness_dataset("json")

    print(f"Number of entries: {len(dataset)}")
    for i in range(3):
        print(f"Entry[{i}]: {dataset[i]}")
