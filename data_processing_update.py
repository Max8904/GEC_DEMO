import json
from lang8_dataset import Lang8Dataset
from train import Trainer

def write_file(dataset):
    print(f"Number of entries: {len(dataset)}")
    # for i in range(3):
    #     print(f"Entry[{i}]: {dataset[i]}")
    #     print(f"Entry[{i}]: {dataset[i].original_sentence}")
    #     print(f"Entry[{i}]: {dataset[i].corrected_sentence}")
    
    f_original = open('lang-8_text\lang-8_original.txt', 'w')
    f_references = open('lang-8_text\lang-8_references.txt', 'w')
    for i in range(len(dataset)):
        f_original.write(f"{dataset[i].original_sentence}\n")
        f_references.write(f"{dataset[i].corrected_sentence}\n")

    f_original.close()
    f_references.close()

# delete [CLS],[SEP]
def delete_tag():
    f1 = open('lang-8_text/result.txt', 'r')
    f2 = open('lang-8_text/lang-8_corrected.txt', 'w')
    for line in f1.readlines():
        # print(line)
        line = line.replace("[CLS] ", "")
        line = line.replace(" [SEP]", "")
        # print(line)
        f2.write(line)

def read_indices():
    with open("indices.json", 'r', encoding='utf-8') as f:
        return json.load(f)

if __name__ == "__main__":
    dataset = Lang8Dataset("datasets/lang-8-en-1.0")

    # of -> original
    # rf -> reference
    train_of = open("lang-8/l8_train_original.txt", 'w', encoding='utf-8')
    train_rf = open("lang-8/l8_train_reference.txt", 'w', encoding='utf-8')
    test_of = open("lang-8/l8_test_original.txt", 'w', encoding='utf-8') 
    test_rf = open("lang-8/l8_test_reference.txt", 'w', encoding='utf-8') 

    train_indices = set(read_indices())

    for idx in train_indices:
        train_of.write((dataset[idx].original_sentence).lower() + "\n")
        train_rf.write((dataset[idx].corrected_sentence).lower() + "\n")
    
    idx = 0
    num = 0
    while True:
        if idx not in train_indices:
            test_of.write(dataset[idx].original_sentence.lower() + "\n")
            test_rf.write(dataset[idx].corrected_sentence.lower() + "\n")
            num += 1
        idx += 1
        if num >= len(train_indices):
        # if num >= 100:
            break

    # delete_tag()

# output ref m2 file
# errant_parallel -ori lang-8_text\lang-8_original.txt -cor lang-8_text\lang-8_references.txt -out lang-8_text/lang-8_ref.m2

# output hyp m2 file
# errant_parallel -ori lang-8_text\lang-8_original.txt -cor lang-8_text\lang-8_corrected.txt -out lang-8_text/lang-8_hyp.m2

# compare ref with hyp
# errant_compare -ref lang-8_text/lang-8_ref.m2 -hyp lang-8_text/lang-8_hyp.m2

# output corrected file
# python train.py --ckpt epoch_4.ckpt  --tags all_tags.json --indices indices.json --eval_file lang-8_text\lang-8_original.txt