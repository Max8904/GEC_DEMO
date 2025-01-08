from lang8_dataset import Lang8Dataset
from train import Trainer
import streamlit as st

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

if __name__ == "__main__":
    # dataset = Lang8Dataset("datasets/lang-8-en-1.0")
    # write_file(dataset)

    delete_tag()


    
    
# output ref m2 file
# errant_parallel -ori lang-8\l8_train_original.txt -cor lang-8\l8_train_reference.txt -out lang-8_m2\lang-8_train_ref.m2
# errant_parallel -ori lang-8\l8_test_original.txt -cor lang-8\l8_test_reference.txt -out lang-8_m2\lang-8_test_ref.m2

# output hyp m2 file
# errant_parallel -ori lang-8\l8_train_original.txt -cor lang-8\l8_train_corrected_result.txt -out lang-8_m2\lang-8_train_hyp.m2
# errant_parallel -ori lang-8\l8_test_original.txt -cor lang-8\l8_test_corrected_result.txt -out lang-8_m2\lang-8_test_hyp.m2

# compare ref with hyp
# errant_compare -ref lang-8_m2\lang-8_train_ref.m2 -hyp lang-8_m2\lang-8_train_hyp.m2
# errant_compare -ref lang-8_m2\lang-8_test_ref.m2 -hyp lang-8_m2\lang-8_test_hyp.m2

# output corrected file(old)
# python train.py --ckpt epoch_4.ckpt  --tags all_tags.json --indices indices.json --eval_file lang-8_text\lang-8_original.txt

# output corrected file(new)
# python train.py --ckpt epoch_4.ckpt  --tags all_tags.json --indices indices.json --eval_file lang-8\l8_train_original.txt
# python train.py --ckpt epoch_4.ckpt  --tags all_tags.json --indices indices.json --eval_file lang-8\l8_test_original.txt