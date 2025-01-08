## Environment
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install transformers
pip install errant
pip install inflect
pip install tensorboard
python3 -m spacy download en
```
## Training
```
python train.py
```

## Interactive Evaluation
```
python train.py --interactive --ckpt <check_point>  --tags <tags_json> --indices <indices_file>
```

## Evaluation on Sentences
```
python train.py --ckpt <check_point>  --tags <tags_json> --indices <indices_file> --eval_file <path_to_sentences_file>
```
