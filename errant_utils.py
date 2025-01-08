from typing import *
from dataclasses import dataclass
import errant


@dataclass
class Edit:
    start: int
    end: int
    correction: str
    type: str


# load spacy tokenizer
from spacy.lang.en import English
nlp = English()
# Create a Tokenizer with the default settings for English
# including punctuation rules and exceptions
tokenizer = nlp.tokenizer

def to_tokens(input_sentence: str) -> List[str]:
    return [str(x) for x in list(tokenizer(input_sentence))]


annotator = errant.load('en')
def generate_errant_edits(original_sentence: str, corrected_sentence: str) -> List[Edit]:

    result: List[Edit] = []

    orig = annotator.parse(original_sentence, tokenise=True)
    cor = annotator.parse(corrected_sentence, tokenise=True)

    edits = annotator.annotate(orig, cor)

    for e in edits:
        result.append(Edit(
            start=e.o_start,
            end=e.o_end,
            correction=e.c_str,
            type=e.type
        ))

    return result


def replace_part(input_tokens: List[str], segment: slice, replacement_tokens: List[str]):
    return input_tokens[:segment.start] + replacement_tokens + input_tokens[segment.stop:]

def corrector(original_sentence: str, edits: List[Edit]) -> str:

    original_sentence_tokens = to_tokens(original_sentence)

    offset = 0

    for edit in edits:
        original_sentence_tokens = replace_part(
            original_sentence_tokens, slice(edit.start + offset, edit.end + offset), [edit.correction, ])

        if edit.end == edit.start:
            offset += 1
        else:    
            num_tokens = edit.end - edit.start
            offset -= num_tokens
            offset += 1

    original_sentence_tokens = [x for x in original_sentence_tokens if x != '']
    return to_tokens(' '.join(original_sentence_tokens))


if __name__ == "__main__":

    def unit_test():

        from lang8_dataset import Lang8Dataset
        dataset = Lang8Dataset("datasets/lang-8-en-1.0")

        for i in range(1000):
            
            entry = dataset[i]
            
            edits = generate_errant_edits(entry.original_sentence, entry.corrected_sentence)

            ground = to_tokens(entry.corrected_sentence)
            corrected_sentence = corrector(entry.original_sentence, edits)

            if corrected_sentence != ground:
                print(f"i = {i}")
                print(f"original = {entry.original_sentence}")
                print(f"ground = {to_tokens(entry.corrected_sentence)}")
                print(f"corred = {corrected_sentence}")
                print("Unit test failed :(")
                break
        
        print("Unit test passed ;)")

    unit_test()

    # original_sentence = "We 've known each other for only half a year , but his lesson was a lot of fun ."
    # corrected_sentence = "We ' ve known each other for only half a year , but lessons were was a lot of fun ."

    # print(f"Original sentence = {original_sentence}")
    # print(f"Corrected sentence = {corrected_sentence}")

    # edits = generate_errant_edits(original_sentence, corrected_sentence)

    # for e in edits:
    #     print(e)

    # c = corrector(original_sentence, edits)
    # print(c)

    # a = ['a', 'b', 'c']
    # print(replace_part(a, slice(1, 2), ['x']))