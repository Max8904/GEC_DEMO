from typing import *
from dataclasses import dataclass
import errant
import json
import inflect
import sys


@dataclass
class Edit:
    start: int
    end: int
    correction: str
    type: str


@dataclass
class TokenTag:
    type: Literal[
        "keep",         # keep the token intact
        "replace",      # replace the content of the token with the content of the tag
        "delete",       # delete the token
        "append",       # append a token to the end of the token
        "pluralize",    # pluralize the content of the token
        "singularize",  # singularize the content of the token
        "0d_append",    # append content to the end of the token
        "1d_append",    # delete the last character of the token and append the content to the end of token
        "2d_append",    # delete the last 2 characters of the token and append the content to the end of token
        "3d_append",    # delete the last 3 characters of the token and append the content to the end of token
    ]
    content: Optional[str] = None

    def to_simple_str(self) -> str:
        if self.type == "replace" or self.type.endswith("append"):
            return self.type.upper() + "_" + self.content
        else:
            return self.type.upper()

    def __hash__(self) -> int:
        return hash(self.to_simple_str())


class TokenTagger:

    def __init__(self, tokenizer) -> None:

        self.annotator = errant.load('en')
        self.tokenizer = tokenizer

        # a set of append and replace tags containing tag specific content string
        self.tag_set: Set[TokenTag] = set()

        self.index_lookup: Optional[dict[TokenTag, int]] = None
        self.tag_lookup: Optional[dict[int, TokenTag]] = None
        self.tag_counter = Counter()

        # inflect
        self.inflect_engine = inflect.engine()

    def tag_original_sentence(self, original_sentence: str, corrected_sentence: str):

        orig_tokens: List[str] = self.tokenizer(original_sentence)
        corr_tokens: List[str] = self.tokenizer(corrected_sentence)

        edits = self.generate_edits(
            ' '.join(orig_tokens), ' '.join(corr_tokens))

        merged_edits: List[Edit] = []

        if len(edits) > 1:

            def merge_edits(edits):
                has_changed = False
                result: List[Edit] = []
                i = 0

                while (i+1) < len(edits):
                    if edits[i].end == edits[i+1].start \
                            and len(edits[i].correction.strip()) > 0 \
                            and len(edits[i + 1].correction.strip()) > 0:
                        result.append(
                            Edit(
                                start=edits[i].start,
                                end=edits[i+1].end,
                                correction=edits[i].correction +
                                ' ' + edits[i+1].correction,
                                type=''
                            )
                        )
                        has_changed = True
                        i += 2
                    else:
                        result.append(edits[i])
                        i += 1

                if i < len(edits):
                    result.append(edits[i])

                return result, has_changed

            merged_edits: List[Edit] = edits[:]

            while True:
                merged_edits, has_changed = merge_edits(merged_edits)

                if not has_changed:
                    edits = merged_edits
                    break

        # for edit in edits:
        #     print(edit.start, edit.end, edit.correction)

        result: List[TokenTag] = []

        cursor = 0
        for index, edit in enumerate(edits):

            for i in range(cursor, edit.start):
                if (i == edit.start - 1 and edit.start == edit.end):
                    tag = TokenTag(type="append", content=edit.correction)
                    result.append(tag)
                else:
                    result.append(TokenTag(type="keep"))

            for index, pos in enumerate(range(edit.start, edit.end)):

                if index == 0 and len(edit.correction.strip()) > 0:
                    tag = TokenTag(type="replace", content=edit.correction)
                    result.append(tag)

                else:
                    result.append(TokenTag(type="delete"))

            cursor = edit.end

        for _ in range(cursor, len(orig_tokens)):
            result.append(TokenTag(type="keep"))

        # deal with plural and singular nouns/verbs

        for index, tag in enumerate(result):
            token = orig_tokens[index]
            if tag.type == "replace" and token.isalpha():
                if len(tag.content.split(' ')) == 1:
                    token_plural = self.inflect_engine.plural(token)

                    if token_plural == tag.content:
                        tag.type = "pluralize"
                        tag.content = None
                        continue

                    token_singular = self.inflect_engine.singular_noun(token)
                    if token_singular == tag.content:
                        tag.type = "singularize"
                        tag.content = None
                        continue

                    if tag.content.startswith(token):
                        # 0d append
                        tag.type = "0d_append"
                        tag.content = tag.content[len(token):]
                        continue
                    if tag.content.startswith(token[:-1]) and len(token) > 1:
                        # 1d append
                        tag.type = "1d_append"
                        tag.content = tag.content[(len(token) - 1):]
                        continue
                    if tag.content.startswith(token[:-2]) and len(token) > 2:
                        # 2d append
                        tag.type = "2d_append"
                        tag.content = tag.content[(len(token) - 2):]
                        continue
                    if tag.content.startswith(token[:-3]) and len(token) > 3:
                        # 3d append
                        tag.type = "3d_append"
                        tag.content = tag.content[(len(token) - 3):]
                        continue
        
        for tag in result:
            if tag.type not in ('keep', 'delete', 'singularize', 'pluralize'):
                self.tag_set.add(tag)
                self.tag_counter[tag] += 1

        return result

    def apply_tag(self, original_sentence: str, tags: List[TokenTag], tag_possibly_longer = False):

        orig_tokens = self.tokenizer(original_sentence)
        output_tokens: List[str] = []

        if tag_possibly_longer:
            tags = tags[:len(orig_tokens)]

        assert len(orig_tokens) == len(tags)

        for i, tag in enumerate(tags):

            if tag.type == "keep":
                output_tokens.append(orig_tokens[i])
            elif tag.type == "replace":
                output_tokens.append(tag.content)
            elif tag.type == "append":
                output_tokens.append(orig_tokens[i])
                output_tokens.append(tag.content)
            # do nothing when type == "delete"
            elif tag.type == "pluralize":
                output_tokens.append(
                    self.inflect_engine.plural(orig_tokens[i]))
            elif tag.type == "singularize":
                output_tokens.append(
                    self.inflect_engine.singular_noun(orig_tokens[i]))
            elif tag.type == "0d_append":
                output_tokens.append(orig_tokens[i] + tag.content)
            elif tag.type == "1d_append":
                output_tokens.append(orig_tokens[i][:-1] + tag.content)
            elif tag.type == "2d_append":
                output_tokens.append(orig_tokens[i][:-2] + tag.content)
            elif tag.type == "3d_append":
                output_tokens.append(orig_tokens[i][:-3] + tag.content)

        return ' '.join(output_tokens)

    def generate_edits(self, original_sentence: str, corrected_sentence: str) -> List[Edit]:

        result: List[Edit] = []

        orig = self.annotator.parse(original_sentence, tokenise=False)
        cor = self.annotator.parse(corrected_sentence, tokenise=False)

        edits = self.annotator.annotate(orig, cor)

        for e in edits:
            result.append(Edit(
                start=e.o_start,
                end=e.o_end,
                correction=e.c_str,
                type=e.type
            ))

        return result

    def generate_indices(self, drop_tag_thresh = 0):
        tag_list = list(self.tag_set)

        self.index_lookup = {
            TokenTag(type='keep'): 0,
            TokenTag(type='delete'): 1,
            TokenTag(type="pluralize"): 2,
            TokenTag(type="singularize"): 3,
        }

        self.tag_lookup = {k: v for (v, k) in self.index_lookup.items()}

        current_index = len(self.index_lookup)

        for tag in tag_list:
            if self.tag_counter[tag] > drop_tag_thresh:
                self.index_lookup[tag] = current_index
                self.tag_lookup[current_index] = tag
                current_index += 1

    def get_tag_index(self, tag: TokenTag) -> int:
        return self.index_lookup[tag]

    def get_tag(self, index: int):
        return self.tag_lookup[index]

    def get_num_tags(self) -> int:
        return len(self.index_lookup)

    def save_tags(self, filename: str):
        with open(filename, 'w', encoding='utf-8') as f:

            items = []

            for tag, index in self.index_lookup.items():
                items.append(
                    {"index": index, "type": tag.type, "content": tag.content})

            json.dump(items, f)

    def load_tags(self, filename: str):

        self.index_lookup = {}
        self.tag_lookup = {}

        with open(filename, 'r', encoding='utf-8') as f:

            items = json.load(f)

            for item in items:

                index = item["index"]
                token = TokenTag(type=item["type"], content=item["content"])

                self.index_lookup[token] = index
                self.tag_lookup[index] = token


if __name__ == "__main__":

    def unit_test():

        passed = True

        from lang8_dataset import Lang8Dataset
        dataset = Lang8Dataset("datasets/lang-8-en-1.0")

        def naive_tokenizer(x):
            return x.split(' ')

        token_tagger = TokenTagger(tokenizer=naive_tokenizer)

        for i in range(0, 5000):
            print(f" i = {i}")
            entry = dataset[i]

            original_sentence = "[SOS] " + entry.original_sentence
            corrected_sentence = "[SOS] " + entry.corrected_sentence

            print(f"o: {original_sentence}")
            print(f"c: {corrected_sentence}")

            tokens = naive_tokenizer(original_sentence)
            tags = token_tagger.tag_original_sentence(
                original_sentence, corrected_sentence)

            # print tokens

            print(len(tokens))
            print(len(tags))

            for i in range(max(len(tokens), len(tags))):
                print(f"{i} -> {tokens[i]} -> {tags[i].to_simple_str()}")
                # print(f"{i} -> {tokens[i]} -> {tags[i]}")

            # for i in range(max(len(tokens), len(tags))):
            #     if tags[i].type.endswith("_append"):
            #         sys.exit()

            applied = token_tagger.apply_tag(original_sentence, tags)
            if (applied != corrected_sentence):
                print(f"a:{applied}")
                print(f"c:{corrected_sentence}")
                passed = False
                break

        print(f"Number of tags: {len(token_tagger.tag_set)}")
        if passed:
            print("Unit test passed ;)")

    unit_test()
