import json
from collections import Counter


class BPETokenizer:
    def __init__(self, vocab=None, vocab_size=10000):
        if vocab is None:
            self.vocab_size = vocab_size
            self.vocab = {"<PAD>": 0}
            self.reverse_vocab = {0: "<PAD>"}

        else:
            self.vocab = vocab
            self.reverse_vocab = {v: k for k, v in vocab.items()}

    def train(self, corpus):
        train_corpus = " ".join(corpus)
        characters = set(train_corpus)
        token_list = []
        for char in characters:
            token_list.append(char)

        current_tokens = list(train_corpus)

        while len(token_list) < self.vocab_size:
            pairs = self.get_pairs(current_tokens)
            new_tokens, most_common = self.replace_most_common_token(
                current_tokens, pairs
            )

            for token in most_common:
                token_list.append(token)

            if current_tokens == new_tokens:
                break

            current_tokens = new_tokens

        for i, token in enumerate(token_list):
            self.vocab[token] = i + 1
            self.reverse_vocab[i + 1] = token

    def get_pairs(self, current_tokens):
        grouped_list = []
        for i in range(0, len(current_tokens), 2):
            if i + 1 < len(current_tokens):
                grouped_list.append(current_tokens[i] + current_tokens[i + 1])
            else:
                grouped_list.append(current_tokens[i])
        return grouped_list

    def replace_most_common_token(self, original_list, token_list):
        counter = Counter(token_list)
        max_count = max(counter.values())
        if max_count == 1:
            return original_list, ""

        most_common = [item for item, count in counter.items() if count == max_count]

        new_tokens = []

        i = 0
        while i < len(original_list) - 1:
            current_token = original_list[i] + original_list[i + 1]
            if current_token in most_common:
                new_tokens.append(current_token)
                i = i + 2
            else:
                new_tokens.append(original_list[i])
                i = i + 1

        return new_tokens, most_common

    def encode(self, input_string):
        encoded_list = []
        while input_string:
            max_token = ""
            for token in self.vocab.keys():
                if input_string.startswith(token) and len(token) > len(max_token):
                    max_token = token
            if max_token:
                encoded_list.append(self.vocab[max_token])
                input_string = input_string[len(max_token) :]
            else:
                encoded_list.append("<UNK>")
                input_string = input_string[1:]
        return encoded_list

    def decode(self, tokens):
        decoded_text = []
        for token in tokens:
            if token in self.reverse_vocab:
                decoded_text.append(self.reverse_vocab[token])
            else:
                decoded_text.append("<UNK>")
        return "".join(decoded_text)

    def save_tokenizer(self, file_path):
        with open(file_path, "w") as f:
            json.dump({"vocab": self.vocab}, f)

    @classmethod
    def load_tokenizer(cls, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
            vocab = data["vocab"]
        return cls(vocab)
