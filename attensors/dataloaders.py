import torch
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text)
        tokens = tokens[: self.max_len]
        tokens = tokens + [self.tokenizer.vocab.get("<PAD>", 0)] * (
            self.max_len - len(tokens)
        )
        tokens = torch.tensor(tokens)
        return tokens, tokens


def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    return inputs, targets


if __name__ == "__main__":
    from attensors.layers import BPETokenizer

    texts = ["hola que tal", "yo estoy bien y tu"]
    tokenizer = BPETokenizer()
    tokenizer.train(texts)
    dataset = TextDataset(texts, tokenizer, max_len=512)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    print(dataset[0][0])
