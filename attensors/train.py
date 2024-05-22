import torch
import torch.nn as nn
import torch.optim as optim
from dataloaders import TextDataset, collate_fn
from torch.utils.data import DataLoader

from attensors.layers import BPETokenizer
from attensors.models import GPT


def train_model(
    model,
    dataloader,
    num_epochs,
    learning_rate,
    device,
    batch_size,
    num_heads,
    max_len,
    d_model,
    vocab_size,
):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = inputs.transpose(0, 1), targets.transpose(0, 1)
            trg_mask = (
                torch.triu(
                    torch.ones(
                        inputs.size(1), int(d_model / num_heads), max_len, max_len
                    )
                )
                == 0
            )
            trg_mask = trg_mask.to(device)

            optimizer.zero_grad()
            print(inputs.shape)
            print(trg_mask.shape)
            outputs = model(inputs, trg_mask)
            # print("------------------")
            outputs = outputs.permute(1, 0, 2)

            targets = targets.permute(1, 0)
            outputs = outputs.reshape(-1, vocab_size)
            targets = targets.reshape(-1)
            # loss = criterion(outputs.reshape(-1,
            # outputs.size(-1)), targets.reshape(-1))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}],Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}"  # noqa
                )


# Example usage:

texts = ["hola que tal"]

tokenizer = BPETokenizer.load_tokenizer("tokenizer.json")

texts = texts[-1] * 100

print(texts)

d_model = 512
num_layers = 1
ff_hidden_dim = 1024
num_heads = 2
dropout = 0
vocab_size = len(tokenizer.vocab)
max_len = 512
batch_size = 8
num_epochs = 10
learning_rate = 1e-4

model = GPT(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    ff_hidden_dim=ff_hidden_dim,
    dropout=dropout,
    vocab_size=vocab_size,
    max_len=max_len,
)
dataset = TextDataset(texts, tokenizer, max_len=512)
dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

train_model(
    model,
    dataloader,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    device=device,
    batch_size=batch_size,
    num_heads=num_heads,
    max_len=max_len,
    d_model=d_model,
    vocab_size=vocab_size,
)


def generate_text(model, tokenizer, start_text, max_length, device, d_model):
    model = model.to(device)
    model.eval()

    tokens = tokenizer.encode(start_text)
    tokens = tokens[:max_length]
    tokens_tensor = torch.tensor(tokens).unsqueeze(1).to(device)
    generated_tokens = tokens

    for _ in range(max_length - len(tokens)):
        seq_len, batch_size = tokens_tensor.size()
        num_heads = model.gpt_blocks[0].attention.num_heads

        decoder_self_mask = (
            torch.triu(
                torch.ones(
                    batch_size,
                    num_heads,
                    seq_len,
                    seq_len,
                    device=device,
                )
            )
            == 0
        )

        with torch.no_grad():
            outputs = model(tokens_tensor, decoder_self_mask)

        next_token_logits = outputs[-1, 0, :]
        next_token = torch.argmax(next_token_logits).item()
        generated_tokens.append(next_token)

        tokens_tensor = torch.tensor(generated_tokens).unsqueeze(1).to(device)

    generated_text = tokenizer.decode(generated_tokens)
    return generated_text


# Example usage:
start_text = "Isabella Cruise "
generated_text = generate_text(
    model, tokenizer, start_text, max_length=50, device=device, d_model=10
)
