import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from tqdm import tqdm

from nepalitokenizers import WordPiece
import datasets

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):
        super().__init__()
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hid_dim, vocab_size)

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        return hidden.detach(), cell.detach()

    def forward(self, src, hidden):
        embed = self.embedding(src)
        output, hidden = self.lstm(embed, hidden)
        output = self.dropout(output)
        prediction = self.fc(output)
        return prediction, hidden


def prepare_limited_tokens(dataset_split, tokenizer, max_tokens=50000):
    tokens = []
    for ex in dataset_split:
        text = ex.get('text') or ''
        if not text:
            continue
        enc = tokenizer.encode(text)
        # WordPiece encode returns object with .tokens
        toks = enc.tokens if hasattr(enc, 'tokens') else list(enc)
        if not toks:
            continue
        tokens.extend(toks + ['<eos>'])
        if len(tokens) >= max_tokens:
            tokens = tokens[:max_tokens]
            break
    return tokens


def build_vocab_from_tokens(tokens):
    def iterator():
        yield tokens
    vocab = torchtext.vocab.build_vocab_from_iterator(iterator(), specials=['<unk>', '<eos>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab


def tokens_to_data_tensor(tokens, vocab, batch_size):
    ids = [vocab[token] if token in vocab.get_itos() else vocab['<unk>'] for token in tokens]
    data = torch.LongTensor(ids)
    num_batches = data.shape[0] // batch_size
    data = data[: num_batches * batch_size]
    if data.numel() == 0:
        raise ValueError('Not enough tokens for the given batch size')
    data = data.view(batch_size, -1)
    return data


def get_batch(data, seq_len, idx):
    src = data[:, idx:idx+seq_len]
    target = data[:, idx+1:idx+seq_len+1]
    return src, target


def train_epoch(model, data, optimizer, criterion, batch_size, seq_len, clip, device):
    model.train()
    epoch_loss = 0
    num_batches = data.shape[-1]
    data = data[:, : num_batches - (num_batches - 1) % seq_len]
    num_batches = data.shape[-1]
    hidden = model.init_hidden(batch_size, device)
    for idx in tqdm(range(0, num_batches - 1, seq_len), desc='Training', leave=False):
        optimizer.zero_grad()
        hidden = model.detach_hidden(hidden)
        src, target = get_batch(data, seq_len, idx)
        src, target = src.to(device), target.to(device)
        batch_size = src.shape[0]
        prediction, hidden = model(src, hidden)
        prediction = prediction.reshape(batch_size * seq_len, -1)
        target = target.reshape(-1)
        loss = criterion(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item() * seq_len
    return epoch_loss / max(1, num_batches)


def main():
    print('Device:', DEVICE)
    print('Loading dataset...')
    dataset = datasets.load_dataset('Sakonii/nepalitext-language-model-dataset')

    print('Initializing tokenizer...')
    tokenizer = WordPiece()

    print('Preparing ~50k tokens from train split...')
    tokens = prepare_limited_tokens(dataset['train'], tokenizer, max_tokens=50000)
    print('Collected tokens:', len(tokens))

    vocab = build_vocab_from_tokens(tokens)
    print('Vocab size:', len(vocab))

    batch_size = 64
    seq_len = 32

    data = tokens_to_data_tensor(tokens, vocab, batch_size)

    vocab_size = len(vocab)
    emb_dim = 256
    hid_dim = 512
    num_layers = 2
    dropout_rate = 0.3

    model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 3
    clip = 0.25

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, data, optimizer, criterion, batch_size, seq_len, clip, DEVICE)
        print(f'Epoch {epoch} loss: {loss:.4f}')

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'lstm_nepali_50k.pth')
    torch.save({'model_state_dict': model.state_dict(), 'vocab': vocab.get_itos()}, out_path)
    print('Saved model to', out_path)


if __name__ == '__main__':
    main()
