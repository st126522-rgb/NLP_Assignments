
import json
import torch
import torch.nn as nn
from transformers import BertTokenizerFast
from pathlib import Path

# -------------------------
# Resolve project root safely
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # go from app/ → project root


class BertConfigScratch:
    def __init__(self, vocab_size, max_len=128, hidden_size=256, num_heads=4, num_layers=4,
                 intermediate_size=1024, dropout=0.1, layer_norm_eps=1e-12):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps


class BertEmbeddings(nn.Module):
    def __init__(self, cfg, pad_id):
        super().__init__()
        self.word_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=pad_id)
        self.position_embeddings = nn.Embedding(cfg.max_len, cfg.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, cfg.hidden_size)

        self.layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, input_ids, token_type_ids=None):
        B, T = input_ids.shape
        if token_type_ids is None:
            token_type_ids = torch.zeros((B, T), dtype=torch.long, device=input_ids.device)
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)

        x = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(positions)
            + self.token_type_embeddings(token_type_ids)
        )
        return self.dropout(self.layer_norm(x))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(cfg.hidden_size, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
        self.attn_ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.attn_dropout = nn.Dropout(cfg.dropout)

        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.intermediate_size),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.intermediate_size, cfg.hidden_size),
        )
        self.ffn_ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.ffn_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, attention_mask):
        key_padding_mask = (attention_mask == 0)
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.attn_ln(x + self.attn_dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.ffn_ln(x + self.ffn_dropout(ffn_out))
        return x


class BertEncoderScratch(nn.Module):
    def __init__(self, cfg, pad_id):
        super().__init__()
        self.emb = BertEmbeddings(cfg, pad_id)
        self.layers = nn.ModuleList([TransformerEncoderLayer(cfg) for _ in range(cfg.num_layers)])

    def forward(self, input_ids, attention_mask):
        x = self.emb(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x


def mean_pooling(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class SBERTSoftmaxNLI(nn.Module):
    def __init__(self, encoder, hidden_size, num_labels=3, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(3 * hidden_size, num_labels)

    def encode(self, input_ids, attention_mask):
        x = self.encoder(input_ids, attention_mask)
        return mean_pooling(x, attention_mask)

    def forward(self, a_ids, a_mask, b_ids, b_mask):
        u = self.encode(a_ids, a_mask)
        v = self.encode(b_ids, b_mask)
        feats = torch.cat([u, v, torch.abs(u - v)], dim=-1)
        feats = self.drop(feats)
        return self.classifier(feats)


def load_sbert(meta_relative_path: str, device="cpu"):

    meta_path = PROJECT_ROOT / meta_relative_path
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    tokenizer = BertTokenizerFast.from_pretrained(PROJECT_ROOT / meta["tokenizer_dir"])

    with open(PROJECT_ROOT / meta["bert_meta_path"], "r", encoding="utf-8") as f:
        bert_meta = json.load(f)

    pad_id = bert_meta["special_ids"]["PAD_ID"]

    cfg = BertConfigScratch(
        vocab_size=bert_meta["vocab_size"],
        max_len=bert_meta["max_len"],
        hidden_size=bert_meta["hidden_size"],
        num_heads=bert_meta["num_heads"],
        num_layers=bert_meta["num_layers"],
        intermediate_size=bert_meta["intermediate_size"],
        dropout=bert_meta["dropout"],
        layer_norm_eps=bert_meta["layer_norm_eps"],
    )

    encoder = BertEncoderScratch(cfg, pad_id)

    # load encoder weights
    ckpt_bert = torch.load(PROJECT_ROOT / meta["bert_ckpt_path"], map_location=device)
    enc_state = {k[len("encoder."):]: v for k, v in ckpt_bert["state_dict"].items() if k.startswith("encoder.")}
    encoder.load_state_dict(enc_state)

    model = SBERTSoftmaxNLI(encoder, cfg.hidden_size, meta["num_labels"])
    ckpt_sbert = torch.load(PROJECT_ROOT / meta["sbert_ckpt_path"], map_location=device)
    model.load_state_dict(ckpt_sbert["state_dict"])

    model.to(device).eval()

    id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    max_len = int(meta["max_len"])

    return tokenizer, model, id2label, max_len
