import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import torch
import torch.nn as nn
import torch.nn.functional as F
import spacy
import os
import random
from torchtext.data import Field, Example, Dataset
from datasets import load_dataset
import numpy as np

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("[INFO] Initializing System...")

# --- Tokenizers ---
try:
    spacy_en = spacy.load('en_core_web_sm')
    def tokenize_en(text): return [tok.text for tok in spacy_en.tokenizer(text)]
except OSError:
    print("[INFO] Downloading Spacy...")
    os.system("python -m spacy download en_core_web_sm")
    spacy_en = spacy.load('en_core_web_sm')
    def tokenize_en(text): return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_ne(text): return text.split()

# --- Fields & Vocab ---
SRC = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)
TRG = Field(tokenize=tokenize_ne, init_token='<sos>', eos_token='<eos>')

print("[INFO] Rebuilding Vocabulary...")
# We load a small chunk just to rebuild the vocab mapping exactly as training
full_dataset = load_dataset('CohleM/english-to-nepali')
small_dataset = full_dataset['train'].train_test_split(test_size=0.7, seed=SEED)['train']
train_data = small_dataset.train_test_split(test_size=0.2, seed=SEED)['train']

examples = [Example.fromlist([item['en'], item['ne']], fields=[('src', SRC), ('trg', TRG)]) for item in train_data]
train_ds = Dataset(examples, fields=[('src', SRC), ('trg', TRG)])

SRC.build_vocab(train_ds, min_freq=2)
TRG.build_vocab(train_ds, min_freq=2)
print(f"[DONE] Vocab Ready. (En: {len(SRC.vocab)}, Ne: {len(TRG.vocab)})")

# ==========================================
# 2. MODEL DEFINITIONS (BOTH TYPES)
# ==========================================
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        return outputs, hidden

class AdditiveAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)
    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2))) 
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)

class GeneralAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.transform = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        projected_encoder_outputs = self.transform(encoder_outputs).permute(1, 0, 2)
        hidden = hidden.unsqueeze(2)
        attention = torch.bmm(projected_encoder_outputs, hidden).squeeze(2)
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs).permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        prediction = self.fc_out(torch.cat((output.squeeze(0), weighted.squeeze(0), embedded.squeeze(0)), dim=1))
        return prediction, hidden.squeeze(0), a.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # Only needed for training, but required for loading state_dict
        return None 

# ==========================================
# 3. SMART MODEL LOADING
# ==========================================
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
DROPOUT = 0.5

# Find the file
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'best_model.pt')
if not os.path.exists(model_path):
    # Fallback to local
    model_path = 'best_model.pt'

print(f"[INFO] Attempting to load: {model_path}")

active_model = None
model_type = "Unknown"

# --- STRATEGY 1: Try Additive Attention ---
try:
    attn = AdditiveAttention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DROPOUT, attn)
    model = Seq2Seq(enc, dec, device).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    active_model = model
    model_type = "Additive (Bahdanau)"
    print("[SUCCESS] Loaded Additive Attention Model")
except RuntimeError as e:
    print("[INFO] Additive load failed (Keys mismatch). Trying General Attention...")
    
    # --- STRATEGY 2: Try General Attention ---
    try:
        attn = GeneralAttention(ENC_HID_DIM, DEC_HID_DIM)
        enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DROPOUT)
        dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DROPOUT, attn)
        model = Seq2Seq(enc, dec, device).to(device)
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        active_model = model
        model_type = "General (Luong)"
        print("[SUCCESS] Loaded General Attention Model")
    except Exception as e2:
        print(f"[CRITICAL ERROR] Could not load model: {e2}")
        print("Please ensure hyperparams (HID_DIM=512) match training.")

# ==========================================
# 4. DASH APPLICATION
# ==========================================
def app_inference(text):
    if not text or active_model is None: return ""
    
    active_model.eval()
    tokens = [token.text.lower() for token in spacy_en(text)]
    tokens = [SRC.init_token] + tokens + [SRC.eos_token]
    src_indexes = [SRC.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = active_model.encoder(src_tensor)

    trg_indexes = [TRG.vocab.stoi[TRG.init_token]]
    for i in range(50):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden, _ = active_model.decoder(trg_tensor, hidden, encoder_outputs)
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == TRG.vocab.stoi[TRG.eos_token]: break
    
    trg_tokens = [TRG.vocab.itos[i] for i in trg_indexes]
    return " ".join(trg_tokens[1:])

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

app.layout = dbc.Container([
    html.Div(style={"height": "60px"}),
    dbc.Card([
        dbc.CardBody([
            html.H2("English to Nepali Neural Translator", className="text-center mb-4", style={"color": "#2c3e50"}),
            dbc.Row([
                dbc.Col([
                    html.Label("ENGLISH INPUT", className="small fw-bold text-muted"),
                    dbc.Textarea(id="input-text", placeholder="Type here...", style={"height": "150px", "borderRadius": "8px"}),
                ], md=6),
                dbc.Col([
                    html.Label("NEPALI OUTPUT", className="small fw-bold text-muted"),
                    html.Div(id="output-text", style={"height": "150px", "borderRadius": "8px", "border": "1px solid #ced4da", "padding": "10px", "backgroundColor": "#f8f9fa"}),
                ], md=6),
            ], className="mb-4 g-3"),
            dbc.Button("Translate", id="translate-btn", color="dark", className="w-100", style={"borderRadius": "20px"})
        ])
    ], style={"boxShadow": "0 8px 16px rgba(0,0,0,0.1)", "border": "none"}),
    html.Div(f"Active Model: {model_type}", className="text-center text-muted mt-3 small")
], fluid=True, style={"maxWidth": "900px"})

@app.callback(Output("output-text", "children"), Input("translate-btn", "n_clicks"), State("input-text", "value"), prevent_initial_call=True)
def update_output(n, text):
    return app_inference(text)

if __name__ == '__main__':
    app.run(jupyter_mode='inline', port=8051)