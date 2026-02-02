import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

import torch
import pickle

from model import LSTMLanguageModel   # your model class

# --------------------
# Load model + vocab
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("vocab_lm.pkl", "rb") as f:
    vocab = pickle.load(f)

model = LSTMLanguageModel(
    vocab_size=len(vocab),
    emb_dim=1024,
    hid_dim=1024,
    num_layers=2,
    dropout_rate=0.65
)

model.load_state_dict(torch.load("best-val-lstm_lm.pt", map_location=device))
model.to(device)
model.eval()


# torchtext vocab helpers
itos = vocab.get_itos() if hasattr(vocab, "get_itos") else vocab.itos

# --------------------
# Generation function
# --------------------
def generate(prompt, max_seq_len, temperature):
    tokens = prompt.split()   # or your tokenizer
    indices = [vocab[token] for token in tokens]

    hidden = model.init_hidden(1, device)

    with torch.no_grad():
        for _ in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            output, hidden = model(src, hidden)

            logits = output[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)

            next_idx = torch.multinomial(probs, 1).item()

            if itos[next_idx] == "<eos>":
                break
            if itos[next_idx] == "<unk>":
                continue

            indices.append(next_idx)

    return " ".join(itos[i] for i in indices)

# --------------------
# Dash app
# --------------------
app = dash.Dash(__name__)

app.layout = html.Div(
    style={"width": "600px", "margin": "auto"},
    children=[
        html.H2("LSTM Language Model Demo"),

        dcc.Textarea(
            id="prompt",
            value="Once upon a time",
            style={"width": "100%", "height": 100}
        ),

        html.Br(),

        html.Label("Temperature"),
        dcc.Slider(
            id="temperature",
            min=0.2,
            max=1.5,
            step=0.1,
            value=1.0,
            marks={0.2: "0.2", 0.5: "0.5", 1.0: "1.0", 1.5: "1.5"}
        ),

        html.Br(),

        html.Button("Generate", id="generate", n_clicks=0),

        html.Hr(),

        html.Div(id="output", style={"whiteSpace": "pre-wrap"})
    ]
)

# --------------------
# Callback
# --------------------
@app.callback(
    Output("output", "children"),
    Input("generate", "n_clicks"),
    State("prompt", "value"),
    State("temperature", "value"),
)
def run_generation(n_clicks, prompt, temperature):
    if n_clicks == 0:
        return ""

    text = generate(
        prompt=prompt,
        max_seq_len=40,
        temperature=temperature
    )
    return text


if __name__ == "__main__":
    app.run_server(debug=True)
