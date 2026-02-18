import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import torch
import torch.nn.functional as F

# Import your custom loader
from model_def import load_sbert

# --- 1. Configuration & Model Loading ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# UPDATE THIS PATH: Set this to the relative path of your SBERT meta JSON
# as expected by your PROJECT_ROOT logic in model_defs.py
META_PATH = "models/sbert_softmax_snli_meta.json" 

print("Loading model and tokenizer...")
try:
    tokenizer, model, id2label, MAX_LEN = load_sbert(META_PATH, device=DEVICE)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure META_PATH points to your valid JSON and all paths inside it are correct.")
    # Fallbacks so the app layout still renders (though prediction will fail)
    tokenizer, model = None, None
    id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    MAX_LEN = 128

# --- 2. Dash Application Layout ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1("SBERT NLI Inference", style={'textAlign': 'center'}),
        html.P("Testing the custom PyTorch SBERT model.", style={'textAlign': 'center', 'color': '#666'}),
        
        html.Div([
            html.Label("Premise (Sentence A):"),
            dcc.Textarea(
                id='premise-input',
                value='A group of people are playing football in the park.',
                style={'width': '100%', 'height': 60, 'marginBottom': '15px'}
            ),
            
            html.Label("Hypothesis (Sentence B):"),
            dcc.Textarea(
                id='hypothesis-input',
                value='Some people are playing a sport outdoors.',
                style={'width': '100%', 'height': 60}
            ),
            
            html.Button('Predict Relationship', id='submit-btn', n_clicks=0, 
                       style={'marginTop': '20px', 'width': '100%', 'height': '40px', 
                              'backgroundColor': '#007BFF', 'color': 'white', 'border': 'none', 'cursor': 'pointer'})
        ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '8px', 'backgroundColor': '#f9f9f9'}),
        
        html.Div(id='results-container', style={'display': 'none'}, children=[
            html.H3(id='prediction-text', style={'textAlign': 'center', 'marginTop': '30px'}),
            dcc.Graph(id='proba-graph')
        ])
    ], style={'fontFamily': 'sans-serif', 'maxWidth': '800px', 'margin': '40px auto'})
])

# --- 3. Callbacks & Inference Logic ---
@app.callback(
    [Output('results-container', 'style'),
     Output('prediction-text', 'children'),
     Output('proba-graph', 'figure')],
    [Input('submit-btn', 'n_clicks')],
    [State('premise-input', 'value'),
     State('hypothesis-input', 'value')]
)
def predict_nli(n_clicks, premise, hypothesis):
    if n_clicks == 0 or not premise or not hypothesis or model is None:
        return {'display': 'none'}, "", go.Figure()

    # Tokenize input A
    enc_a = tokenizer(
        premise, 
        return_tensors="pt", 
        padding="max_length", 
        truncation=True, 
        max_length=MAX_LEN
    ).to(DEVICE)

    # Tokenize input B
    enc_b = tokenizer(
        hypothesis, 
        return_tensors="pt", 
        padding="max_length", 
        truncation=True, 
        max_length=MAX_LEN
    ).to(DEVICE)

    # Forward pass
    with torch.no_grad():
        logits = model(
            enc_a["input_ids"], enc_a["attention_mask"],
            enc_b["input_ids"], enc_b["attention_mask"]
        )
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    # Process results
    labels = [id2label[0], id2label[1], id2label[2]]
    winner_idx = int(probs.argmax())
    winner_label = id2label[winner_idx].capitalize()

    # Create visualization
    colors = ['#cccccc'] * 3
    colors[winner_idx] = '#28a745' if winner_label == 'Entailment' else '#ffc107' if winner_label == 'Neutral' else '#dc3545'

    fig = go.Figure(data=[
        go.Bar(x=labels, y=probs, marker_color=colors, text=[f"{p:.1%}" for p in probs], textposition='auto')
    ])
    fig.update_layout(
        title="Confidence Scores",
        yaxis_range=[0, 1],
        template="plotly_white",
        height=350,
        margin=dict(t=40, b=40, l=40, r=40)
    )

    return {'display': 'block'}, f"Prediction: {winner_label}", fig

if __name__ == '__main__':
    app.run(debug=True)