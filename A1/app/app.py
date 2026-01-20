from dash import Dash, html, dcc, Input, Output, State
import numpy as np
import pickle
import torch
import torch.nn as nn
from heapq import nlargest

class Skipgram(nn.Module):

    def __init__(self, vocab_size, emb_size):
        super(Skipgram,self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, emb_size)
        self.embedding_u = nn.Embedding(vocab_size, emb_size)

    def forward(self, center_words, target_words, all_vocabs):
        center_embeds = self.embedding_v(center_words) # [batch_size, 1, emb_size]
        target_embeds = self.embedding_u(target_words) # [batch_size, 1, emb_size]
        all_embeds    = self.embedding_u(all_vocabs) #   [batch_size, voc_size, emb_size]

        scores      = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)
        #[batch_size, 1, emb_size] @ [batch_size, emb_size, 1] = [batch_size, 1, 1] = [batch_size, 1]

        norm_scores = all_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)
        #[batch_size, voc_size, emb_size] @ [batch_size, emb_size, 1] = [batch_size, voc_size, 1] = [batch_size, voc_size]

        nll = -torch.mean(torch.log(torch.exp(scores)/torch.sum(torch.exp(norm_scores), 1).unsqueeze(1))) # log-softmax
        # scalar (loss must be scalar)

        return nll # negative log likelihood

class SkipgramNeg(nn.Module):

    def __init__(self, voc_size, emb_size):
        super(SkipgramNeg, self).__init__()
        self.embedding_center  = nn.Embedding(voc_size, emb_size)
        self.embedding_outside = nn.Embedding(voc_size, emb_size)
        self.logsigmoid        = nn.LogSigmoid()

    def forward(self, center, outside, negative):
        #center, outside:  (bs, 1)
        #negative       :  (bs, k)

        center_embed   = self.embedding_center(center) #(bs, 1, emb_size)
        outside_embed  = self.embedding_outside(outside) #(bs, 1, emb_size)
        negative_embed = self.embedding_outside(negative) #(bs, k, emb_size)


        uovc           = outside_embed.bmm(center_embed.transpose(1, 2)).squeeze(2) #(bs, 1)
        ukvc           = -negative_embed.bmm(center_embed.transpose(1, 2)).squeeze(2) #(bs, k)
        ukvc_sum       = torch.sum(ukvc, 1).reshape(-1, 1) #(bs, 1)

        loss           = self.logsigmoid(uovc) + self.logsigmoid(ukvc_sum)

        return -torch.mean(loss)

# Load the negative sampling model
with open(r"C:\Users\gaurav\OneDrive\Desktop\NLP\A1\app\skipgram_neg_sample_model.pkl", 'rb') as pickle_file:
    model = pickle.load(pickle_file)

# Load vocab and word2index from the notebook
with open(r"C:\Users\gaurav\OneDrive\Desktop\NLP\A1\vocab.pkl", 'rb') as f:
    vocab = pickle.load(f)
with open(r"C:\Users\gaurav\OneDrive\Desktop\NLP\A1\word2index.pkl", 'rb') as f:
    word2index = pickle.load(f)

# Create embeddings dict from the model
embeddings = {}
for word, idx in word2index.items():
    embeddings[word] = model.embedding_center.weight[idx].detach().cpu().numpy()

def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def similarWords(target_word, embeddings, top_n=10):
    if target_word not in embeddings:
        return ["Word not in Corpus"]

    target_vector = embeddings[target_word]
    cosine_similarities = [(word, cosine_similarity(target_vector, embeddings[word])) for word in embeddings.keys()]
    top_n_words = nlargest(top_n + 1, cosine_similarities, key=lambda x: x[1]) # '+1' because we want to exclude the target word itself

    # Exclude the target word itself
    top_n_words = [word for word, _ in top_n_words if word != target_word]

    return top_n_words[:10]


app = Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("A1 Search Engine", style={'textAlign': 'center', 'font-family': 'Arial, sans-serif', 'margin-top': '20px'}),
    html.Div([
        html.Div([
            dcc.Input(
                id='search-query',
                type='text',
                placeholder='Enter your search query...',
                style={
                    'width': '70%',
                    'margin': '0 auto',
                    'padding': '10px',
                    'display': 'block'
                }
            ),
            html.Button(
                'Search',
                id='search-button',
                n_clicks=0,
                style={
                    'padding': '10px 20px',
                    'background-color': '#007BFF',
                    'color': 'white',
                    'border': 'none',
                    'border-radius': '5px',
                    'margin-top': '20px',
                    'display': 'block',
                    'margin-left': 'auto',
                    'margin-right': 'auto'
                }
            ),
        ], style={
            'textAlign': 'center',
            'padding': '20px',
            'background-color': '#f9f9f9',
            'border': '1px solid #e0e0e0',
            'border-radius': '10px',
            'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
            'width': '50%',
            'margin': '0 auto'
        }),
    ], style={'margin-top': '40px'}),
    html.Div(
        id='search-results',
        style={
            'margin-top': '40px',
            'padding': '20px',
            'textAlign': 'center',
            'font-family': 'Arial, sans-serif'
        }
    ),
])

# For displaying the search results
model_name = 'Skip-gram (Negative)'

# Callback to handle search queries
@app.callback(
    Output('search-results', 'children'),
    [Input('search-button', 'n_clicks')],
    [State('search-query', 'value')]
)
def search(n_clicks, query):
    if n_clicks > 0:
        if not query:
            return html.Div("Please enter a query.", style={'color': 'red'})
        
        results = similarWords(query, embeddings)
        return html.Div([
            html.H4(f"Results for '{query}' using model '{model_name}':"),
            html.Ul([html.Li(result) for result in results], style={'list-style-type': 'none'})
        ])
    return html.Div("Enter a query to see results.", style={'color': 'gray'})

# Running the app
if __name__ == '__main__':
    app.run_server(debug=True)