from dash import Dash, html, dcc, Input, Output, State
import numpy as np
import pickle
from heapq import nlargest

embedding_dicts = {}

for embedding_type in ['glove', 'skipgram', 'skipgram_negative']:
    file_path = f'embed_{embedding_type}.pkl'
    
    with open(file_path, 'rb') as pickle_file:
        embedding_dicts[embedding_type] = pickle.load(pickle_file)

with open('model_gensim.pkl', 'rb') as model_file:
    model_gensim = pickle.load(model_file)

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
            dcc.Dropdown(
                id='model-selector',
                options=[
                    {'label': 'GloVe', 'value': 'glove'},
                    {'label': 'Skip-gram', 'value': 'skipgram'},
                    {'label': 'Skip-gram (Negative)', 'value': 'skipgram_negative'},
                ],
                placeholder='Select a model...',
                style={
                    'width': '70%',
                    'margin': '0 auto',
                    'margin-bottom': '20px',
                }
            ),
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
mapping = {
    'skipgram_negative': 'Skipgram (Negative)',
    'glove': 'GloVe',
    'skipgram': 'Skipgram'
}

# Callback to handle search queries
@app.callback(
    Output('search-results', 'children'),
    [Input('search-button', 'n_clicks')],
    [State('search-query', 'value'), State('model-selector', 'value')]
)
def search(n_clicks, query, model):
    if n_clicks > 0:
        if not query:
            return html.Div("Please enter a query.", style={'color': 'red'})
        if not model:
            return html.Div("Please select a model from the dropdown.", style={'color': 'red'})
        
        embeddings = embedding_dicts.get(model) # using the chosen model
        results = similarWords(query, embeddings)
        return html.Div([
            html.H4(f"Results for '{query}' using model '{mapping[model]}':"),
            html.Ul([html.Li(result) for result in results], style={'list-style-type': 'none'})
        ])
    return html.Div("Enter a query and select a model to see results.", style={'color': 'gray'})

# Running the app
if __name__ == '__main__':
    app.run_server(debug=True)