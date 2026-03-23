# A6: NLP Assignment - RAG and Contextual Retrieval
Student ID=126522
Hence, Chapter 2 of required online book pdf is selected.

## Project Overview

This project implements and evaluates Retrieval Augmented Generation (RAG) techniques using Chapter 2 of the Jurafsky & Martin NLP book as the knowledge source.

The assignment focuses on:

- Preparing a QA dataset from the chapter
- Implementing a Naive RAG pipeline
- Implementing Contextual Retrieval
- Comparing both approaches using ROUGE evaluation
- Building an interactive chatbot in the notebook that allows users to ask questions about the chapter.

The system retrieves relevant information from the chapter and generates answers using a local Large Language Model (LLM).

## Project Structure

```
A6/
│
├── 2.pdf                     # Source chapter PDF
├── a6.ipynb                  # Main notebook implementation
├── answer/
│   └── response-st126522-chapter-2.json  # Evaluation JSON output
└── README.md
```

## Task 1 — Dataset Preparation

The first task creates a Question Answer dataset from Chapter 2.

### Steps

- Extract raw text from the chapter PDF
- Clean and normalize the text
- Split the chapter into chunks
- Manually generate 20 question-answer pairs based on chapter content
- Create the ground truth QA dataset

### Output Files

- `a6.ipynb` contains the cleaned text, chunks, and QA pairs
- `answer/response-st126522-chapter-2.json` includes the QA pairs with generated answers

This dataset is used to evaluate the RAG systems in Task 2.

## Task 2 — RAG Pipeline Implementation

Two retrieval pipelines were implemented.

### 1. Naive RAG

Naive RAG uses basic chunking and semantic retrieval.

#### Pipeline

```
Question
   ↓
Sentence Transformer Embedding
   ↓
Nearest Neighbors Search
   ↓
Top-K Chunk Retrieval
   ↓
Prompt Construction
   ↓
LLM Answer Generation
```

#### Model Components

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Language Model**: `google/flan-t5-small`
- **Vector Search**: `sklearn.neighbors.NearestNeighbors`

### 2. Contextual Retrieval

Contextual Retrieval improves RAG by adding additional context to each chunk before embedding.

Instead of embedding raw chunks only, the system generates context prefixes describing the chunk.

#### Example

**Context Prefix:**  
This chunk from Jurafsky & Martin Chapter 2 discusses n-gram language models and their probability estimations.

**Chunk:**  
An n-gram model predicts the next word given the previous n-1 words...

This improves retrieval accuracy because the embedding better captures the semantic meaning of the text.

### Evaluation

Both pipelines were evaluated using ROUGE metrics.

#### Metrics Used

- ROUGE-1
- ROUGE-2
- ROUGE-L

The generated answers were compared with the ground truth answers.

#### Results

| Method              | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---------------------|---------|---------|---------|
| Naive RAG          | 0.0282 | 0.0010 | 0.0282 |
| Contextual Retrieval | 0.0302 | 0.0000 | 0.0276 |

#### Analysis

Contextual Retrieval achieved slightly higher ROUGE-1 (0.0302 vs 0.0282), but lower in ROUGE-2 and ROUGE-L. This marginal improvement may be due to the small generator model struggling with enriched context. Larger models could show more significant benefits from contextual embeddings.

## Task 3 — Interactive Chatbot

An interactive chatbot was implemented in the notebook using Python cells.

The chatbot allows users to:

- Ask questions about Chapter 2
- Retrieve relevant contextual chunks
- Generate answers using the Contextual Retrieval pipeline
- View the source chunks used for the answer

### Features

- Interactive question input in notebook cells
- Top-K retrieval (configurable)
- LLM generated answers
- Source chunk transparency

### Example Questions

Examples that work with the system:

- What is an n-gram language model?
- How does a bigram model differ from a unigram model?
- What is smoothing in language modeling?

### Running the Chatbot

Run the notebook cells in order. The final cells implement the interactive chatbot.

## Technologies Used

| Component       | Tool                  |
|-----------------|-----------------------|
| Language        | Python                |
| LLM             | google/flan-t5-small  |
| Embeddings      | Sentence Transformers |
| Vector Search   | scikit-learn          |
| Evaluation      | ROUGE                 |
| Notebook        | Jupyter               |
| Data Processing | Python / JSON         |

## Key Learnings

This assignment demonstrates:

- How Retrieval Augmented Generation works
- The importance of retrieval quality in RAG systems
- How contextual embeddings can improve semantic search
- Evaluation of generative QA systems
- Integration of LLM pipelines into interactive notebooks

## Conclusion

The project successfully implemented two RAG systems and evaluated their performance on a textbook QA task.

The final system provides an interactive interface in the notebook where users can query textbook knowledge using modern RAG techniques. Contextual Retrieval shows potential for improvement with larger models.
