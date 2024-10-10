# Arabic Information Retrieval System using PyTerrier and MContriever

This project is an information retrieval system that processes and ranks Arabic text documents based on relevance using BM25 and MContriever embeddings. The system performs preprocessing of Arabic text, ranking of query results, and re-ranking of top results using deep embeddings with the MContriever model.

## Features

- **Text Preprocessing**:
  - Cleaning of text by removing URLs, special characters, and punctuation.
  - Normalization of Arabic text by removing diacritics (tashkeel), and longation.
  - Stemming and stopword removal using `arabicstopwords` and `snowballstemmer`.
- **BM25 Retrieval**:
  - Initial ranking of Arabic documents using BM25 as the baseline retrieval model from the PyTerrier library.
- **Deep Re-ranking with MContriever**:
  - Embeddings are generated using the `MContriever` model for both queries and documents.
  - Re-ranking of top BM25 results based on cosine similarity between query and document embeddings.
- **Evaluation**:
  - The system is evaluated using standard information retrieval metrics such as Precision (P), Recall, Reciprocal Rank, and Mean Average Precision (MAP).
  - Evaluation results are compared with the ground-truth relevance judgments (qrels).

## Requirements

You can install the dependencies by running:

```bash
pip install pyterrier pandas transformers torch faiss arabicstopwords snowballstemmer
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Noor-Khial/arabic-ir-system.git
   cd arabic-ir-system
   ```

2. Download the required datasets:
   - Place your dataset `data.csv`.

## Usage

1. **Preprocess the dataset**:
   The dataset is cleaned, normalized, and stemmed using the following script:

   ```python
   df = pd.read_csv('data.csv', encoding='utf-8-sig')
   df['Question'] = df['Question'].apply(data_preprocessing_text)
   df['Answer'] = df['Answer'].apply(data_preprocessing_text)
   df['Category'] = df['Category'].apply(data_preprocessing_text)
   df['Title'] = df['Title'].apply(data_preprocessing_text)
   df.to_csv('data.csv', index=False, encoding='utf-8-sig')
   ```

2. **Initial Retrieval with BM25**:
   Retrieve the top 5 results for each query using BM25:

   ```python
   bm25_res = bm25_retr.transform(queriesDF)
   ```

3. **Re-ranking with MContriever**:
   Re-rank the BM25 results based on embedding similarity using MContriever:

   ```python
   score = []
   query_emb = MContriever(queries[doc])
   results = query_emb.get_embeddings()
   # Re-rank using the similarity score
   ```

4. **Evaluate the system**:
   Use the relevance judgments (`qrels.txt`) to evaluate the retrieval system:
   ```python
   bm25_eval = pt.Utils.evaluate(answer_after_rerank, qrels[['qid','docno','label']], metrics=["map","recip_rank","P"])
   print(bm25_eval)
   ```

## Future Improvements

- Incorporate other advanced ranking algorithms.
- Experiment with different embedding models like BERT or multilingual models.
- Implement more advanced re-ranking mechanisms and retrieval depth tuning.
