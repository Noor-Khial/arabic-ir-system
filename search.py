# Import necessary libraries
import torch
import faiss
import pandas as pd
import numpy as np
import logging
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# Set up logging configuration
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

# Mean pooling function to calculate sentence embeddings
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Function to search query in the FAISS index
def search_query(model, num_passages, output_file):
    logging.info("Started uploading index")
    
    # Load FAISS index
    index = faiss.read_index("index_mContriever.bin")
    logging.info("Finished uploading index")
    
    # Load the test collection and database
    test_collection = 'test_collection.csv'
    df_test_collection = pd.read_csv(test_collection, encoding='utf-8-sig')
    database = "data.csv"
    def_database = pd.read_csv(database, encoding='utf-8-sig')
    
    # Extract queries and document identifiers
    queries = df_test_collection['Queries'].tolist()
    qids = df_test_collection['QID'].tolist()
    docnos = def_database['docno'].tolist()
    
    logging.info("Waiting for results from the index")

    data = []
    # Loop through each query and search in the index
    for q, query in enumerate(queries):
        if pd.notna(query):
            qid = qids[q]
            
            # Encode query using the SentenceTransformer model
            query_embeddings = model.encode([query])
            
            # Search in the FAISS index
            dists, ids = index.search(query_embeddings, k=num_passages)
            ids = np.array(ids)[0].copy()

            # Append results
            for d in range(len(ids)):
                docno = docnos[ids[d]]
                data.append([qid, docno, d, dists[0][d], query])

    df = pd.DataFrame(data, columns=["qid", "docno", "rank", "score", "query"])
    df.to_csv(output_file, index=False)

    logging.info("Fetched results successfully from the index")

def search():
    # Load SentenceTransformer model
    mcontriever_model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v1'
    model = SentenceTransformer(mcontriever_model_name)

    logging.info("Finished loading models ...")
    
    results_file = "results.csv"
    
    # Run the search query function
    search_query(model, num_passages=5, output_file=results_file)

if __name__ == "__main__":
    search()
