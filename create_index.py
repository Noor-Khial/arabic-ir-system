from mContriever import MContriever
import faiss

def retrieve_passages_faiss(sentence_embeddings):
    """
    Retrieve passages using Faiss index.

    Args:
    - sentence_embeddings: Sentence embeddings for the passages.

    Returns:
    - Faiss index.
    """
    X = sentence_embeddings.numpy()
    #X = sentence_embeddings.copy()
    index = faiss.IndexFlatL2(len(X[0]))
    
    index.add(X)
    return index

test_collection = 'test_collection.csv'
embedding_extractor = MContriever(test_collection)
results = embedding_extractor.get_embeddings()
mcontriever_model = results['mcontriever_model']
mcontriever_embeddings = results['mcontriever_embeddings']
print("done with model ")
index = retrieve_passages_faiss(mcontriever_embeddings)
print("done with creating index")
faiss.write_index(index, "index_mcontriever.bin")
