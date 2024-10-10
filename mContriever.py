# Import necessary libraries
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import logging

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

class MContriever:
    def __init__(self, passage):
        """
        Initialize the MContriever model and tokenizer.
        
        Args:
            passage (str or list): Passage(s) to encode.
        """
        self.mcontriever_model_name = 'nthakur/mcontriever-base-msmarco'
        self.mcontriever_tokenizer = AutoTokenizer.from_pretrained(self.mcontriever_model_name)
        self.mcontriever_model = AutoModel.from_pretrained(self.mcontriever_model_name)
        self.passage = passage

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        """
        Apply mean pooling to obtain sentence embeddings from token embeddings.
        
        Args:
            model_output: The output of the transformer model containing token embeddings.
            attention_mask: The attention mask to apply to token embeddings.
            
        Returns:
            torch.Tensor: Sentence embeddings obtained through mean pooling.
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self):
        """
        Encode the passages and return the model, tokenizer, and embeddings.
        
        Returns:
            dict: Dictionary containing the MContriever model, tokenizer, and embeddings.
        """
        # Tokenize the input passages
        encoded_input = self.mcontriever_tokenizer(self.passage, padding=True, truncation=True, return_tensors='pt')
        
        # Compute embeddings without gradient calculation
        with torch.no_grad():
            mcontriever_model_output = self.mcontriever_model(**encoded_input)
        
        # Perform mean pooling on the model output
        mcontriever_embeddings = self.mean_pooling(mcontriever_model_output, encoded_input['attention_mask'])
        
        # Return model, tokenizer, and embeddings as a dictionary
        return {
            'mcontriever_model': self.mcontriever_model,
            'mcontriever_embeddings': mcontriever_embeddings,
            'mcontriever_tokenizer': self.mcontriever_tokenizer
        }

# Example usage:
# passages = ["Example passage 1", "Example passage 2"]
# embedding_extractor = MContriever(passages)
# results = embedding_extractor.get_embeddings()
# mcontriever_model = results['mcontriever_model']
# mcontriever_embeddings = results['mcontriever_embeddings']
# tokenizer = results['mcontriever_tokenizer']
