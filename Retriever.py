from sentence_transformers import CrossEncoder,SentenceTransformer
from elasticsearch import Elasticsearch
import pandas as pd

class Retriever:
    def __init__(self, embedding_model,cr_model, index_name='cord19', device='cpu'):
        self.model = embedding_model
        self.index_name = index_name
        self.client = Elasticsearch()
        self.cross_encoder = cr_model

    def get_knn_results(self, query, k=50):
        """
        Retrieves the K-nearest neighbors for a given query.

        Args:
            query (str): The query text.
            k (int, optional): The number of nearest neighbors to retrieve. Defaults to 50.

        Returns:
            pandas.DataFrame: A DataFrame containing the K-nearest neighbors.
        """
        query_vector = self.model.encode(query)
        response = self.client.search(index=self.index_name, body={
            "knn": {
                "field": "document_vector",
                "query_vector": query_vector,
                "k": k,
                "num_candidates": k,
                "boost": 1
            },
            "size": k,
            "_source": {"includes": ['document_id', 'document_title']}
        })
        output = []
        for hit in response['hits']['hits']:
            hit['_source']['score'] = hit['_score']
            output.append(hit['_source'])
        df = pd.DataFrame(output)
        return df

    def rerank_results(self, query, df):
        """
        Reranks the search results using a cross encoder.

        Args:
            query (str): The query text.
            df (pandas.DataFrame): The DataFrame containing search results.

        Returns:
            pandas.DataFrame: A DataFrame containing reranked search results.
        """
        # Create list of query-document pairs
        pairs = [(query, title) for title in df['document_title']]
        
        # Predict scores for query-document pairs
        scores = self.cross_encoder.predict(pairs)
        
        # Assign rerank scores to DataFrame
        df['rerank_score'] = scores
        
        # Sort DataFrame based on rerank scores
        df = df.sort_values(by='rerank_score', ascending=False).reset_index(drop=True)
        
        return df