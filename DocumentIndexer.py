import numpy as np
import torch
import os
import pandas as pd
import time
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import models

class DocumentIndexer:
    """
    A class to handle document indexing and retrieval using Elasticsearch.

    Attributes:
        model_name (str): The name of the SentenceTransformer model to use.
        index_name (str): The name of the Elasticsearch index to create.
        device (str): The device to use for computation ('cuda' or 'cpu').
        model (SentenceTransformer): The SentenceTransformer model instance.
        index_name (str): The Elasticsearch index name.
        client (Elasticsearch): The Elasticsearch client instance.
        data (list): List of document titles for indexing.
        docs (list): List of dictionaries containing document information.
        encoded_data (numpy.ndarray): Encoded vectors of document titles.
    """

    def __init__(self, model_name='thenlper/gte-base', index_name='cord19', device='cuda'):
        """
        Initializes the DocumentIndexer object.

        Args:
            model_name (str, optional): The name of the SentenceTransformer model to use. Defaults to 'thenlper/gte-base'.
            index_name (str, optional): The name of the Elasticsearch index to create. Defaults to 'cord19'.
            device (str, optional): The device to use for computation ('cuda' or 'cpu'). Defaults to 'cuda'.
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.index_name = index_name
        self.client = Elasticsearch()

    def preprocess_data(self, data_path):
        """
        Preprocesses the data from a CSV file.

        Args:
            data_path (str): The path to the CSV file containing the data.
        """
        df = pd.read_csv(data_path)
        df = df.dropna()
        df = df.reset_index()
        df['len']=df['document_text'].str.split().str.len()
        df=df[df['len']<=300]
        df=df[0:10000]
        df = df.reset_index()
        df['document_title']=df['document_title']+' - '+df['document_text']
        self.data =(df['document_title']).to_list()
        self.docs = [{'document_id': df['document_id'][i], 'document_title': df['document_title'][i]} for i in range(df.shape[0])]

    def encode_data(self):
        """Encodes the document titles using the SentenceTransformer model."""
        start = time.time()
        self.encoded_data = self.model.encode(self.data, convert_to_numpy=True)
        end = time.time()
        print("Encoding time:", end - start)

    def create_index_mapping(self):
        """Creates the Elasticsearch index mapping."""
        source = {
            "mappings": {
                "properties": {
                    "document_vector": {
                        "type": "dense_vector",
                        "dims": 768,
                        "similarity": "dot_product"
                    },
                    "document_id": {
                        "type": "keyword"
                    },
                    "document_title": {
                        "type": "text"
                    },
                }
            }
        }
        print("Creating the index:", self.index_name)
        self.client.indices.delete(index=self.index_name, ignore=[404])
        self.client.indices.create(index=self.index_name, body=source)

    def index_documents(self):
        """Indexes the documents into Elasticsearch."""
        requests = []
        for i in range(len(self.docs)):
            request = self.docs[i]
            request["_op_type"] = "index"
            request["_index"] = self.index_name
            request["document_vector"] = [vector.tolist() for vector in self.encoded_data[i]]
            requests.append(request)
        bulk(self.client, requests)
        print("Documents indexed successfully.")

indexer = DocumentIndexer()
indexer.preprocess_data("data_for_indexing.csv")
indexer.encode_data()
indexer.create_index_mapping()
indexer.index_documents()