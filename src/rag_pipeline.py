# src/rag_pipeline.py
"""
RAG Pipeline Module for Complaint Analysis System
"""

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGPipeline:
    def __init__(self, embedding_model=None, faiss_index=None, metadata=None):
        """Initialize the RAG pipeline with vector store and embedding model"""
        self.embedding_model = embedding_model
        self.index = faiss_index
        self.metadata = metadata
    
    def retrieve(self, query, k=5):
        """Retrieve relevant chunks for a query"""
        if self.embedding_model is None or self.index is None:
            return []
            
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata['chunk_text']):
                results.append({
                    'chunk_text': self.metadata['chunk_text'][idx],
                    'complaint_id': self.metadata['complaint_id'][idx],
                    'product': self.metadata['product'][idx],
                    'issue': self.metadata['issue'][idx],
                    'company': self.metadata['company'][idx],
                    'distance': distances[0][i],
                    'similarity': 1 / (1 + distances[0][i])
                })
        return results
    
    def query(self, question, k=5):
        """Complete RAG query"""
        sources = self.retrieve(question, k)
        return {
            'question': question,
            'sources': sources,
            'num_sources': len(sources)
        }