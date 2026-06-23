# ============================================================================
# Task 4: Interactive Chat Interface - FIXED Streamlit Version
# CrediTrust Financial - Intelligent Complaint Analysis System
# ============================================================================

import streamlit as st
import pickle
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Complaint Analysis Chatbot",
    page_icon="",
    layout="wide"
)

# ============================================================================
# 1. LOAD SYSTEM (CACHED)
# ============================================================================

@st.cache_resource
def load_system():
    """Load the vector store and embedding model"""
    
    # Load embedding model
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    except:
        # Fallback to local model
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize
        
        class LocalEmbeddingModel:
            def __init__(self, dimension=384):
                self.dimension = dimension
                self.vectorizer = None
                self.svd = None
                self.is_fitted = False
                
            def encode(self, texts, normalize_embeddings=True):
                if isinstance(texts, str):
                    texts = [texts]
                
                if not self.is_fitted:
                    self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
                    self.vectorizer.fit(texts)
                    tfidf = self.vectorizer.transform(texts)
                    n_components = min(self.dimension, tfidf.shape[1] - 1)
                    self.svd = TruncatedSVD(n_components=n_components, random_state=42)
                    self.svd.fit(tfidf)
                    self.is_fitted = True
                
                tfidf = self.vectorizer.transform(texts)
                reduced = self.svd.transform(tfidf)
                
                if reduced.shape[1] < self.dimension:
                    padded = np.zeros((reduced.shape[0], self.dimension))
                    padded[:, :reduced.shape[1]] = reduced
                    reduced = padded
                
                if normalize_embeddings:
                    reduced = normalize(reduced, norm='l2')
                
                return reduced.astype(np.float32)
            
            def get_sentence_embedding_dimension(self):
                return self.dimension
        
        embedding_model = LocalEmbeddingModel()
    
    # Load FAISS index
    vector_paths = [
        'vector_store/faiss/index.faiss',
        '../vector_store/faiss/index.faiss',
        './vector_store/faiss/index.faiss'
    ]
    
    faiss_index = None
    metadata = None
    
    for path in vector_paths:
        if os.path.exists(path):
            faiss_index = faiss.read_index(path)
            metadata_path = path.replace('index.faiss', 'metadata.pkl')
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            break
    
    return embedding_model, faiss_index, metadata

# ============================================================================
# 2. IMPROVED RAG FUNCTIONS
# ============================================================================

def retrieve_chunks(query, index, metadata, model, k=5):
    """Retrieve relevant chunks with better similarity scoring"""
    query_embedding = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_embedding.astype('float32'), k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata['chunk_text']):
            # Better similarity calculation
            similarity = 1.0 / (1.0 + distances[0][i])
            
            # Get chunk text and clean it
            chunk_text = metadata['chunk_text'][idx]
            chunk_text = chunk_text.strip()
            if chunk_text.startswith('.'):
                chunk_text = chunk_text[1:].strip()
            
            results.append({
                'chunk_text': chunk_text,
                'complaint_id': metadata['complaint_id'][idx],
                'product': metadata['product'][idx],
                'issue': metadata['issue'][idx],
                'company': metadata['company'][idx],
                'distance': distances[0][i],
                'similarity': similarity
            })
    return results

def generate_answer(question, sources):
    """Generate better answer from sources"""
    if not sources:
        return "I don't have enough information to answer this question."
    
    # Extract relevant sentences from sources
    relevant_sentences = []
    question_words = question.lower().split()
    
    # Get top 3 sources
    for source in sources[:3]:
        text = source['chunk_text']
        sentences = text.split('.')
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 10:
                if any(word in sent.lower() for word in question_words[:5]):
                    relevant_sentences.append(sent)
    
    if relevant_sentences:
        # Remove duplicates
        seen = set()
        unique_sentences = []
        for sent in relevant_sentences:
            if sent not in seen:
                seen.add(sent)
                unique_sentences.append(sent)
        
        # Build response
        response = "Based on customer complaints, here's what I found:\n\n"
        
        products = set([s['product'] for s in sources[:3]])
        issues = set([s['issue'] for s in sources[:3]])
        
        if products:
            response += f"**Products mentioned:** {', '.join(products)}\n"
        if issues:
            response += f"**Issues mentioned:** {', '.join(issues)}\n"
        response += "\n"
        
        for i, sent in enumerate(unique_sentences[:3], 1):
            response += f"{i}. {sent}\n\n"
        
        return response
    else:
        return "I found complaint records related to this topic, but the specific information you're looking for isn't in the retrieved sources. Try rephrasing your question."

# ============================================================================
# 3. FIXED SOURCE DISPLAY (NO HTML)
# ============================================================================

def display_sources(sources):
    """Display sources using Streamlit native components (NO HTML)"""
    if not sources:
        st.info("No sources retrieved.")
        return
    
    st.markdown("#### 📚 Retrieved Sources:")
    
    for i, source in enumerate(sources[:5], 1):
        # Clean text
        text = source['chunk_text'][:250]
        if text.startswith('.'):
            text = text[1:].strip()
        
        sim_pct = source['similarity'] * 100
        
        # Use columns for better layout
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Source {i}:**")
            st.markdown(f"• **Product:** {source['product']}")
            st.markdown(f"• **Issue:** {source['issue']}")
            st.markdown(f"• **Preview:** \"{text}...\"")
        with col2:
            st.markdown(f"**Similarity:** {sim_pct:.1f}%")
        st.divider()

# ============================================================================
# 4. LOAD SYSTEM
# ============================================================================

with st.spinner("Loading complaint analysis system..."):
    embedding_model, faiss_index, metadata = load_system()

# ============================================================================
# 5. UI
# ============================================================================

st.title(" CrediTrust Financial - Complaint Analysis Chatbot")
st.markdown("Ask questions about customer complaints and get evidence-based answers with source attribution.")

# Sidebar
with st.sidebar:
    st.header(" Tips")
    st.markdown("""
    - Be specific in your questions
    - Ask about specific products
    - Use natural language
    
    **Example questions:**
    - What are the most common complaints about credit cards?
    - Why do customers complain about money transfers?
    - What issues do customers face with personal loans?
    - How do customers complain about billing disputes?
    """)
    
    st.header(" Statistics")
    st.metric("Total Vectors", f"{faiss_index.ntotal:,}")
    st.metric("Embedding Dimension", faiss_index.d)
    st.metric("Metadata Entries", f"{len(metadata['chunk_text']):,}")
    
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            display_sources(message["sources"])

# Chat input
if prompt := st.chat_input("Ask a question about customer complaints..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process question
    with st.chat_message("assistant"):
        with st.spinner("Analyzing complaints..."):
            chunks = retrieve_chunks(prompt, faiss_index, metadata, embedding_model, k=5)
            answer = generate_answer(prompt, chunks)
            
            st.markdown(answer)
            
            if chunks:
                display_sources(chunks)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": chunks
            })