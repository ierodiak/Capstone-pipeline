"""
Vector store management module for creating, persisting, and loading vector stores.
"""
from pathlib import Path
from typing import List, Any
from IPython.display import display, Markdown

from core.types import Document, OpenAIEmbeddings, FAISS, Chroma
from core.config import RAGConfig


class VectorStoreManager:
    """Manages vector store creation, persistence, and loading"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.vector_store = None
        self.embeddings = None
        self._initialize_embeddings()
        
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        self.embeddings = OpenAIEmbeddings(
            model=self.config.retriever.model
        )
        print(f"Initialized embeddings with model: {self.config.retriever.model}")
        
    def create_vector_store(self, chunks: List[Document], persist: bool = True) -> Any:
        """Create vector store from chunks"""
        display(Markdown(f"### Creating {self.config.vector_store_type.upper()} Vector Store"))
        print(f"Processing {len(chunks)} chunks...")
        
        if self.config.vector_store_type == "faiss":
            self.vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
            
            if persist:
                self.save_vector_store()
                
        elif self.config.vector_store_type == "chroma":
            persist_dir = "./data/chroma_db" if persist else None
            
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=persist_dir,
                collection_name="rag_collection"
            )
            
        else:
            raise ValueError(f"Unknown vector store type: {self.config.vector_store_type}")
        
        print(f"Vector store created successfully")
        return self.vector_store
    
    def save_vector_store(self):
        """Save vector store to disk"""
        if self.config.vector_store_type == "faiss" and self.vector_store:
            save_path = "./data/faiss_index"
            self.vector_store.save_local(save_path)
            print(f"FAISS index saved to: {save_path}")
            
    def load_vector_store(self) -> Any:
        """Load vector store from disk"""
        if self.config.vector_store_type == "faiss":
            load_path = "./data/faiss_index"
            if Path(load_path).exists():
                self.vector_store = FAISS.load_local(
                    load_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"FAISS index loaded from: {load_path}")
            else:
                print(f"No saved index found at: {load_path}")
                
        elif self.config.vector_store_type == "chroma":
            persist_dir = "./data/chroma_db"
            if Path(persist_dir).exists():
                self.vector_store = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=self.embeddings,
                    collection_name="rag_collection"
                )
                print(f"Chroma database loaded from: {persist_dir}")
            else:
                print(f"No saved database found at: {persist_dir}")
                
        return self.vector_store
    
    def test_retrieval(self, query: str, k: int = 3):
        """Test retrieval with a sample query"""
        if not self.vector_store:
            print("No vector store available")
            return
        
        display(Markdown(f"### Testing Retrieval"))
        print(f"Query: '{query}'")
        print(f"Retrieving top {k} documents...\n")
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        for i, (doc, score) in enumerate(results):
            display(Markdown(f"#### Result {i+1} (Score: {score:.4f})"))
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Page: {doc.metadata.get('page', 'N/A')}")
            print(f"Chunk: {doc.metadata.get('chunk_index', 'N/A')}")
            print(f"Content: {doc.page_content[:200]}...")
            print()