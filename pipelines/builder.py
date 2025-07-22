"""
RAG pipeline builder module for creating different pipeline patterns.
"""
from pathlib import Path
from typing import List, Dict
from core.types import (
    Document, BaseRetriever, BaseChatModel, ChatPromptTemplate,
    RunnablePassthrough, RunnableParallel, StrOutputParser
)
from factories.retriever import RetrieverFactory
from factories.generator import GeneratorFactory


class RAGPipelineBuilder:
    """Builder for different RAG pipeline patterns"""
    
    def __init__(self, retriever_factory: RetrieverFactory, generator_factory: GeneratorFactory):
        self.retriever_factory = retriever_factory
        self.generator_factory = generator_factory
        self.prompt_template = self._create_prompt_template()
        
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the RAG prompt template"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant tasked with answering questions based on the provided context.
            
Instructions:
- Answer the question using ONLY the information from the context provided
- If the context doesn't contain enough information, say so clearly
- Be concise but comprehensive in your response
- Quote relevant parts from the context when appropriate
- Maintain accuracy and avoid speculation
- Use bullet points for clarity when listing multiple items"""),
            
            ("human", """Context information:
{context}

Question: {question}

Please provide a clear and accurate answer based on the context above.""")
        ])
    
    def _format_documents(self, docs: List[Document]) -> str:
        """Format retrieved documents for the prompt"""
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            chunk = doc.metadata.get('chunk_index', 'N/A')
            
            formatted.append(
                f"[Document {i+1} - Source: {Path(source).name}, Page: {page}, Chunk: {chunk}]\n"
                f"{doc.page_content}\n"
            )
        return "\n".join(formatted)
    
    def build_linear_pipeline(self, retriever: BaseRetriever, generator: BaseChatModel):
        """Build a linear RAG pipeline"""
        return (
            {
                "context": retriever | self._format_documents,
                "question": RunnablePassthrough()
            }
            | self.prompt_template
            | generator
            | StrOutputParser()
        )
    
    def build_parallel_pipeline(self, retrievers: Dict[str, BaseRetriever], generator: BaseChatModel):
        """Build a parallel retrieval pipeline with fusion"""
        
        def fuse_results(results: Dict[str, List[Document]]) -> str:
            """Fuse results from multiple retrievers"""
            all_docs = []
            seen_content = set()
            
            for retriever_name, docs in results.items():
                for doc in docs:
                    # Deduplicate based on content
                    content_hash = hash(doc.page_content)
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        doc.metadata['retriever'] = retriever_name
                        all_docs.append(doc)
            
            return self._format_documents(all_docs)
        
        # Create parallel retrieval
        parallel_retrieval = RunnableParallel(
            {name: retriever for name, retriever in retrievers.items()}
        )
        
        return (
            {
                "context": parallel_retrieval | fuse_results,
                "question": RunnablePassthrough()
            }
            | self.prompt_template
            | generator
            | StrOutputParser()
        )
    