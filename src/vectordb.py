import os
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from chroma_client import ChromaDBClient
from config import CHUNK_SIZE_DEFAULT, CHUNK_OVERLAP_DEFAULT
from embeddings import initialize_embedding_model


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, chunk_size: int = CHUNK_SIZE_DEFAULT, chunk_overlap: int = CHUNK_OVERLAP_DEFAULT):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            chunk_size: Approximate number of characters per chunk for text splitting
            chunk_overlap: Number of characters to overlap between chunks
        """

        # Initialize ChromaDB client and get or create collection
        self.collection = ChromaDBClient().get_or_create_collection(
            collection_name or os.getenv(
                "CHROMA_COLLECTION_NAME", "rag_documents"
            )
        )

        # Initialize embedding model with device detection
        self.embedding_model = initialize_embedding_model()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print(f"Vector database initialized with collection: {self.collection.name}")

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks using recursive character splitting.
        Uses the instance chunk_size and chunk_overlap configuration for optimal text splitting.

        Args:
            text: Input text to chunk
        Returns:
            List of text chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return text_splitter.split_text(text)

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents
        """
        # TODO: Implement document ingestion logic
        # HINT: Loop through each document in the documents list
        # HINT: Extract 'content' and 'metadata' from each document dict
        # HINT: Use self.chunk_text() to split each document into chunks
        # HINT: Create unique IDs for each chunk (e.g., "doc_0_chunk_0")
        # HINT: Use self.embedding_model.encode() to create embeddings for all chunks
        # HINT: Store the embeddings, documents, metadata, and IDs in your vector database
        # HINT: Print progress messages to inform the user

        print(f"Processing {len(documents)} documents...")
        # Your implementation here
        print("Documents added to vector database")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        # TODO: Implement similarity search logic
        # HINT: Use self.embedding_model.encode([query]) to create query embedding
        # HINT: Convert the embedding to appropriate format for your vector database
        # HINT: Use your vector database's search/query method with the query embedding and n_results
        # HINT: Return a dictionary with keys: 'documents', 'metadatas', 'distances', 'ids'
        # HINT: Handle the case where results might be empty

        # Your implementation here
        return {
            "documents": [],
            "metadatas": [],
            "distances": [],
            "ids": [],
        }
