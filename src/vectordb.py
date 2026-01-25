"""Vector database wrapper using ChromaDB with HuggingFace embeddings."""
from typing import Any, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from chroma_client import ChromaDBClient
from config import CHUNK_OVERLAP_DEFAULT, CHUNK_SIZE_DEFAULT, COLLECTION_NAME_DEFAULT
from embeddings import initialize_embedding_model
from logger import logger


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME_DEFAULT,
        chunk_size: int = CHUNK_SIZE_DEFAULT,
        chunk_overlap: int = CHUNK_OVERLAP_DEFAULT,
    ):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            chunk_size: Approximate number of characters per chunk for text splitting
            chunk_overlap: Number of characters to overlap between chunks
        """

        # Initialize ChromaDB client and get or create collection
        self.collection = ChromaDBClient().get_or_create_collection(collection_name)
        logger.info("Vector database collection %s ready for use", self.collection.name)

        self.embedding_model = initialize_embedding_model()
        logger.info("Embedding model: %s", self.embedding_model.model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _chunk_documents(
        self, documents: list[str] | list[dict[str, str]]
    ) -> list[tuple[str, dict[str, str]]]:
        """
        Chunk documents into smaller pieces. Each chunk is paired with metadata.

        Args:
            documents: List of documents (strings or dicts with 'content',
                      'title', 'filename', and 'tags')

        Returns:
            List of tuples containing chunk text and metadata dictionary with
            title, filename, and tags
        """
        docs = documents if isinstance(documents, list) else [documents]
        chunks_with_metadata = [
            (
                chunk,
                {
                    "title": doc.get("title", "") if isinstance(doc, dict) else "",
                    "filename": (
                        doc.get("filename", "") if isinstance(doc, dict) else ""
                    ),
                    "tags": doc.get("tags", "") if isinstance(doc, dict) else "",
                },
            )
            for doc in docs
            for chunk in self.text_splitter.split_text(
                doc["content"] if isinstance(doc, dict) else doc
            )
            if chunk.strip()
            != (doc.get("title", "").strip() if isinstance(doc, dict) else "")
        ]
        return chunks_with_metadata

    def add_documents(self, documents: list[str] | list[dict[str, str]]) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents
        """
        chunks_with_metadata = self._chunk_documents(documents=documents)
        logger.info(
            "Created %d chunks from %d documents",
            len(chunks_with_metadata),
            len(documents),
        )
        self._insert_chunks_into_db(chunks_with_metadata)

    def _insert_chunks_into_db(self, chunks: list[tuple[str, dict]]) -> None:
        """Insert deduplicated chunks into the vector database."""
        deduplicated_chunks = self._filter_duplicate_chunks(chunks)

        if deduplicated_chunks:
            if len(deduplicated_chunks) < len(chunks):
                logger.info("Deduplicated to %d chunks", len(deduplicated_chunks))
            next_id = self.collection.count()
            keys = [
                f"document_{idx}"
                for idx in range(next_id, next_id + len(deduplicated_chunks))
            ]
            chunk_texts = [
                chunk.strip().lstrip(".,;:!? ") for chunk, _ in deduplicated_chunks
            ]
            metadata = [metadata for _, metadata in deduplicated_chunks]
            embeddings = self.embedding_model.embed_documents(chunk_texts)
            self.collection.add(
                ids=keys,
                embeddings=embeddings,
                documents=chunk_texts,
                metadatas=metadata,
            )
            logger.info(
                "Added %d chunks to the vector database.", len(deduplicated_chunks)
            )
        else:
            logger.warning("No new chunks to add (all are duplicates)")

    # ...existing code...

    def _filter_duplicate_chunks(
        self, chunks: list[tuple[str, dict]]
    ) -> list[tuple[str, dict]]:
        """
        Filter out duplicate chunks.
        Removes chunks that already exist in the database AND duplicates within the current batch.

        Args:
            chunks: List of tuples containing chunk text and metadata

        Returns:
            List of new chunks that don't already exist in the database or batch
        """
        # Get existing documents from the database
        existing_docs = self.collection.get()
        existing_texts = set(existing_docs.get("documents", []))

        # Filter out chunks that already exist in database
        # Normalize text the same way as in _insert_chunks_into_db
        new_chunks = [
            chunk
            for chunk in chunks
            if chunk[0].strip().lstrip(".,;:!? ") not in existing_texts
        ]

        # Also remove duplicates within the current batch
        seen = set()
        final_chunks = []
        for chunk_text, metadata in new_chunks:
            normalized_text = chunk_text.strip().lstrip(".,;:!? ")
            if normalized_text not in seen:
                seen.add(normalized_text)
                final_chunks.append((chunk_text, metadata))

        return final_chunks

    def _extract_search_results(self, results: dict) -> tuple:
        """Extract search result components from ChromaDB response."""

        def safe_get(key: str):
            """Safely extract nested list from results."""
            return results.get(key, [[]])[0] if results.get(key) else []

        return (
            safe_get("documents"),
            safe_get("metadatas"),
            safe_get("distances"),
            safe_get("ids"),
        )

    def search(
        self, query: str, n_results: int = 5
    ) -> Dict[str, Any]:  # pylint: disable=too-many-locals
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents',
            'metadatas', 'distances', 'ids'
        """
        query_embedding = self.embedding_model.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=n_results
        )

        documents, metadatas, distances, ids = self._extract_search_results(results)

        logger.debug("Search query: %s", query)
        logger.debug("Retrieved %d results", len(documents))

        for i, (doc_id, distance, metadata) in enumerate(
            zip(ids, distances, metadatas), 1
        ):
            cosine_similarity = 1 - distance
            title = metadata.get("title", "N/A")
            filename = metadata.get("filename", "N/A")
            logger.debug(
                "  Result %d: %s | Similarity: %.4f | Title: %s | File: %s",
                i,
                doc_id,
                cosine_similarity,
                title,
                filename,
            )

        return {
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
            "ids": ids,
        }
